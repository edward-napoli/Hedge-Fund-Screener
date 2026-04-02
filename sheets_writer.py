"""
sheets_writer.py — Google Sheets writing, formatting, summary, analytics, and retention.

Uses gspread for cell operations and the underlying Sheets API v4
(via gspread's spreadsheet.batch_update) for advanced conditional formatting.

Tab strategy
------------
- Daily data tab: named "YYYY-MM-DD" (e.g. "2026-03-10")
  - Morning run  → clear tab, write header + data rows
  - Afternoon run → append separator row + header + data below morning section
- Summary tab     : always updated to reflect the most recent run's top 25
- Analytics tab   : persistent; rebuilt after every run with 4 data sections
- 30-day retention: tabs with date names older than 30 calendar days are deleted

Delta highlighting (in daily tab data rows)
-------------------------------------------
- Rank Delta < -5  (improved by more than 5 places) → green row background
- Rank Delta >  5  (worsened by more than 5 places)  → red   row background
"""

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from config import (
    COLUMNS,
    CREDENTIALS_FILE,
    HEADER_BG_COLOR,
    HEADER_TEXT_COLOR,
    ROW_COLOR_ODD,
    ROW_COLOR_EVEN,
    GOLD_COLOR,
    GREEN_COLOR,
    RED_COLOR,
    WHITE_COLOR,
)

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column name constants
_SCORE_COL_NAME      = "Composite Score"
_RANK_COL_NAME       = "Rank"
_SCORE_DELTA_COL     = "Score Delta"
_RANK_DELTA_COL      = "Rank Delta"

# How many calendar days of daily tabs to retain
_RETENTION_DAYS = 30

# Pattern for recognising dated tab names ("YYYY-MM-DD")
_DATE_TAB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Permanent tabs that are NEVER deleted by _delete_old_tabs
_PERMANENT_TABS = {"Summary", "Analytics", "Price History", "Score History", "Efficacy Analysis", "Backtest Results"}

RUN_HISTORY_FILE = "run_history.json"


# ---------------------------------------------------------------------------
# Auth & spreadsheet helpers
# ---------------------------------------------------------------------------

def _get_client(credentials_file: str = CREDENTIALS_FILE) -> gspread.Client:
    """Authenticate and return a gspread client."""
    creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
    return gspread.authorize(creds)


def _rgb(color_dict: dict) -> dict:
    """Return a Sheets API color object from a normalised RGB dict."""
    return {
        "red":   color_dict["red"],
        "green": color_dict["green"],
        "blue":  color_dict["blue"],
    }


def _col_idx(col_name: str) -> int:
    """Return 0-based column index for a named column in COLUMNS."""
    return COLUMNS.index(col_name)


def get_or_create_spreadsheet(
    client: gspread.Client,
    sheet_id: Optional[str],
) -> gspread.Spreadsheet:
    """
    Open an existing spreadsheet by ID (from .env) or create a new one.

    Parameters
    ----------
    client : gspread.Client
        Authenticated gspread client.
    sheet_id : str or None
        Google Sheets document ID from GOOGLE_SHEET_ID env var.

    Returns
    -------
    gspread.Spreadsheet
    """
    if sheet_id:
        try:
            ss = client.open_by_key(sheet_id)
            logger.info(f"Opened existing spreadsheet: {ss.url}")
            return ss
        except gspread.exceptions.SpreadsheetNotFound:
            logger.warning("Sheet ID not found — creating a new spreadsheet.")

    ss = client.create("Hedge Fund Stock Screener")
    ss.share(None, perm_type="anyone", role="writer")
    logger.info(f"Created new spreadsheet: {ss.url}")
    return ss


def _get_or_create_worksheet(
    ss: gspread.Spreadsheet,
    title: str,
    rows: int = 3000,
    cols: int = 30,
) -> gspread.Worksheet:
    """
    Return an existing worksheet by title, or create a new one.

    Parameters
    ----------
    ss : gspread.Spreadsheet
    title : str
        Tab name.
    rows, cols : int
        Initial dimensions if the tab must be created.

    Returns
    -------
    gspread.Worksheet
    """
    try:
        return ss.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=rows, cols=cols)


# ---------------------------------------------------------------------------
# Column-index helpers derived from COLUMNS
# ---------------------------------------------------------------------------

def _num_cols() -> int:
    """Total number of columns."""
    return len(COLUMNS)


def _rank_col_idx() -> int:
    """0-based index of the Rank column."""
    return _col_idx(_RANK_COL_NAME)


def _score_col_idx() -> int:
    """0-based index of the Composite Score column."""
    return _col_idx(_SCORE_COL_NAME)


def _rank_delta_col_idx() -> int:
    """0-based index of the Rank Delta column."""
    return _col_idx(_RANK_DELTA_COL)


# ---------------------------------------------------------------------------
# Batch-update request builders (pure functions returning request dicts)
# ---------------------------------------------------------------------------

def _freeze_header_request(sheet_id: int) -> dict:
    """Freeze the first row of a sheet."""
    return {
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    }


def _header_format_request(sheet_id: int, num_cols: int, start_row: int = 0) -> dict:
    """Bold, white-text, dark-navy header row at start_row (0-based)."""
    return {
        "repeatCell": {
            "range": {
                "sheetId":        sheet_id,
                "startRowIndex":  start_row,
                "endRowIndex":    start_row + 1,
                "startColumnIndex": 0,
                "endColumnIndex": num_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": _rgb(HEADER_BG_COLOR),
                    "textFormat": {
                        "bold": True,
                        "foregroundColor": _rgb(HEADER_TEXT_COLOR),
                        "fontSize": 10,
                    },
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment":   "MIDDLE",
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }
    }


def _row_bg_request(sheet_id: int, row_index: int, color: dict, num_cols: int) -> dict:
    """Set the background colour of a single data row (0-based row_index)."""
    return {
        "repeatCell": {
            "range": {
                "sheetId":          sheet_id,
                "startRowIndex":    row_index,
                "endRowIndex":      row_index + 1,
                "startColumnIndex": 0,
                "endColumnIndex":   num_cols,
            },
            "cell": {
                "userEnteredFormat": {"backgroundColor": _rgb(color)}
            },
            "fields": "userEnteredFormat.backgroundColor",
        }
    }


def _auto_resize_request(sheet_id: int, num_cols: int) -> dict:
    """Auto-resize all columns."""
    return {
        "autoResizeDimensions": {
            "dimensions": {
                "sheetId":    sheet_id,
                "dimension":  "COLUMNS",
                "startIndex": 0,
                "endIndex":   num_cols,
            }
        }
    }


def _number_format_request(
    sheet_id: int,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    pattern: str,
) -> dict:
    """Apply a number format pattern to a cell range."""
    return {
        "repeatCell": {
            "range": {
                "sheetId":          sheet_id,
                "startRowIndex":    start_row,
                "endRowIndex":      end_row,
                "startColumnIndex": start_col,
                "endColumnIndex":   end_col,
            },
            "cell": {
                "userEnteredFormat": {
                    "numberFormat": {"type": "NUMBER", "pattern": pattern}
                }
            },
            "fields": "userEnteredFormat.numberFormat",
        }
    }


def _bold_text_request(
    sheet_id: int,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    bold: bool = True,
    font_size: int = 11,
) -> dict:
    """Apply bold text formatting to a cell range."""
    return {
        "repeatCell": {
            "range": {
                "sheetId":          sheet_id,
                "startRowIndex":    start_row,
                "endRowIndex":      end_row,
                "startColumnIndex": start_col,
                "endColumnIndex":   end_col,
            },
            "cell": {
                "userEnteredFormat": {
                    "textFormat": {"bold": bold, "fontSize": font_size}
                }
            },
            "fields": "userEnteredFormat.textFormat",
        }
    }


# ---------------------------------------------------------------------------
# Conditional formatting request builders
# ---------------------------------------------------------------------------

def _score_gradient_cf(sheet_id: int, start_row: int, end_row: int, score_col: int) -> dict:
    """Red-white-green gradient on Composite Score column."""
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId":          sheet_id,
                    "startRowIndex":    start_row,
                    "endRowIndex":      end_row,
                    "startColumnIndex": score_col,
                    "endColumnIndex":   score_col + 1,
                }],
                "gradientRule": {
                    "minpoint": {
                        "color": _rgb(RED_COLOR),
                        "type":  "PERCENTILE",
                        "value": "0",
                    },
                    "midpoint": {
                        "color": _rgb(WHITE_COLOR),
                        "type":  "PERCENTILE",
                        "value": "50",
                    },
                    "maxpoint": {
                        "color": _rgb(GREEN_COLOR),
                        "type":  "PERCENTILE",
                        "value": "100",
                    },
                },
            },
            "index": 0,
        }
    }


def _rank_gold_cf(
    sheet_id: int,
    start_row: int,
    end_row: int,
    num_cols: int,
    rank_col: int,
) -> dict:
    """Gold background + bold for rows where Rank <= 10 (formula-based)."""
    # Sheets column letters (A=0, B=1, …)
    col_letter = _col_index_to_letter(rank_col)
    # The formula anchors the rank column but the row reference adjusts per row.
    formula = f"=${col_letter}{start_row + 1}<=10"
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId":          sheet_id,
                    "startRowIndex":    start_row,
                    "endRowIndex":      end_row,
                    "startColumnIndex": 0,
                    "endColumnIndex":   num_cols,
                }],
                "booleanRule": {
                    "condition": {
                        "type":   "CUSTOM_FORMULA",
                        "values": [{"userEnteredValue": formula}],
                    },
                    "format": {
                        "backgroundColor": _rgb(GOLD_COLOR),
                        "textFormat":      {"bold": True},
                    },
                },
            },
            "index": 1,
        }
    }


def _col_index_to_letter(col_index: int) -> str:
    """
    Convert a 0-based column index to a Sheets column letter (A, B, …, Z, AA, …).

    Parameters
    ----------
    col_index : int
        0-based column index.

    Returns
    -------
    str
        Column letter string.
    """
    result = ""
    n = col_index + 1  # 1-based
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


# ---------------------------------------------------------------------------
# Delta row highlighting
# ---------------------------------------------------------------------------

def _apply_delta_highlighting(
    requests: list,
    sheet_id: int,
    df: pd.DataFrame,
    data_start_row: int,
    num_cols: int,
) -> None:
    """
    Append row-level background-colour requests for rows with |Rank Delta| > 5.

    - Rank Delta < -5 (rank improved by > 5 places) → green background
    - Rank Delta >  5 (rank worsened by > 5 places)  → red   background

    The alternating base colour is overwritten by the highlight, so this must
    be called AFTER the alternating-row requests are appended.

    Parameters
    ----------
    requests : list
        The in-progress batch_update requests list (mutated in place).
    sheet_id : int
        gspread worksheet sheet_id.
    df : pd.DataFrame
        The ranked DataFrame (used to read Rank Delta values).
    data_start_row : int
        0-based Sheets row index of the first data row (row after the header).
    num_cols : int
        Total number of columns.
    """
    if _RANK_DELTA_COL not in df.columns:
        return

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            rd = int(row[_RANK_DELTA_COL])
        except (TypeError, ValueError):
            continue

        sheet_row = data_start_row + i

        if rd < -5:
            # Improved significantly — green highlight
            requests.append(_row_bg_request(sheet_id, sheet_row, GREEN_COLOR, num_cols))
        elif rd > 5:
            # Worsened significantly — red highlight
            requests.append(_row_bg_request(sheet_id, sheet_row, RED_COLOR, num_cols))


# ---------------------------------------------------------------------------
# Main data writer
# ---------------------------------------------------------------------------

def _build_data_rows(df: pd.DataFrame) -> list:
    """
    Convert the DataFrame to a list of lists for Sheets upload.

    Only columns present in COLUMNS are included, in COLUMNS order.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[list]
        Each inner list is one data row.
    """
    present_cols = [c for c in COLUMNS if c in df.columns]
    return df[present_cols].values.tolist()


def _apply_main_formatting(
    ss: gspread.Spreadsheet,
    ws: gspread.Worksheet,
    df: pd.DataFrame,
    header_row: int,
    data_start_row: int,
    data_end_row: int,
) -> None:
    """
    Apply all batch formatting to a data section (header + data rows).

    Parameters
    ----------
    ss : gspread.Spreadsheet
    ws : gspread.Worksheet
    df : pd.DataFrame
        The data written to the section (used for delta highlighting).
    header_row : int
        0-based row index of the header row.
    data_start_row : int
        0-based row index of the first data row.
    data_end_row : int
        0-based row index AFTER the last data row (exclusive).
    """
    sheet_id  = ws.id
    num_cols  = _num_cols()
    num_data  = data_end_row - data_start_row
    rank_col  = _rank_col_idx()
    score_col = _score_col_idx()

    batch_reqs: list = []

    # Freeze header row (only applied once; harmless to repeat)
    batch_reqs.append(_freeze_header_request(sheet_id))

    # Header formatting
    batch_reqs.append(_header_format_request(sheet_id, num_cols, header_row))

    # Alternating row background colours
    for i in range(num_data):
        color     = ROW_COLOR_ODD if i % 2 == 0 else ROW_COLOR_EVEN
        sheet_row = data_start_row + i
        batch_reqs.append(_row_bg_request(sheet_id, sheet_row, color, num_cols))

    # Delta highlighting overwrites alternating colour where |Rank Delta| > 5
    _apply_delta_highlighting(batch_reqs, sheet_id, df, data_start_row, num_cols)

    # Auto-resize columns
    batch_reqs.append(_auto_resize_request(sheet_id, num_cols))

    # Number format: 2 decimal places for numeric columns (col 6 onwards)
    first_numeric = 6
    last_rank     = num_cols - 1   # Rank column index

    if first_numeric < num_cols:
        batch_reqs.append(_number_format_request(
            sheet_id,
            data_start_row, data_end_row,
            first_numeric, last_rank,
            "#,##0.00",
        ))

    # Rank column: integer format
    batch_reqs.append(_number_format_request(
        sheet_id,
        data_start_row, data_end_row,
        last_rank, last_rank + 1,
        "0",
    ))

    # Rank Delta column: integer format (may be negative)
    rank_delta_col = _rank_delta_col_idx()
    batch_reqs.append(_number_format_request(
        sheet_id,
        data_start_row, data_end_row,
        rank_delta_col, rank_delta_col + 1,
        "+0;-0;0",
    ))

    ss.batch_update({"requests": batch_reqs})

    # Conditional formatting (separate batch — some CF rules can't mix with plain formats)
    cf_reqs: list = []
    cf_reqs.append(_score_gradient_cf(sheet_id, data_start_row, data_end_row, score_col))
    cf_reqs.append(_rank_gold_cf(sheet_id, data_start_row, data_end_row, num_cols, rank_col))

    try:
        ss.batch_update({"requests": cf_reqs})
    except Exception as exc:
        logger.warning(f"Conditional formatting batch partially failed: {exc}")


def write_main_sheet(
    ss: gspread.Spreadsheet,
    df: pd.DataFrame,
    sheet_title: str,
    run_type: str = "morning",
) -> gspread.Worksheet:
    """
    Write the ranked stock DataFrame to a dated daily worksheet.

    Morning run
    -----------
    Clears the tab, writes header row at row 1, then data rows.

    Afternoon run
    -------------
    Appends to the existing morning data:
    1. A separator row with the afternoon run timestamp.
    2. Another header row.
    3. Afternoon data rows.
    All afternoon rows are formatted identically to the morning section.

    Parameters
    ----------
    ss : gspread.Spreadsheet
    df : pd.DataFrame
        Fully scored and ranked DataFrame.
    sheet_title : str
        Tab name (expected format: "YYYY-MM-DD").
    run_type : str
        "morning" or "afternoon".

    Returns
    -------
    gspread.Worksheet
    """
    num_cols  = _num_cols()
    ws = _get_or_create_worksheet(ss, sheet_title, rows=len(df) * 2 + 50, cols=num_cols + 2)

    header    = [c for c in COLUMNS]   # list copy for safety
    data_rows = _build_data_rows(df)

    if run_type == "morning":
        # ---- Morning: clear + write fresh ---------------------------------
        ws.clear()
        all_rows   = [header] + data_rows
        ws.update("A1", all_rows, value_input_option="USER_ENTERED")

        _apply_main_formatting(
            ss, ws, df,
            header_row=0,
            data_start_row=1,
            data_end_row=1 + len(data_rows),
        )
        logger.info(f"[MORNING] Tab '{sheet_title}' written with {len(data_rows)} rows.")

    else:
        # ---- Afternoon: append separator + header + data -----------------
        # Find where to start appending (next empty row after existing content)
        existing = ws.get_all_values()
        append_at = len(existing)  # 0-based row index of next empty row

        now_utc   = datetime.utcnow().strftime("%H:%M UTC")
        separator = [f"--- AFTERNOON RUN — {now_utc} ---"] + [""] * (num_cols - 1)

        # Write separator, header, then data rows
        section_rows = [separator, header] + data_rows
        start_cell   = f"A{append_at + 1}"   # Sheets is 1-based
        ws.update(start_cell, section_rows, value_input_option="USER_ENTERED")

        # Separator row formatting (bold, light grey background)
        separator_row_idx = append_at          # 0-based
        header_row_idx    = append_at + 1      # 0-based
        data_start_idx    = append_at + 2      # 0-based
        data_end_idx      = append_at + 2 + len(data_rows)   # exclusive

        sheet_id = ws.id
        sep_reqs = [
            _row_bg_request(
                sheet_id, separator_row_idx,
                {"red": 0.85, "green": 0.85, "blue": 0.85},
                num_cols,
            ),
            _bold_text_request(sheet_id, separator_row_idx, separator_row_idx + 1, 0, num_cols),
        ]
        ss.batch_update({"requests": sep_reqs})

        _apply_main_formatting(
            ss, ws, df,
            header_row=header_row_idx,
            data_start_row=data_start_idx,
            data_end_row=data_end_idx,
        )
        logger.info(
            f"[AFTERNOON] Tab '{sheet_title}' appended with {len(data_rows)} rows "
            f"(starting at row {append_at + 1})."
        )

    return ws


# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------

def write_summary_sheet(
    ss: gspread.Spreadsheet,
    df: pd.DataFrame,
    total_found: int,
    total_skipped: int,
    elapsed_seconds: float,
    run_type: str = "morning",
    is_partial: bool = False,
    top25_changes: Optional[dict] = None,
    regime_label: Optional[str] = None,
    weighting_scheme: Optional[str] = None,
    regime_filter_on: Optional[bool] = None,
) -> gspread.Worksheet:
    """
    Create / update the 'Summary' worksheet with key statistics and top-25 changes.

    Always reflects the most recent run's data (cleared and rewritten each time).

    Parameters
    ----------
    ss : gspread.Spreadsheet
    df : pd.DataFrame
        Most recent ranked DataFrame.
    total_found : int
    total_skipped : int
    elapsed_seconds : float
    run_type : str
        "morning" or "afternoon".
    is_partial : bool
        True if fewer stocks were fetched than the threshold.
    top25_changes : dict or None
        Output of delta_tracker.get_top25_changes() — for entered/exited lists.

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Summary", rows=200, cols=10)
    ws.clear()

    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_score = df["Composite Score"].mean() if not df.empty else 0.0
    top25     = df.head(25)[["Ticker", "Company Name", "Composite Score", "Rank"]]

    # Sector distribution in top 100
    top100        = df.head(100)
    sector_counts = (
        top100["Sector"].value_counts()
        .reset_index()
        .rename(columns={"index": "Sector", "Sector": "Count"})
    )

    partial_label = " [PARTIAL RUN]" if is_partial else ""
    run_label     = run_type.upper()

    rows: list = []
    rows.append([f"HEDGE FUND STOCK SCREENER — SUMMARY ({run_label}{partial_label})"])
    rows.append([])
    rows.append(["Last Run:",            now_str])
    rows.append(["Run Type:",            run_label])
    rows.append(["Stocks Screened:",     total_found])
    rows.append(["Stocks Skipped:",      total_skipped])
    rows.append(["Stocks Ranked:",       len(df)])
    rows.append(["Avg Composite Score:", round(avg_score, 2)])
    rows.append(["Elapsed Time (s):",    round(elapsed_seconds, 1)])
    if regime_label is not None:
        rows.append(["Market Regime:",   regime_label])
    if weighting_scheme is not None:
        rows.append(["Weighting Scheme:", weighting_scheme.replace("_", " ").title()])
    if regime_filter_on is not None:
        rows.append(["Regime Filter:",   "ON" if regime_filter_on else "OFF"])
    rows.append([])

    # Top-25 changes section
    if top25_changes:
        entered = top25_changes.get("entered", [])
        exited  = top25_changes.get("exited",  [])
        if entered or exited:
            rows.append(["TOP 25 CHANGES"])
            if entered:
                rows.append(["  Entered ↑:", ", ".join(entered)])
            if exited:
                rows.append(["  Exited ↓:", ", ".join(exited)])
            rows.append([])

    header_rows_indices = [0]   # 0-based row indices for bold formatting
    top25_header_idx    = len(rows)
    rows.append(["TOP 25 STOCKS BY COMPOSITE SCORE"])
    rows.append(["Rank", "Ticker", "Company Name", "Composite Score"])
    for _, r in top25.iterrows():
        rows.append([r["Rank"], r["Ticker"], r["Company Name"], r["Composite Score"]])

    rows.append([])
    sector_header_idx = len(rows)
    rows.append(["SECTOR DISTRIBUTION — TOP 100 STOCKS"])
    rows.append(["Sector", "Count"])
    for _, r in sector_counts.iterrows():
        rows.append([r.iloc[0], r.iloc[1]])

    ws.update("A1", rows, value_input_option="USER_ENTERED")

    # Bold the section titles
    sheet_id = ws.id
    bold_reqs = []
    for bi in [0, top25_header_idx, sector_header_idx]:
        bold_reqs.append(_bold_text_request(sheet_id, bi, bi + 1, 0, 5, bold=True, font_size=11))

    bold_reqs.append({
        "autoResizeDimensions": {
            "dimensions": {
                "sheetId":    sheet_id,
                "dimension":  "COLUMNS",
                "startIndex": 0,
                "endIndex":   6,
            }
        }
    })
    ss.batch_update({"requests": bold_reqs})

    logger.info("Summary sheet written.")
    return ws


# ---------------------------------------------------------------------------
# 30-day tab retention
# ---------------------------------------------------------------------------

def _delete_old_tabs(ss: gspread.Spreadsheet) -> None:
    """
    Delete dated daily tabs (YYYY-MM-DD) older than 30 calendar days.

    Parameters
    ----------
    ss : gspread.Spreadsheet
    """
    cutoff = datetime.utcnow() - timedelta(days=_RETENTION_DAYS)
    to_delete: list = []

    for ws in ss.worksheets():
        if ws.title in _PERMANENT_TABS:
            continue  # never delete permanent tabs
        if _DATE_TAB_RE.match(ws.title):
            try:
                tab_date = datetime.strptime(ws.title, "%Y-%m-%d")
                if tab_date < cutoff:
                    to_delete.append(ws)
            except ValueError:
                pass  # title matched regex but couldn't parse — skip

    for ws in to_delete:
        try:
            ss.del_worksheet(ws)
            logger.info(f"Deleted old tab: '{ws.title}' (older than {_RETENTION_DAYS} days).")
        except Exception as exc:
            logger.warning(f"Could not delete tab '{ws.title}': {exc}")


# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------

def _load_run_history() -> list:
    """
    Load run_history.json.

    Returns
    -------
    list[dict]
        List of run metadata dicts, or empty list if file not found / unreadable.
    """
    if not os.path.exists(RUN_HISTORY_FILE):
        return []
    try:
        with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not read run_history.json: {exc}")
        return []


def write_analytics_tab(
    ss: gspread.Spreadsheet,
    df: pd.DataFrame,
) -> gspread.Worksheet:
    """
    Create / update the persistent 'Analytics' worksheet.

    Sections
    --------
    1. 30-Day Rolling Average Score
       Source: run_history.json — one row per run with date + avg score.
    2. Sector Breakdown
       Source: current run's df — average composite score per sector.
    3. Most Consistent Top 25
       Source: run_history.json — tickers appearing most in top 25 across all runs.
    4. Biggest Movers (5-run)
       Source: run_history.json — tickers with largest cumulative rank change
       over the last 5 runs (based on rank position in top10 list of each run).

    Parameters
    ----------
    ss : gspread.Spreadsheet
    df : pd.DataFrame
        Current run's ranked DataFrame (for live sector data).

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Analytics", rows=500, cols=10)
    ws.clear()

    history = _load_run_history()
    rows: list = []

    # -----------------------------------------------------------------------
    # Section 1: 30-Day Rolling Average Score
    # -----------------------------------------------------------------------
    rows.append(["ANALYTICS — STOCK SCREENER"])
    rows.append([])
    rows.append(["SECTION 1: 30-DAY ROLLING AVERAGE COMPOSITE SCORE"])
    rows.append(["Timestamp", "Run Type", "Avg Score", "Stocks Ranked"])

    cutoff_dt = datetime.utcnow() - timedelta(days=30)
    for entry in history:
        ts_str = entry.get("timestamp", "")
        try:
            # Timestamps stored as "YYYY-MM-DD HH:MM:SS UTC"
            ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff_dt:
            continue

        top10   = entry.get("top10", [])
        scores  = [float(s["score"]) for s in top10 if "score" in s]
        avg_s   = round(sum(scores) / len(scores), 2) if scores else "N/A"
        ranked  = entry.get("stocks_fetched", "N/A")
        run_t   = entry.get("run_type", "N/A")
        rows.append([ts_str, run_t, avg_s, ranked])

    rows.append([])

    # -----------------------------------------------------------------------
    # Section 2: Sector Breakdown (current run)
    # -----------------------------------------------------------------------
    rows.append(["SECTION 2: SECTOR BREAKDOWN (CURRENT RUN)"])
    rows.append(["Sector", "Stock Count", "Avg Composite Score"])

    if "Sector" in df.columns and "Composite Score" in df.columns:
        sector_df = (
            df.groupby("Sector")["Composite Score"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "Count", "mean": "Avg Score"})
            .sort_values("Avg Score", ascending=False)
        )
        for _, r in sector_df.iterrows():
            rows.append([r["Sector"], int(r["Count"]), round(float(r["Avg Score"]), 2)])
    else:
        rows.append(["(No sector data available)", "", ""])

    rows.append([])

    # -----------------------------------------------------------------------
    # Section 3: Most Consistent Top 25
    # -----------------------------------------------------------------------
    rows.append(["SECTION 3: MOST CONSISTENT TOP 25 APPEARANCES"])
    rows.append(["Ticker", "Appearances in Top 25", "Runs Analysed"])

    ticker_appearances: Counter = Counter()
    runs_with_top10 = 0

    for entry in history:
        top10 = entry.get("top10", [])
        if not top10:
            continue
        runs_with_top10 += 1
        for stock in top10:
            t = stock.get("ticker", "")
            if t:
                ticker_appearances[t] += 1

    rows.append(["(Note: based on top-10 per run stored in run_history.json)", "", ""])
    for ticker, count in ticker_appearances.most_common(25):
        rows.append([ticker, count, runs_with_top10])

    rows.append([])

    # -----------------------------------------------------------------------
    # Section 4: Biggest Movers (last 5 runs)
    # -----------------------------------------------------------------------
    rows.append(["SECTION 4: BIGGEST MOVERS — LAST 5 RUNS (CUMULATIVE RANK CHANGE)"])
    rows.append(["Ticker", "Position in Run 1 (oldest)", "Position in Run 5 (latest)", "Change"])

    recent_runs = [e for e in history if e.get("top10")][-5:]

    if len(recent_runs) >= 2:
        first_run  = recent_runs[0].get("top10", [])
        latest_run = recent_runs[-1].get("top10", [])

        first_pos  = {s["ticker"]: i + 1 for i, s in enumerate(first_run)}
        latest_pos = {s["ticker"]: i + 1 for i, s in enumerate(latest_run)}

        movers: list = []
        all_tickers = set(first_pos.keys()) | set(latest_pos.keys())
        for t in all_tickers:
            p1 = first_pos.get(t, 11)   # outside top-10 = 11
            p2 = latest_pos.get(t, 11)
            movers.append((t, p1, p2, p1 - p2))   # positive = improved

        movers.sort(key=lambda x: abs(x[3]), reverse=True)
        for t, p1, p2, delta in movers[:20]:
            rows.append([t, p1, p2, delta])
    else:
        rows.append(["(Need at least 2 runs with top10 data)", "", "", ""])

    # Write all rows
    ws.update("A1", rows, value_input_option="USER_ENTERED")

    # Bold section headers
    sheet_id = ws.id
    section_title_rows = [0, 3 + max(0, len(history)), ]   # approximate; bold specific ones below
    # Identify section header row indices from what we built
    section_rows_indices: list = []
    for i, r in enumerate(rows):
        if r and isinstance(r[0], str) and r[0].startswith("SECTION"):
            section_rows_indices.append(i)
    section_rows_indices.insert(0, 0)  # Title row

    bold_reqs = []
    for ri in section_rows_indices:
        bold_reqs.append(_bold_text_request(sheet_id, ri, ri + 1, 0, 5, bold=True, font_size=11))

    bold_reqs.append({
        "autoResizeDimensions": {
            "dimensions": {
                "sheetId":    sheet_id,
                "dimension":  "COLUMNS",
                "startIndex": 0,
                "endIndex":   6,
            }
        }
    })

    try:
        ss.batch_update({"requests": bold_reqs})
    except Exception as exc:
        logger.warning(f"Analytics tab formatting partially failed: {exc}")

    logger.info("Analytics tab written.")
    return ws


# ---------------------------------------------------------------------------
# Price History tab
# ---------------------------------------------------------------------------

def update_price_history_tab(
    ss: gspread.Spreadsheet,
    price_history: dict,
    score_history: dict,
) -> gspread.Worksheet:
    """
    Create / update the permanent 'Price History' worksheet.

    Layout
    ------
    - Row 1 : Header — ["Ticker", <date1>, <date2>, ...] (dates ascending)
    - For each ticker row: price_usd per date (or blank if unavailable)
    - Row after each price row: percentage change from previous date's price
      (positive → green text, negative → red text)
    - First column (Ticker) is frozen.

    Parameters
    ----------
    ss : gspread.Spreadsheet
    price_history : dict
        Mapping date_str → {ticker → price_data_dict}.
    score_history : dict
        Mapping date_str → {stocks: {ticker → {score, rank}}}.

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Price History", rows=5000, cols=200)
    ws.clear()

    sorted_dates = sorted(price_history.keys())
    if not sorted_dates:
        ws.update("A1", [["Price History — no data yet."]])
        return ws

    # Collect all tickers (union across all dates)
    all_tickers: set = set()
    for date_str in sorted_dates:
        all_tickers.update(price_history[date_str].keys())
    all_tickers_sorted = sorted(all_tickers)

    # Header row
    header = ["Ticker"] + sorted_dates
    all_rows = [header]

    # Per-ticker rows + pct-change rows
    for ticker in all_tickers_sorted:
        price_row = [ticker]
        pct_row   = [""]   # blank ticker cell for the pct-change row

        prev_price = None
        for date_str in sorted_dates:
            day_data   = price_history[date_str].get(ticker, {})
            price_usd  = day_data.get("price_usd") if isinstance(day_data, dict) else None

            price_row.append(round(price_usd, 4) if price_usd is not None else "")

            if price_usd is not None and prev_price is not None and prev_price != 0:
                pct = round((price_usd - prev_price) / prev_price * 100, 2)
                pct_row.append(pct)
            else:
                pct_row.append("")

            prev_price = price_usd if price_usd is not None else prev_price

        all_rows.append(price_row)
        all_rows.append(pct_row)

    ws.update("A1", all_rows, value_input_option="USER_ENTERED")

    # ---- Formatting ----
    sheet_id  = ws.id
    num_cols  = len(header)
    num_tickers = len(all_tickers_sorted)

    batch_reqs: list = []

    # Freeze header row + first column
    batch_reqs.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "frozenRowCount": 1,
                    "frozenColumnCount": 1,
                },
            },
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
        }
    })

    # Header formatting
    batch_reqs.append(_header_format_request(sheet_id, num_cols, start_row=0))

    # Auto-resize
    batch_reqs.append(_auto_resize_request(sheet_id, num_cols))

    # Colour the pct-change rows: positive → green text, negative → red text
    # pct rows are at 0-based row indices: 2, 4, 6, ... (row 1 = header, row 2 = first price, row 3 = first pct, ...)
    pct_row_requests: list = []
    for i in range(num_tickers):
        pct_row_idx = 1 + i * 2 + 1  # 0-based: header=0, then price+pct pairs
        # We use a conditional formatting rule per cell column range for pct rows
        # Apply green text for positive values
        pct_row_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId":          sheet_id,
                        "startRowIndex":    pct_row_idx,
                        "endRowIndex":      pct_row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex":   num_cols,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type":   "NUMBER_GREATER",
                            "values": [{"userEnteredValue": "0"}],
                        },
                        "format": {
                            "textFormat": {"foregroundColor": _rgb(GREEN_COLOR)},
                        },
                    },
                },
                "index": 0,
            }
        })
        # Apply red text for negative values
        pct_row_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId":          sheet_id,
                        "startRowIndex":    pct_row_idx,
                        "endRowIndex":      pct_row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex":   num_cols,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type":   "NUMBER_LESS",
                            "values": [{"userEnteredValue": "0"}],
                        },
                        "format": {
                            "textFormat": {"foregroundColor": _rgb(RED_COLOR)},
                        },
                    },
                },
                "index": 0,
            }
        })

    try:
        ss.batch_update({"requests": batch_reqs})
    except Exception as exc:
        logger.warning(f"Price History tab base formatting failed: {exc}")

    if pct_row_requests:
        # Send CF rules in batches of 100 to avoid API limits
        chunk_size = 100
        for start in range(0, len(pct_row_requests), chunk_size):
            chunk = pct_row_requests[start: start + chunk_size]
            try:
                ss.batch_update({"requests": chunk})
            except Exception as exc:
                logger.warning(f"Price History pct-row CF batch failed: {exc}")

    logger.info(
        f"Price History tab updated: {len(all_tickers_sorted)} tickers, "
        f"{len(sorted_dates)} dates."
    )
    return ws


# ---------------------------------------------------------------------------
# Score History tab
# ---------------------------------------------------------------------------

def update_score_history_tab(
    ss: gspread.Spreadsheet,
    score_history: dict,
    price_history: dict,
) -> gspread.Worksheet:
    """
    Create / update the permanent 'Score History' worksheet.

    Layout
    ------
    - Row 1: Header — ["Ticker", <date1_score>, <date1_rank>, <date2_score>, ...]
      (paired Score | Rank columns per date, dates ascending)
    - For each ticker row: Clayton Score and Rank per date
    - Row after each stock row: absolute score change and rank change from previous date
      (score/rank improvements → green text, deteriorations → red text)
    - First column (Ticker) is frozen.

    Parameters
    ----------
    ss : gspread.Spreadsheet
    score_history : dict
        Mapping date_str → {run_type, timestamp, stocks: {ticker → {score, rank}}}.
    price_history : dict
        Not used for data, but accepted for API consistency.

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Score History", rows=5000, cols=200)
    ws.clear()

    sorted_dates = sorted(score_history.keys())
    if not sorted_dates:
        ws.update("A1", [["Score History — no data yet."]])
        return ws

    # Collect all tickers
    all_tickers: set = set()
    for date_str in sorted_dates:
        all_tickers.update(score_history[date_str].get("stocks", {}).keys())
    all_tickers_sorted = sorted(all_tickers)

    # Header: Ticker, then pairs of Score/Rank columns per date
    header = ["Ticker"]
    for date_str in sorted_dates:
        header.append(f"{date_str} Score")
        header.append(f"{date_str} Rank")

    all_rows = [header]

    for ticker in all_tickers_sorted:
        score_row = [ticker]
        delta_row = [""]

        prev_score = None
        prev_rank  = None
        for date_str in sorted_dates:
            stocks = score_history[date_str].get("stocks", {})
            data   = stocks.get(ticker)
            if data:
                score = data.get("score")
                rank  = data.get("rank")
                score_row.append(round(score, 2) if score is not None else "")
                score_row.append(rank if rank is not None else "")

                # Delta vs previous date
                if prev_score is not None and score is not None:
                    delta_row.append(round(score - prev_score, 2))
                else:
                    delta_row.append("")
                if prev_rank is not None and rank is not None:
                    delta_row.append(rank - prev_rank)
                else:
                    delta_row.append("")

                prev_score = score
                prev_rank  = rank
            else:
                score_row.extend(["", ""])
                delta_row.extend(["", ""])

        all_rows.append(score_row)
        all_rows.append(delta_row)

    ws.update("A1", all_rows, value_input_option="USER_ENTERED")

    # ---- Formatting ----
    sheet_id  = ws.id
    num_cols  = len(header)
    num_tickers = len(all_tickers_sorted)

    batch_reqs: list = []

    # Freeze header row + first column
    batch_reqs.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "frozenRowCount": 1,
                    "frozenColumnCount": 1,
                },
            },
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
        }
    })

    # Header formatting
    batch_reqs.append(_header_format_request(sheet_id, num_cols, start_row=0))

    # Auto-resize
    batch_reqs.append(_auto_resize_request(sheet_id, num_cols))

    try:
        ss.batch_update({"requests": batch_reqs})
    except Exception as exc:
        logger.warning(f"Score History tab base formatting failed: {exc}")

    # Conditional formatting for delta rows (green = improvement, red = deterioration)
    cf_reqs: list = []
    for i in range(num_tickers):
        delta_row_idx = 1 + i * 2 + 1  # 0-based
        # Score delta: positive is good (score went up)
        cf_reqs.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId":          sheet_id,
                        "startRowIndex":    delta_row_idx,
                        "endRowIndex":      delta_row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex":   num_cols,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type":   "NUMBER_GREATER",
                            "values": [{"userEnteredValue": "0"}],
                        },
                        "format": {"textFormat": {"foregroundColor": _rgb(GREEN_COLOR)}},
                    },
                },
                "index": 0,
            }
        })
        cf_reqs.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId":          sheet_id,
                        "startRowIndex":    delta_row_idx,
                        "endRowIndex":      delta_row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex":   num_cols,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type":   "NUMBER_LESS",
                            "values": [{"userEnteredValue": "0"}],
                        },
                        "format": {"textFormat": {"foregroundColor": _rgb(RED_COLOR)}},
                    },
                },
                "index": 0,
            }
        })

    chunk_size = 100
    for start in range(0, len(cf_reqs), chunk_size):
        chunk = cf_reqs[start: start + chunk_size]
        try:
            ss.batch_update({"requests": chunk})
        except Exception as exc:
            logger.warning(f"Score History CF batch failed: {exc}")

    logger.info(
        f"Score History tab updated: {len(all_tickers_sorted)} tickers, "
        f"{len(sorted_dates)} dates."
    )
    return ws


# ---------------------------------------------------------------------------
# Efficacy Analysis tab
# ---------------------------------------------------------------------------

def write_efficacy_tab(
    ss: gspread.Spreadsheet,
    metrics: dict,
    trading_days: int,
) -> gspread.Worksheet:
    """
    Create / update the permanent 'Efficacy Analysis' worksheet.

    Writes a human-readable report of Clayton Score predictive power across
    five sections:
    1. Portfolio-level correlation (Score → Forward Returns)
    2. Quintile performance analysis
    3. Rolling 30-day correlation (Score vs 5D return)
    4. Top-25 entry validation
    5. Per-stock correlation (top 30 by |r| for 5D return)

    Parameters
    ----------
    ss : gspread.Spreadsheet
    metrics : dict
        Output of efficacy_analyzer compute_* functions, merged into one dict.
    trading_days : int
        Number of trading days analysed.

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Efficacy Analysis", rows=500, cols=20)
    ws.clear()

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def _fmt_r(v) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    def _fmt_p(v) -> str:
        if v is None:
            return "N/A"
        return f"{v:.6f}"

    def _sig(p) -> str:
        if p is None:
            return "—"
        return "YES (p<0.05)" if p < 0.05 else "NO"

    def _fmt_pct(v) -> str:
        return f"{v:.4f}%" if v is not None else "N/A"

    rows: list = []

    # Title
    rows.append(["EFFICACY ANALYSIS — CLAYTON SCORE PREDICTIVE POWER"])
    rows.append([f"Data as of: {now_str}  |  Trading days analysed: {trading_days}"])
    rows.append([])

    # Section 1: Portfolio-level correlation
    rows.append(["SECTION 1: PORTFOLIO-LEVEL CORRELATION (Clayton Score → Forward Returns)"])
    rows.append(["Forward Period", "Pearson r", "p-value", "Significant?"])
    for period, r_key, p_key in [
        ("1-Day",  "portfolio_1d",  "p_value_1d"),
        ("5-Day",  "portfolio_5d",  "p_value_5d"),
        ("21-Day", "portfolio_21d", "p_value_21d"),
    ]:
        r = metrics.get(r_key)
        p = metrics.get(p_key)
        rows.append([period, _fmt_r(r), _fmt_p(p), _sig(p)])
    rows.append([])

    # Section 2: Quintile performance
    rows.append(["SECTION 2: QUINTILE PERFORMANCE ANALYSIS"])
    rows.append(["Quintile", "Avg 1D Return", "Avg 5D Return", "Avg 21D Return"])
    quintile_labels = [
        "Q1 (Lowest 20%)",
        "Q2",
        "Q3",
        "Q4",
        "Q5 (Highest 20%)",
    ]
    q1d  = metrics.get("quintile_avg_1d",  [None] * 5)
    q5d  = metrics.get("quintile_avg_5d",  [None] * 5)
    q21d = metrics.get("quintile_avg_21d", [None] * 5)
    for i, label in enumerate(quintile_labels):
        v1  = q1d[i]  if i < len(q1d)  else None
        v5  = q5d[i]  if i < len(q5d)  else None
        v21 = q21d[i] if i < len(q21d) else None
        rows.append([label, _fmt_pct(v1), _fmt_pct(v5), _fmt_pct(v21)])

    # Q5 - Q1 spread
    rows.append([
        "Q5 − Q1 Spread",
        _fmt_pct(metrics.get("q5_minus_q1_1d")),
        _fmt_pct(metrics.get("q5_minus_q1_5d")),
        _fmt_pct(metrics.get("q5_minus_q1_21d")),
    ])
    rows.append([])

    # Section 3: Rolling correlation
    rows.append(["SECTION 3: ROLLING 30-DAY CORRELATION (Score vs 5D Return)"])
    rows.append(["Date", "Correlation"])
    rolling = metrics.get("rolling_correlation", [])
    if rolling:
        for entry in rolling:
            rows.append([entry.get("date", ""), _fmt_r(entry.get("correlation"))])
    else:
        rows.append(["(No rolling data yet)", ""])
    rows.append([])

    # Section 4: Top-25 validation
    rows.append(["SECTION 4: TOP-25 ENTRY VALIDATION"])
    rows.append(["Metric", "1-Day", "5-Day", "21-Day"])
    rows.append([
        "Avg Top-25 Entry Return",
        _fmt_pct(metrics.get("top25_avg_top25_1d")),
        _fmt_pct(metrics.get("top25_avg_top25_5d")),
        _fmt_pct(metrics.get("top25_avg_top25_21d")),
    ])
    rows.append([
        "Avg All-Stock Return",
        _fmt_pct(metrics.get("top25_avg_all_1d")),
        _fmt_pct(metrics.get("top25_avg_all_5d")),
        _fmt_pct(metrics.get("top25_avg_all_21d")),
    ])
    # Outperformance
    def _diff(a, b):
        if a is not None and b is not None:
            return round(a - b, 4)
        return None

    rows.append([
        "Outperformance",
        _fmt_pct(_diff(metrics.get("top25_avg_top25_1d"), metrics.get("top25_avg_all_1d"))),
        _fmt_pct(_diff(metrics.get("top25_avg_top25_5d"), metrics.get("top25_avg_all_5d"))),
        _fmt_pct(_diff(metrics.get("top25_avg_top25_21d"), metrics.get("top25_avg_all_21d"))),
    ])
    event_count = metrics.get("top25_event_count", 0)
    rows.append([f"Events analysed: {event_count}", "", "", ""])
    rows.append([])

    # Section 5: Per-stock correlation (top 30 by |5D r|)
    rows.append(["SECTION 5: PER-STOCK CORRELATION (Top 30 by |r| for 5D return)"])
    rows.append(["Ticker", "1D r", "5D r", "21D r"])
    per_5d = metrics.get("per_stock_5d", {})
    per_1d = metrics.get("per_stock_1d", {})
    per_21d = metrics.get("per_stock_21d", {})
    sorted_tickers = sorted(per_5d.keys(), key=lambda t: abs(per_5d[t]), reverse=True)[:30]
    if sorted_tickers:
        for ticker in sorted_tickers:
            rows.append([
                ticker,
                _fmt_r(per_1d.get(ticker)),
                _fmt_r(per_5d.get(ticker)),
                _fmt_r(per_21d.get(ticker)),
            ])
    else:
        rows.append(["(No per-stock data yet)", "", "", ""])

    ws.update("A1", rows, value_input_option="USER_ENTERED")

    # Bold section title rows
    sheet_id = ws.id
    bold_indices = [i for i, r in enumerate(rows) if r and isinstance(r[0], str) and
                    (r[0].startswith("SECTION") or r[0].startswith("EFFICACY"))]
    bold_reqs = [
        _bold_text_request(sheet_id, ri, ri + 1, 0, 10, bold=True, font_size=11)
        for ri in bold_indices
    ]
    bold_reqs.append({
        "autoResizeDimensions": {
            "dimensions": {
                "sheetId":    sheet_id,
                "dimension":  "COLUMNS",
                "startIndex": 0,
                "endIndex":   10,
            }
        }
    })
    try:
        ss.batch_update({"requests": bold_reqs})
    except Exception as exc:
        logger.warning(f"Efficacy tab formatting failed: {exc}")

    logger.info("Efficacy Analysis tab written.")
    return ws


def write_efficacy_tab_insufficient(
    ss: gspread.Spreadsheet,
    days_collected: int = 0,
) -> gspread.Worksheet:
    """
    Write a placeholder message to the 'Efficacy Analysis' tab when there is
    insufficient data (fewer than 21 trading days).

    Parameters
    ----------
    ss : gspread.Spreadsheet
    days_collected : int
        Number of trading days of data currently available.

    Returns
    -------
    gspread.Worksheet
    """
    ws = _get_or_create_worksheet(ss, "Efficacy Analysis", rows=10, cols=5)
    ws.clear()
    ws.update(
        "A1",
        [[
            f"Insufficient data — efficacy analysis will populate after "
            f"{21} trading days. "
            f"Currently: {days_collected} day(s) of data collected."
        ]],
    )
    logger.info("Efficacy Analysis placeholder written (insufficient data).")
    return ws


# ---------------------------------------------------------------------------
# Backtest Results tab
# ---------------------------------------------------------------------------

def write_backtest_tab(
    ss: gspread.Spreadsheet,
    backtest_results:   dict | None = None,
    wf_results:         dict | None = None,
    stress_results:     dict | None = None,
    factor_results:     dict | None = None,
    comparison_results: dict | None = None,
) -> gspread.Worksheet:
    """
    Write a 'Backtest Results' permanent tab with sections for:
      - Strategy performance overview (vs SPY benchmark)
      - Walk-forward optimisation summary
      - Stress period table
      - Monte Carlo summary
      - TC sensitivity table
      - Top-N sensitivity table
      - Factor contribution (LOO) table
      - Factor IC table

    Parameters
    ----------
    ss : gspread.Spreadsheet
    backtest_results : dict from backtest.run_backtest()
    wf_results : dict from weight_optimizer.walk_forward_optimise()
    stress_results : dict from stress_test.run_all_stress_tests()
    factor_results : dict from factor_analysis.run_factor_analysis()

    Returns
    -------
    gspread.Worksheet
    """
    import pandas as pd

    ws = _get_or_create_worksheet(ss, "Backtest Results", rows=500, cols=20)
    ws.clear()

    all_rows: list[list] = []

    def _section_header(title: str) -> None:
        all_rows.append([title])
        all_rows.append([])

    def _row(*cells) -> None:
        all_rows.append(list(cells))

    def _fmt(v, decimals: int = 2) -> str:
        """Format a numeric value to N decimal places, or return 'n/a'."""
        if v is None or v == "n/a":
            return "n/a"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    # ── Section 0: 4-Variant Comparison (risk-parity / regime filter) ─────────
    if comparison_results:
        _section_header("STRATEGY VARIANT COMPARISON")
        _row("Variant", "CAGR %", "Sharpe", "Sortino", "Max DD %", "Calmar", "Win Rate %", "Volatility %", "Periods")
        for variant_key, res in comparison_results.items():
            m  = res.get("metrics", {})
            _row(
                variant_key,
                _fmt(m.get("cagr")),
                _fmt(m.get("sharpe"), 3),
                _fmt(m.get("sortino"), 3),
                _fmt(m.get("max_drawdown")),
                _fmt(m.get("calmar"), 3),
                _fmt(m.get("win_rate")),
                _fmt(m.get("volatility")),
                m.get("n_periods", ""),
            )
        all_rows.append([])

    # ── Section 1: Strategy Performance ──────────────────────────────────────
    _section_header("STRATEGY PERFORMANCE")

    if backtest_results and "metrics" in backtest_results:
        m  = backtest_results["metrics"]
        bm = backtest_results.get("benchmark_metrics", {})
        p  = backtest_results.get("params", {})

        _row("Parameter", "Value")
        _row("Period", f"{p.get('start','?')} → {p.get('end','?')}")
        _row("Rebalancing", p.get("freq", "?"))
        _row("Portfolio", f"Top {p.get('top_n','?')} | {'Long-Short' if p.get('long_short') else 'Long-Only'}")
        _row("Transaction Cost (bps)", p.get("tc_bps", "?"))
        all_rows.append([])

        _row("Metric", "Strategy", "SPY Benchmark")
        for label, key in [
            ("CAGR (%)",           "cagr"),
            ("Total Return (%)",   "total_return"),
            ("Sharpe Ratio",       "sharpe"),
            ("Sortino Ratio",      "sortino"),
            ("Calmar Ratio",       "calmar"),
            ("Max Drawdown (%)",   "max_drawdown"),
            ("Volatility (%)",     "volatility"),
            ("Win Rate (%)",       "win_rate"),
            ("Best Period (%)",    "best_period"),
            ("Worst Period (%)",   "worst_period"),
            ("Periods",            "n_periods"),
        ]:
            decimals = 0 if key == "n_periods" else 2
            _row(label, _fmt(m.get(key), decimals), _fmt(bm.get(key), decimals))

        # Relative metrics vs SPY benchmark
        rel = backtest_results.get("relative_metrics", {})
        if rel:
            all_rows.append([])
            _row("VS SPY BENCHMARK")
            _row("Metric", "Value")
            _row("Excess CAGR (%)",      _fmt(rel.get("excess_cagr")))
            _row("Beta",                 _fmt(rel.get("beta"), 4))
            _row("Jensen Alpha (ann %)", _fmt(rel.get("alpha")))
            _row("Correlation",          _fmt(rel.get("correlation"), 4))
            _row("Information Ratio",    _fmt(rel.get("information_ratio"), 4))

        # Regime breakdown
        regime_m = backtest_results.get("regime_metrics", {})
        if regime_m:
            all_rows.append([])
            _row("Regime", "CAGR (%)", "Sharpe", "Periods")
            for regime, rm in regime_m.items():
                _row(
                    regime.capitalize(),
                    _fmt(rm.get("cagr")),
                    _fmt(rm.get("sharpe"), 3),
                    rm.get("n_periods", "n/a"),
                )

        # Year-by-year breakdown derived from portfolio_values and benchmark_returns
        pv_raw = backtest_results.get("portfolio_values")
        if pv_raw is not None:
            # Normalise to {date: float} dict regardless of whether it's a pd.Series or dict
            if hasattr(pv_raw, "items"):
                pv_dict = {
                    (k.date() if hasattr(k, "date") else k): float(v)
                    for k, v in pv_raw.items()
                }
            else:
                pv_dict = {}

            # Build year-end NAV: take the last recorded NAV in each calendar year
            year_end_nav: dict[int, float] = {}
            for dt, nav in sorted(pv_dict.items(), key=lambda x: str(x[0])):
                yr = int(str(dt)[:4])
                year_end_nav[yr] = nav  # keep overwriting → last entry wins

            # Build SPY year returns from benchmark_returns Series if available
            bench_rets_raw = backtest_results.get("benchmark_returns")
            spy_year_ret: dict[int, float] = {}
            if bench_rets_raw is not None and hasattr(bench_rets_raw, "items"):
                import math
                yr_compound: dict[int, float] = {}
                for dt, r in bench_rets_raw.items():
                    yr = int(str(dt)[:4])
                    if yr not in yr_compound:
                        yr_compound[yr] = 1.0
                    yr_compound[yr] *= (1.0 + float(r))
                spy_year_ret = {yr: (v - 1.0) * 100 for yr, v in yr_compound.items()}

            if year_end_nav:
                all_rows.append([])
                _row("YEAR-BY-YEAR RETURNS")
                _row("Year", "Strategy Return (%)", "SPY Return (%)")
                prev_nav = None
                for yr in sorted(year_end_nav):
                    nav = year_end_nav[yr]
                    ret_strat = ((nav / prev_nav) - 1) * 100 if prev_nav is not None else "n/a"
                    ret_spy   = spy_year_ret.get(yr)
                    _row(
                        yr,
                        _fmt(ret_strat) if ret_strat != "n/a" else "n/a",
                        _fmt(ret_spy) if ret_spy is not None else "n/a",
                    )
                    prev_nav = nav

    else:
        _row("No backtest results available. Run python main.py --backtest first.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 2: Walk-Forward Optimisation ──────────────────────────────────
    _section_header("WALK-FORWARD OPTIMISATION")

    if wf_results and "windows" in wf_results:
        p = wf_results.get("params", {})
        _row("Train years", p.get("train_years"), "Test years", p.get("test_years"))
        all_rows.append([])

        _row("Window", "Train Sharpe", "Test Sharpe (Optimised)", "Test Sharpe (Default)")
        for w in wf_results["windows"]:
            label = f"{w['train_start'][:7]}→{w['test_end'][:7]}"
            _row(label, w.get("train_sharpe"), w.get("test_sharpe"), w.get("default_test_sharpe"))

        oos = wf_results.get("oos_combined_metrics", {})
        if oos:
            all_rows.append([])
            _row("OOS Combined", f"CAGR={oos.get('cagr','?')}%", f"Sharpe={oos.get('sharpe','?')}", f"MaxDD={oos.get('max_drawdown','?')}%")

        avg_w = wf_results.get("average_optimal_weights", {})
        if avg_w:
            all_rows.append([])
            _row("Weight Key", "Default", "Optimised")
            from config import WEIGHTS as DW
            for k in avg_w:
                _row(k, DW.get(k, 0), avg_w[k])
    else:
        _row("No walk-forward results. Run python main.py --optimize first.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 3: Stress Periods ─────────────────────────────────────────────
    _section_header("HISTORICAL STRESS PERIODS")

    sp_df = (stress_results or {}).get("stress_periods")
    if sp_df is not None and not sp_df.empty:
        _row(*sp_df.columns.tolist())
        for _, row in sp_df.iterrows():
            _row(*[str(v) for v in row.tolist()])
    else:
        _row("No stress period results. Run python main.py --stress-test first.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 4: Monte Carlo ────────────────────────────────────────────────
    _section_header("MONTE CARLO SIMULATION")

    mc = (stress_results or {}).get("monte_carlo", {})
    if mc:
        for label, key in [
            ("Runs",                    "n_runs"),
            ("Base Sharpe",             "base_sharpe"),
            ("Mean Sharpe",             "mean_sharpe"),
            ("Median Sharpe",           "median_sharpe"),
            ("5th Percentile Sharpe",   "p5_sharpe"),
            ("95th Percentile Sharpe",  "p95_sharpe"),
            ("% Positive Sharpe",       "pct_positive_sharpe"),
        ]:
            _row(label, mc.get(key, "n/a"))
    else:
        _row("No Monte Carlo results.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 5: TC Sensitivity ─────────────────────────────────────────────
    _section_header("TRANSACTION COST SENSITIVITY")

    tc_df = (stress_results or {}).get("tc_sensitivity")
    if tc_df is not None and not tc_df.empty:
        _row(*tc_df.columns.tolist())
        for _, row in tc_df.iterrows():
            _row(*[str(round(v, 3)) if isinstance(v, float) else str(v) for v in row.tolist()])
    else:
        _row("No TC sensitivity results.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 6: Top-N Sensitivity ──────────────────────────────────────────
    _section_header("TOP-N PORTFOLIO SIZE SENSITIVITY")

    tn_df = (stress_results or {}).get("topn_sensitivity")
    if tn_df is not None and not tn_df.empty:
        _row(*tn_df.columns.tolist())
        for _, row in tn_df.iterrows():
            _row(*[str(round(v, 3)) if isinstance(v, float) else str(v) for v in row.tolist()])
    else:
        _row("No top-N sensitivity results.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 7: Factor LOO Contribution ────────────────────────────────────
    _section_header("FACTOR LEAVE-ONE-OUT CONTRIBUTION")

    loo_df = (factor_results or {}).get("loo_contribution")
    if loo_df is not None and not loo_df.empty:
        _row(*loo_df.columns.tolist())
        for _, row in loo_df.iterrows():
            _row(*[str(round(v, 3)) if isinstance(v, float) else str(v) for v in row.tolist()])
    else:
        _row("No LOO results. Run python main.py --factor-analysis first.")

    all_rows.append([])
    all_rows.append([])

    # ── Section 8: Factor IC Table ────────────────────────────────────────────
    _section_header("FACTOR INFORMATION COEFFICIENTS (IC)")

    ic_df = (factor_results or {}).get("ic_table")
    if ic_df is not None and not ic_df.empty:
        # Show IC columns only
        ic_cols = [c for c in ic_df.columns if c.endswith("_IC")]
        _row("Factor", *ic_cols)
        for factor_name in ic_df.index:
            row_vals = [ic_df.at[factor_name, c] for c in ic_cols]
            _row(factor_name, *[round(v, 4) if isinstance(v, float) else "n/a" for v in row_vals])
    else:
        _row("No IC results. Run python main.py --factor-analysis first.")

    all_rows.append([])
    _row(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}")

    # ── Write to sheet ────────────────────────────────────────────────────────
    # Ensure we have enough rows
    if len(all_rows) > ws.row_count:
        ws.add_rows(len(all_rows) - ws.row_count + 10)

    ws.update("A1", all_rows, value_input_option="USER_ENTERED")

    # Apply section header formatting (bold, dark background)
    section_header_rows = []
    for i, row in enumerate(all_rows, start=1):
        if row and isinstance(row[0], str) and row[0].isupper() and len(row) == 1:
            section_header_rows.append(i)

    if section_header_rows:
        sheet_id_int = ws.id
        fmt_requests = []
        for r in section_header_rows:
            fmt_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id_int,
                        "startRowIndex": r - 1,
                        "endRowIndex": r,
                        "startColumnIndex": 0,
                        "endColumnIndex": 10,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": HEADER_BG_COLOR,
                            "textFormat": {
                                "foregroundColor": HEADER_TEXT_COLOR,
                                "bold": True,
                                "fontSize": 11,
                            },
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            })
        if fmt_requests:
            ss.batch_update({"requests": fmt_requests})

    logger.info(f"Backtest Results tab updated ({len(all_rows)} rows).")
    return ws


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_to_sheets(
    df: pd.DataFrame,
    sheet_id: Optional[str],
    total_found: int,
    total_skipped: int,
    elapsed_seconds: float,
    credentials_file: str = CREDENTIALS_FILE,
    run_type: str = "morning",
    is_partial: bool = False,
    top25_changes: Optional[dict] = None,
    regime_label: Optional[str] = None,
    weighting_scheme: Optional[str] = None,
    regime_filter_on: Optional[bool] = None,
) -> str:
    """
    Write ranked DataFrame to Google Sheets and return the spreadsheet URL.

    Performs the following in order:
    1. Open / create the spreadsheet.
    2. Write the daily dated tab (morning: fresh; afternoon: append section).
    3. Update the Summary tab.
    4. Update the Analytics tab.
    5. Delete dated tabs older than 30 calendar days.

    Parameters
    ----------
    df : pd.DataFrame
        Fully scored and ranked DataFrame.
    sheet_id : str or None
        Google Sheets document ID.  Pass None to auto-create.
    total_found : int
    total_skipped : int
    elapsed_seconds : float
    credentials_file : str
        Path to Google Service Account JSON credentials.
    run_type : str
        "morning" or "afternoon".
    is_partial : bool
        True if fewer than PARTIAL_RUN_THRESHOLD stocks were fetched.
    top25_changes : dict or None
        Output of delta_tracker.get_top25_changes() for the Summary tab.

    Returns
    -------
    str
        URL of the Google Spreadsheet.
    """
    client = _get_client(credentials_file)
    ss     = get_or_create_spreadsheet(client, sheet_id)

    # Daily tab: named "YYYY-MM-DD"
    sheet_title = datetime.now().strftime("%Y-%m-%d")

    write_main_sheet(ss, df, sheet_title, run_type=run_type)
    write_summary_sheet(
        ss,
        df,
        total_found=total_found,
        total_skipped=total_skipped,
        elapsed_seconds=elapsed_seconds,
        run_type=run_type,
        is_partial=is_partial,
        top25_changes=top25_changes,
        regime_label=regime_label,
        weighting_scheme=weighting_scheme,
        regime_filter_on=regime_filter_on,
    )
    write_analytics_tab(ss, df)
    _delete_old_tabs(ss)

    return ss.url
