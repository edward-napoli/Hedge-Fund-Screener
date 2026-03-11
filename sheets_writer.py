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
    )
    write_analytics_tab(ss, df)
    _delete_old_tabs(ss)

    return ss.url
