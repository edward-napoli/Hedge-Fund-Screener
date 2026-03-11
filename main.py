"""
main.py — Entry point and run_screener() function for the Hedge Fund Stock Screener.

CLI usage:
    python main.py                         # morning run (default)
    python main.py --run-type afternoon    # afternoon run

Called by scheduler.py as:
    from main import run_screener
    run_screener(run_type="morning")   # or "afternoon"

Run pipeline
------------
1.  Load global stock universe (S&P 500 + international indices)
2.  Load previous-run cache for delta comparison
3.  Fetch financial data in parallel with retry + rate-limiting
4.  Build DataFrame, score & rank stocks
5.  Compute Score Delta / Rank Delta vs previous run
6.  Health-check: flag run as partial if < PARTIAL_RUN_THRESHOLD stocks
7.  Save dated CSV backup
8.  Write to Google Sheets (morning: fresh tab; afternoon: append section)
9.  Analyse top-25 changes vs previous run
10. Persist current run to cache for next-run deltas
11. Send Slack alert (if configured)
12. Append metadata entry to run_history.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory bootstrap — must happen BEFORE logging is configured so the log
# file's parent directory exists.
# ---------------------------------------------------------------------------
os.makedirs("logs",  exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ---------------------------------------------------------------------------
# Imports that may themselves import from config (which reads os.getenv)
# ---------------------------------------------------------------------------
from config import COLUMNS, PARTIAL_RUN_THRESHOLD
from data_fetcher import get_stock_universe, fetch_all_stocks
from scorer import score_and_rank
from sheets_writer import write_to_sheets
from delta_tracker import (
    load_previous_run,
    save_current_run,
    compute_deltas,
    get_top25_changes,
)
from alerts import send_slack_alert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "logs/errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUN_HISTORY_FILE  = "run_history.json"
_RETRY_WAIT_SECS  = 15 * 60   # 15 minutes between automatic retries


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, run_type: str) -> str:
    """
    Save the ranked DataFrame to a dated CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Fully scored and ranked DataFrame.
    run_type : str
        "morning" or "afternoon" — included in the filename.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    filename = f"stock_screener_{datetime.now().strftime('%Y-%m-%d')}_{run_type}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename


def _append_run_history(entry: dict) -> None:
    """
    Append a metadata dict to run_history.json without overwriting prior entries.

    Creates the file if it does not exist.  Silently continues if the file
    cannot be read (treats it as an empty history).

    Parameters
    ----------
    entry : dict
        Metadata for this run (timestamp, run_type, durations, top10, etc.).
    """
    history: list = []
    if os.path.exists(RUN_HISTORY_FILE):
        try:
            with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read {RUN_HISTORY_FILE}: {exc} — starting fresh.")
            history = []

    history.append(entry)

    try:
        with open(RUN_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        logger.error(f"Could not write {RUN_HISTORY_FILE}: {exc}")


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def _do_run(run_type: str) -> dict:
    """
    Execute one complete screener pipeline.

    Parameters
    ----------
    run_type : str
        "morning" or "afternoon".

    Returns
    -------
    dict
        Run metadata suitable for appending to run_history.json.

    Raises
    ------
    RuntimeError
        If no stock data could be retrieved at all (catastrophic failure).
    """
    start_time  = time.monotonic()
    timestamp   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    error_count = 0

    # ------------------------------------------------------------------
    # 1. Stock universe
    # ------------------------------------------------------------------
    tickers = get_stock_universe()
    print(f"\n[INFO] Loading stock universe... {len(tickers):,} stocks found")

    # ------------------------------------------------------------------
    # 2. Load previous run cache (for delta computation)
    # ------------------------------------------------------------------
    previous = load_previous_run()
    if previous:
        logger.info(
            f"Loaded previous run cache from {previous.get('timestamp', 'unknown')} "
            f"({len(previous.get('stocks', []))} stocks)."
        )
    else:
        logger.info("No previous run cache found — deltas will be 0.")

    # ------------------------------------------------------------------
    # 3. Fetch financial data
    # ------------------------------------------------------------------
    print("[INFO] Fetching financial data...")
    raw_data      = fetch_all_stocks(tickers)
    total_found   = len(raw_data)
    total_skipped = len(tickers) - total_found

    if not raw_data:
        raise RuntimeError(
            "No stock data retrieved — possible network outage or API failure."
        )

    # ------------------------------------------------------------------
    # 4. Build DataFrame
    # COLUMNS includes Score Delta + Rank Delta (added by compute_deltas),
    # so we exclude those plus Composite Score and Rank from the initial build.
    # ------------------------------------------------------------------
    base_cols = [
        c for c in COLUMNS
        if c not in ("Score Delta", "Rank Delta", "Composite Score", "Rank")
    ]
    df = pd.DataFrame(raw_data, columns=base_cols)

    # ------------------------------------------------------------------
    # 5. Score & Rank
    # ------------------------------------------------------------------
    print("[INFO] Calculating composite scores...")
    df = score_and_rank(df)

    # ------------------------------------------------------------------
    # 6. Compute deltas vs previous run
    # ------------------------------------------------------------------
    df = compute_deltas(df, previous)

    # ------------------------------------------------------------------
    # 7. Health check — flag partial runs
    # ------------------------------------------------------------------
    is_partial = total_found < PARTIAL_RUN_THRESHOLD
    if is_partial:
        logger.warning(
            f"PARTIAL RUN: only {total_found} stocks fetched "
            f"(threshold: {PARTIAL_RUN_THRESHOLD})."
        )

    # ------------------------------------------------------------------
    # 8. CSV backup
    # ------------------------------------------------------------------
    csv_file = _save_csv(df, run_type)
    print(f"[INFO] CSV backup saved: {csv_file}")

    # ------------------------------------------------------------------
    # 9. Top-25 change analysis (computed before Sheets write so it can
    #    be included in the Summary tab)
    # ------------------------------------------------------------------
    top25_changes = get_top25_changes(df, previous)
    if top25_changes["entered"]:
        logger.info(f"Entered top 25: {top25_changes['entered']}")
    if top25_changes["exited"]:
        logger.info(f"Exited top 25:  {top25_changes['exited']}")
    if top25_changes["big_movers"]:
        logger.info(f"Big movers:     {top25_changes['big_movers']}")

    # ------------------------------------------------------------------
    # 10. Google Sheets
    # ------------------------------------------------------------------
    sheet_id      = os.getenv("GOOGLE_SHEET_ID", "").strip() or None
    credentials_f = os.getenv("CREDENTIALS_FILE", "credentials.json")
    elapsed       = time.monotonic() - start_time

    sheet_url: str | None = None
    if not os.path.exists(credentials_f):
        logger.warning(
            f"Credentials file not found at '{credentials_f}'. "
            "Skipping Google Sheets upload — data saved to CSV only."
        )
    else:
        try:
            print("[INFO] Writing to Google Sheets...")
            sheet_url = write_to_sheets(
                df=df,
                sheet_id=sheet_id,
                total_found=total_found,
                total_skipped=total_skipped,
                elapsed_seconds=elapsed,
                credentials_file=credentials_f,
                run_type=run_type,
                is_partial=is_partial,
                top25_changes=top25_changes,
            )
        except Exception as exc:
            logger.error(f"Google Sheets write failed: {exc}", exc_info=True)
            error_count += 1

    # ------------------------------------------------------------------
    # 11. Save current run to cache (for next run's deltas)
    # ------------------------------------------------------------------
    save_current_run(df, run_type, timestamp)

    # ------------------------------------------------------------------
    # 12. Slack alert (top25_changes already computed in step 9)
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - start_time
    try:
        send_slack_alert(
            run_type=run_type,
            df=df,
            top25_changes=top25_changes,
            total_found=total_found,
            total_skipped=total_skipped,
            elapsed_seconds=elapsed,
            sheet_url=sheet_url,
            is_partial=is_partial,
        )
    except Exception as exc:
        logger.error(f"Slack alert error: {exc}")
        error_count += 1

    # ------------------------------------------------------------------
    # 13. Console summary
    # ------------------------------------------------------------------
    top_stock  = df.iloc[0] if not df.empty else None
    top_ticker = top_stock["Ticker"]          if top_stock is not None else "N/A"
    top_score  = top_stock["Composite Score"] if top_stock is not None else "N/A"
    partial_note = " [PARTIAL RUN]" if is_partial else ""

    print("\n" + "=" * 60)
    print(
        f"[INFO] Done{partial_note}! {total_found:,} stocks ranked. "
        f"Top: {top_ticker} ({top_score})"
    )
    if sheet_url:
        print(f"[INFO] Sheet URL: {sheet_url}")
    else:
        print(f"[INFO] Results saved locally: {csv_file}")
    print(
        f"[INFO] Total time: {elapsed:.1f}s  |  "
        f"Processed: {total_found:,}  |  Skipped: {total_skipped:,}"
    )
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 14. Build and return metadata dict
    # ------------------------------------------------------------------
    return {
        "timestamp":      timestamp,
        "run_type":       run_type,
        "duration_s":     round(elapsed, 1),
        "stocks_fetched": total_found,
        "stocks_skipped": total_skipped,
        "errors":         error_count,
        "is_partial":     is_partial,
        "top10": [
            {
                "ticker": str(row["Ticker"]),
                "score":  float(row["Composite Score"]),
            }
            for _, row in df.head(10).iterrows()
        ],
        "sheet_url": sheet_url,
    }


# ---------------------------------------------------------------------------
# Public API (called by scheduler.py)
# ---------------------------------------------------------------------------

def run_screener(run_type: str = "morning") -> None:
    """
    Execute the screener pipeline with a single automatic retry on total failure.

    On success, metadata is appended to run_history.json.
    On total failure (both attempt and retry), a failure record is written.

    Parameters
    ----------
    run_type : str
        "morning" or "afternoon" (default: "morning").
    """
    try:
        meta = _do_run(run_type)
        _append_run_history(meta)
    except Exception as exc:
        logger.error(
            f"Screener run ({run_type}) failed: {exc}. "
            f"Retrying in {_RETRY_WAIT_SECS // 60} minutes...",
            exc_info=True,
        )
        time.sleep(_RETRY_WAIT_SECS)
        try:
            meta = _do_run(run_type)
            meta["retried"] = True
            _append_run_history(meta)
        except Exception as exc2:
            logger.error(
                f"Retry also failed: {exc2}. Run aborted.",
                exc_info=True,
            )
            _append_run_history({
                "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "run_type":   run_type,
                "status":     "FAILED",
                "error":      str(exc2),
            })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hedge Fund Global Stock Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --run-type afternoon\n"
        ),
    )
    parser.add_argument(
        "--run-type",
        choices=["morning", "afternoon"],
        default="morning",
        help="Which run to execute (default: morning)",
    )
    args = parser.parse_args()
    run_screener(run_type=args.run_type)
