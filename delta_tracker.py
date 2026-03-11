"""
delta_tracker.py — Compares current run against previous run for delta tracking.

Stores the previous run's ticker scores and ranks in cache/last_run.json so that
the next run can compute Score Delta and Rank Delta for every stock.
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR  = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "last_run.json")


def load_previous_run() -> Optional[dict]:
    """
    Load cached data from the previous run.

    Returns
    -------
    dict or None
        Parsed JSON with keys 'timestamp', 'run_type', and 'stocks' (list of dicts),
        or None if no cache file exists or it cannot be parsed.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return None


def save_current_run(df: pd.DataFrame, run_type: str, timestamp: str) -> None:
    """
    Persist current run's ticker/score/rank data to cache for next-run delta computation.

    Parameters
    ----------
    df : pd.DataFrame
        The fully scored and ranked DataFrame.
    run_type : str
        Either "morning" or "afternoon".
    timestamp : str
        ISO-style timestamp string (UTC).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    records = df[["Ticker", "Composite Score", "Rank"]].copy()
    data = {
        "timestamp": timestamp,
        "run_type":  run_type,
        "stocks":    records.to_dict(orient="records"),
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cache saved: {len(records)} stocks → {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")


def compute_deltas(df: pd.DataFrame, previous: Optional[dict]) -> pd.DataFrame:
    """
    Add 'Score Delta' and 'Rank Delta' columns by comparing against the previous run.

    Rules
    -----
    - New entries (not in previous cache) → delta = 0.
    - Stocks present in cache → delta = current - previous.
    - Stocks in cache but absent from current run are ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Current run DataFrame (must already contain 'Composite Score' and 'Rank').
    previous : dict or None
        Output of load_previous_run().

    Returns
    -------
    pd.DataFrame
        Copy of df with 'Score Delta' and 'Rank Delta' columns added.
    """
    df = df.copy()
    df["Score Delta"] = 0.0
    df["Rank Delta"]  = 0

    if previous is None:
        logger.info("No previous run found — Score Delta and Rank Delta set to 0.")
        return df

    prev_stocks = {s["Ticker"]: s for s in previous.get("stocks", [])}

    updated = 0
    for idx, row in df.iterrows():
        ticker = row["Ticker"]
        if ticker in prev_stocks:
            prev = prev_stocks[ticker]
            try:
                score_delta = float(row["Composite Score"]) - float(prev["Composite Score"])
                rank_delta  = int(row["Rank"]) - int(prev["Rank"])
                df.at[idx, "Score Delta"] = round(score_delta, 2)
                df.at[idx, "Rank Delta"]  = rank_delta
                updated += 1
            except (TypeError, ValueError):
                pass  # leave as 0 if conversion fails

    logger.info(f"Deltas computed for {updated} stocks vs previous run ({previous.get('timestamp', 'unknown')}).")
    return df


def get_top25_changes(df: pd.DataFrame, previous: Optional[dict]) -> dict:
    """
    Analyse which tickers entered or exited the top 25 since the last run,
    and identify big movers (|rank_delta| > 10).

    Parameters
    ----------
    df : pd.DataFrame
        Current ranked DataFrame (must contain 'Ticker', 'Rank', 'Rank Delta').
    previous : dict or None
        Output of load_previous_run().

    Returns
    -------
    dict with keys:
        'entered'    — list of tickers newly in top 25
        'exited'     — list of tickers that dropped out of top 25
        'big_movers' — list of (ticker, rank_delta) for |rank_delta| > 10
    """
    current_top25 = set(df.head(25)["Ticker"].tolist())
    entered: list  = []
    exited: list   = []
    big_movers: list = []

    if previous:
        prev_top25 = {
            s["Ticker"]
            for s in previous.get("stocks", [])
            if int(s.get("Rank", 9999)) <= 25
        }

        entered = [t for t in current_top25 if t not in prev_top25]
        exited  = [t for t in prev_top25  if t not in current_top25]

        for _, row in df.iterrows():
            try:
                rd = int(row["Rank Delta"])
                if abs(rd) > 10:
                    big_movers.append((row["Ticker"], rd))
            except (TypeError, ValueError):
                pass

    return {"entered": entered, "exited": exited, "big_movers": big_movers}


def apply_fallback_from_cache(
    current_tickers_fetched: set,
    all_tickers: list,
    previous: Optional[dict],
) -> list:
    """
    For tickers that failed to fetch this run, attempt to use cached values.

    Fallback records are flagged with an asterisk appended to the ticker symbol
    (e.g., "AAPL*") to indicate stale/cached data.

    Parameters
    ----------
    current_tickers_fetched : set
        Set of tickers successfully fetched this run.
    all_tickers : list
        Full universe of tickers attempted.
    previous : dict or None
        Output of load_previous_run().

    Returns
    -------
    list[dict]
        List of minimal fallback records (Ticker, Composite Score, Rank only).
        Returns empty list if previous is None.
    """
    if previous is None:
        return []

    fallbacks = []
    prev_map = {s["Ticker"]: s for s in previous.get("stocks", [])}

    for ticker in all_tickers:
        if ticker not in current_tickers_fetched and ticker in prev_map:
            prev = prev_map[ticker]
            fallback: dict = {col: "N/A" for col in ["Ticker", "Composite Score", "Rank"]}
            fallback["Ticker"]          = ticker + "*"   # asterisk = cached/fallback
            fallback["Composite Score"] = prev["Composite Score"]
            fallback["Rank"]            = prev["Rank"]
            fallbacks.append(fallback)

    if fallbacks:
        logger.info(f"Applied {len(fallbacks)} fallback entries from cache.")

    return fallbacks
