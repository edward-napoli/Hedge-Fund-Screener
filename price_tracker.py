"""
price_tracker.py — Daily closing price logger for all screened stocks.

Fetches official closing prices after all markets close, converts non-USD
prices to USD using the day's FX closing rate, and persists data to
cache/price_history.json.

Called by scheduler.py after the latest market close + PRICE_FETCH_DELAY_MINUTES.
Also updates the Price History and Score History tabs in Google Sheets.
"""

import json
import logging
import logging.handlers
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from config import EXCHANGE_MAP

logger = logging.getLogger(__name__)

PRICE_HISTORY_FILE = "cache/price_history.json"
SCORE_HISTORY_FILE = "cache/score_history.json"
PRICE_TRACKER_LOG  = "logs/price_tracker.log"

# FX ticker mapping: exchange suffix → (Yahoo Finance FX pair, currency code)
# These FX pairs return units of local currency per 1 USD, so to get USD per
# local unit we use the pair directly (e.g. GBPUSD=X gives GBP→USD rate).
SUFFIX_TO_FX: dict[str, tuple[str, str]] = {
    ".L":  ("GBPUSD=X", "GBP"),
    ".DE": ("EURUSD=X", "EUR"),
    ".PA": ("EURUSD=X", "EUR"),
    ".T":  ("JPYUSD=X", "JPY"),
    ".TO": ("CADUSD=X", "CAD"),
    ".AX": ("AUDUSD=X", "AUD"),
    ".HK": ("HKDUSD=X", "HKD"),
    ".AS": ("EURUSD=X", "EUR"),
    ".BR": ("EURUSD=X", "EUR"),
    ".HE": ("EURUSD=X", "EUR"),
    ".MI": ("EURUSD=X", "EUR"),
    ".MC": ("EURUSD=X", "EUR"),
    ".SW": ("CHFUSD=X", "CHF"),
    ".ST": ("SEKUSD=X", "SEK"),
    ".OL": ("NOKUSD=X", "NOK"),
    ".CO": ("DKKUSD=X", "DKK"),
}

# Module-level FX rate cache (populated during a single run_price_fetch call)
_fx_rate_cache: dict[str, Optional[float]] = {}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_price_history() -> dict:
    """
    Load and return cache/price_history.json.

    Returns
    -------
    dict
        Parsed history dict, or empty dict if the file is missing or unreadable.
    """
    if not os.path.exists(PRICE_HISTORY_FILE):
        return {}
    try:
        with open(PRICE_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load price history: {exc}")
        return {}


def save_price_history(history: dict) -> None:
    """
    Write the full price history dict to cache/price_history.json.

    Parameters
    ----------
    history : dict
        Complete price history mapping date_str → {ticker → price_data}.
    """
    os.makedirs("cache", exist_ok=True)
    try:
        with open(PRICE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Price history saved to {PRICE_HISTORY_FILE}.")
    except Exception as exc:
        logger.error(f"Could not save price history: {exc}")


# ---------------------------------------------------------------------------
# FX / suffix helpers
# ---------------------------------------------------------------------------

def _get_suffix(ticker: str) -> Optional[str]:
    """
    Return the first matching suffix from SUFFIX_TO_FX for the given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AZN.L" or "AAPL".

    Returns
    -------
    str or None
        The matching suffix string (e.g. ".L"), or None for USD-denominated
        stocks (no recognised suffix).
    """
    for suffix in SUFFIX_TO_FX:
        if ticker.endswith(suffix):
            return suffix
    return None


def _fetch_fx_rate(fx_pair: str) -> Optional[float]:
    """
    Fetch the latest closing rate for a Yahoo Finance FX pair.

    Results are cached in the module-level _fx_rate_cache dict so the same
    pair is only fetched once per run_price_fetch() call.

    Parameters
    ----------
    fx_pair : str
        Yahoo Finance FX ticker, e.g. "GBPUSD=X".

    Returns
    -------
    float or None
        Closing rate (units of target currency per 1 unit of base currency),
        or None if the fetch fails.
    """
    if fx_pair in _fx_rate_cache:
        return _fx_rate_cache[fx_pair]

    try:
        hist = yf.Ticker(fx_pair).history(period="2d")
        if hist.empty:
            logger.warning(f"FX rate for {fx_pair}: empty history.")
            _fx_rate_cache[fx_pair] = None
            return None
        rate = float(hist["Close"].iloc[-1])
        _fx_rate_cache[fx_pair] = rate
        logger.debug(f"FX rate {fx_pair} = {rate:.6f}")
        return rate
    except Exception as exc:
        logger.warning(f"FX rate fetch failed for {fx_pair}: {exc}")
        _fx_rate_cache[fx_pair] = None
        return None


# ---------------------------------------------------------------------------
# Single-ticker price fetch
# ---------------------------------------------------------------------------

def _fetch_closing_price(ticker: str, date_str: str) -> dict:
    """
    Fetch the closing price for a single ticker on the given date.

    Conversion to USD is applied for non-US tickers using the day's FX rate.
    If the exact date is unavailable, the most recent available close is used
    and flagged with ``fallback=True``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    date_str : str
        Target date in "YYYY-MM-DD" format.

    Returns
    -------
    dict with keys:
        price_usd   : float or None
        price_local : float or None
        currency    : str
        fallback    : bool  (True if exact date was unavailable)
        error       : bool  (True only on complete failure)
    """
    suffix   = _get_suffix(ticker)
    currency = "USD"
    fx_rate  = 1.0

    if suffix is not None:
        fx_pair, currency = SUFFIX_TO_FX[suffix]
        fetched_rate = _fetch_fx_rate(fx_pair)
        if fetched_rate is not None:
            fx_rate = fetched_rate
        else:
            # Cannot convert — price will still be logged in local currency
            fx_rate = None

    try:
        # Primary: fetch the exact trading day
        next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=date_str, end=next_day)

        fallback = False
        if hist.empty or "Close" not in hist.columns or hist["Close"].empty:
            # Fallback: most recent available close in last 5 days
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty or "Close" not in hist.columns or hist["Close"].empty:
                raise ValueError("No price data available")
            fallback = True

        price_local = float(hist["Close"].iloc[-1])

        if fx_rate is not None:
            price_usd = round(price_local * fx_rate, 4)
        else:
            # Unable to convert; store local price as USD placeholder (will be flagged)
            price_usd = None
            fallback  = True

        return {
            "price_usd":   price_usd,
            "price_local": round(price_local, 4),
            "currency":    currency,
            "fallback":    fallback,
            "error":       False,
        }

    except Exception as exc:
        logger.debug(f"Price fetch failed for {ticker}: {exc}")
        return {
            "price_usd":   None,
            "price_local": None,
            "currency":    currency,
            "fallback":    True,
            "error":       True,
        }


# ---------------------------------------------------------------------------
# Batch price fetcher
# ---------------------------------------------------------------------------

def fetch_all_closing_prices(tickers: list, date_str: str) -> dict:
    """
    Fetch closing prices for all tickers on the given date using a single
    batch yf.download() call, then fall back to per-ticker fetch for any
    that are missing, and finally to last known price from history.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch.
    date_str : str
        Target date in "YYYY-MM-DD" format.

    Returns
    -------
    dict
        Mapping ticker → price data dict.
    """
    import yfinance as yf

    results: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Batch download all tickers at once (fast — single API call)
    # ------------------------------------------------------------------
    next_day = (
        datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=3)
    ).strftime("%Y-%m-%d")

    logger.info(f"Batch downloading closing prices for {len(tickers)} tickers...")
    try:
        batch = yf.download(
            tickers,
            start=date_str,
            end=next_day,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # batch["Close"] is a DataFrame: rows=dates, cols=tickers
        close_df = batch["Close"] if "Close" in batch.columns.get_level_values(0) else batch
        if hasattr(close_df, "columns") and hasattr(close_df.columns, "get_level_values"):
            # Multi-level columns from yf.download
            if "Close" in batch:
                close_df = batch["Close"]
        logger.info(f"Batch download complete: {close_df.shape}")
    except Exception as exc:
        logger.warning(f"Batch download failed ({exc}) — falling back to per-ticker fetch.")
        close_df = pd.DataFrame()

    # Build results from the batch data
    fetched_via_batch: set = set()
    if not close_df.empty:
        # Get the last available row (most recent close)
        last_row = close_df.iloc[-1]
        for ticker in tickers:
            price_local = None
            if ticker in last_row.index:
                val = last_row[ticker]
                if pd.notna(val):
                    price_local = float(val)

            if price_local is not None:
                suffix  = _get_suffix(ticker)
                currency = "USD"
                fx_rate  = 1.0
                if suffix:
                    fx_pair, currency = SUFFIX_TO_FX[suffix]
                    fetched_rate = _fetch_fx_rate(fx_pair)
                    fx_rate = fetched_rate if fetched_rate is not None else 1.0

                results[ticker] = {
                    "price_usd":   round(price_local * fx_rate, 4),
                    "price_local": round(price_local, 4),
                    "currency":    currency,
                    "fallback":    False,
                    "error":       False,
                }
                fetched_via_batch.add(ticker)

    # ------------------------------------------------------------------
    # Step 2: Per-ticker fallback for anything the batch missed
    # ------------------------------------------------------------------
    missing = [t for t in tickers if t not in fetched_via_batch]
    if missing:
        logger.info(f"Per-ticker fallback fetch for {len(missing)} stocks...")
        for ticker in missing:
            results[ticker] = _fetch_closing_price(ticker, date_str)

    # ------------------------------------------------------------------
    # Step 3: Last-resort — use most recent historical price from cache
    # ------------------------------------------------------------------
    existing_history = load_price_history()
    sorted_dates = sorted(existing_history.keys(), reverse=True)

    for ticker, data in results.items():
        if data.get("price_usd") is not None:
            continue
        for past_date in sorted_dates:
            if past_date >= date_str:
                continue
            past_data = existing_history[past_date].get(ticker)
            if past_data and past_data.get("price_usd") is not None:
                results[ticker] = {
                    "price_usd":   past_data["price_usd"],
                    "price_local": past_data.get("price_local"),
                    "currency":    past_data.get("currency", "USD"),
                    "fallback":    True,
                    "error":       False,
                }
                break

    successful = sum(1 for d in results.values() if d.get("price_usd") is not None)
    fallbacks  = sum(1 for d in results.values() if d.get("fallback"))
    logger.info(
        f"Price fetch complete: {successful}/{len(tickers)} prices obtained, "
        f"{fallbacks} used fallback."
    )
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_price_fetch() -> None:
    """
    Main entry point called by scheduler.py after the last market close.

    Steps
    -----
    1. Configure file logger to logs/price_tracker.log.
    2. Load screened universe from cache/last_run.json.
    3. Fetch closing prices for all stocks in the universe.
    4. Persist prices to cache/price_history.json.
    5. Update Price History and Score History tabs in Google Sheets.
    6. Run efficacy analysis.
    7. Log summary statistics.
    """
    # ------------------------------------------------------------------
    # 1. File logger setup
    # ------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(PRICE_TRACKER_LOG, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    # Attach to root logger so all sub-module logs also appear in this file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Price Tracker — starting run_price_fetch()")

    # Clear the per-run FX cache
    global _fx_rate_cache
    _fx_rate_cache = {}

    # ------------------------------------------------------------------
    # 2. Load screened universe — from cache if available, else fetch fresh
    # ------------------------------------------------------------------
    cache_file = "cache/last_run.json"
    tickers: list[str] = []

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                last_run = json.load(f)
            stocks_list = last_run.get("stocks", [])
            tickers = [s["Ticker"] for s in stocks_list if "Ticker" in s]
            logger.info(f"Loaded {len(tickers)} tickers from last_run.json.")
        except Exception as exc:
            logger.warning(f"Could not read {cache_file}: {exc} — will fetch universe fresh.")

    if not tickers:
        logger.info("No cached ticker list — fetching stock universe from source.")
        try:
            from data_fetcher import get_stock_universe
            tickers = get_stock_universe()
            logger.info(f"Fetched {len(tickers)} tickers from stock universe.")
        except Exception as exc:
            logger.error(f"Could not load stock universe: {exc}. Price fetch aborted.")
            return

    # ------------------------------------------------------------------
    # 3. Fetch closing prices
    # ------------------------------------------------------------------
    today_str = date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching closing prices for date: {today_str}")
    price_data = fetch_all_closing_prices(tickers, today_str)

    # ------------------------------------------------------------------
    # 4. Persist to price_history.json
    # ------------------------------------------------------------------
    history = load_price_history()
    history[today_str] = price_data
    save_price_history(history)

    # ------------------------------------------------------------------
    # 5. Update Google Sheets tabs
    # ------------------------------------------------------------------
    credentials_file = os.getenv("CREDENTIALS_FILE", "credentials.json")
    sheet_id         = os.getenv("GOOGLE_SHEET_ID", "").strip() or None

    if not os.path.exists(credentials_file):
        logger.warning(
            f"Credentials file not found at '{credentials_file}'. "
            "Skipping Sheets update."
        )
    else:
        ss = None
        try:
            from sheets_writer import (
                _get_client,
                get_or_create_spreadsheet,
                update_price_history_tab,
                update_score_history_tab,
            )
            client = _get_client(credentials_file)
            ss     = get_or_create_spreadsheet(client, sheet_id)

            score_history = _load_score_history()

            update_price_history_tab(ss, history, score_history)
            update_score_history_tab(ss, score_history, history)

        except Exception as exc:
            logger.error(f"Sheets update failed: {exc}", exc_info=True)

        # ------------------------------------------------------------------
        # 6. Run efficacy analysis
        # ------------------------------------------------------------------
        if ss is not None:
            try:
                from efficacy_analyzer import run_efficacy_analysis
                run_efficacy_analysis(ss, credentials_file)
            except Exception as exc:
                logger.error(f"Efficacy analysis failed: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # 7. Summary log
    # ------------------------------------------------------------------
    successful = sum(1 for d in price_data.values() if d.get("price_usd") is not None)
    fallbacks  = sum(1 for d in price_data.values() if d.get("fallback"))
    errors     = sum(1 for d in price_data.values() if d.get("error"))

    logger.info(
        f"Price fetch summary: {successful}/{len(tickers)} successful, "
        f"{fallbacks} fallback, {errors} errors."
    )
    logger.info("Price Tracker — run_price_fetch() complete.")
    logger.info("=" * 60)

    # Remove the file handler we added (avoid duplicate handlers on repeated calls)
    root_logger.removeHandler(file_handler)
    file_handler.close()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_score_history() -> dict:
    """
    Load cache/score_history.json, returning an empty dict on failure.

    Returns
    -------
    dict
    """
    if not os.path.exists(SCORE_HISTORY_FILE):
        return {}
    try:
        with open(SCORE_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load score history: {exc}")
        return {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_price_fetch()
