"""
efficacy_analyzer.py — Clayton Score predictive efficacy analysis.

Calculates:
- Pearson correlation between Clayton Score and forward price returns (1d, 5d, 21d)
- Quintile performance analysis (Q1–Q5 average returns)
- Rolling 30-day correlation
- Top-25 entry validation

Data sources:
- cache/price_history.json  — daily closing prices per stock
- cache/score_history.json  — daily Clayton Scores per stock

Minimum data gate: 21 trading days required before displaying results.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PRICE_HISTORY_FILE = "cache/price_history.json"
SCORE_HISTORY_FILE = "cache/score_history.json"
MIN_TRADING_DAYS   = 21   # minimum days before showing efficacy metrics


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """
    Load a JSON file and return its contents.

    Parameters
    ----------
    path : str
        File path to read.

    Returns
    -------
    dict
        Parsed contents, or empty dict on any error.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")
        return {}


def _get_sorted_dates(history: dict) -> list:
    """
    Return a sorted list of date keys from a history dict.

    Parameters
    ----------
    history : dict
        History dict with "YYYY-MM-DD" keys.

    Returns
    -------
    list[str]
        Ascending-sorted list of date strings.
    """
    return sorted(history.keys())


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _build_aligned_df(price_history: dict, score_history: dict) -> pd.DataFrame:
    """
    Build a merged DataFrame aligning daily scores with daily prices.

    Only rows where both a Clayton Score AND a USD price are available for
    the same (date, ticker) pair are included.

    Columns: date, ticker, score, rank, price_usd

    Parameters
    ----------
    price_history : dict
        Mapping date_str → {ticker: price_data_dict}.
    score_history : dict
        Mapping date_str → {run_type, timestamp, stocks: {ticker: {score, rank}}}.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame, or empty DataFrame if insufficient data.
    """
    rows = []
    score_dates = set(_get_sorted_dates(score_history))
    price_dates = set(_get_sorted_dates(price_history))
    common_dates = sorted(score_dates & price_dates)

    for date_str in common_dates:
        stocks_scores = score_history[date_str].get("stocks", {})
        day_prices    = price_history[date_str]

        for ticker, score_data in stocks_scores.items():
            price_data = day_prices.get(ticker)
            if price_data is None:
                continue
            price_usd = price_data.get("price_usd")
            if price_usd is None:
                continue

            try:
                score = float(score_data["score"])
                rank  = int(score_data["rank"])
            except (KeyError, TypeError, ValueError):
                continue

            rows.append({
                "date":      date_str,
                "ticker":    ticker,
                "score":     score,
                "rank":      rank,
                "price_usd": price_usd,
            })

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "score", "rank", "price_usd"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def _calc_forward_returns(df: pd.DataFrame, n_days: int) -> pd.Series:
    """
    Calculate n-day forward price returns for each (date, ticker) row.

    Matches to the nearest available date at or after date + n_days using
    the sorted available dates in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned DataFrame with columns [date, ticker, price_usd].
    n_days : int
        Number of calendar days forward.

    Returns
    -------
    pd.Series
        Forward return (%) aligned with df.index; NaN where unavailable.
    """
    from datetime import timedelta

    # Build a quick lookup: (date, ticker) → price_usd
    price_lookup: dict = {}
    for _, row in df.iterrows():
        price_lookup[(row["date"], row["ticker"])] = row["price_usd"]

    # Sorted unique dates per ticker for approximate matching
    dates_by_ticker: dict = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        if t not in dates_by_ticker:
            dates_by_ticker[t] = []
        dates_by_ticker[t].append(row["date"])
    # Sort in place
    for t in dates_by_ticker:
        dates_by_ticker[t] = sorted(set(dates_by_ticker[t]))

    returns = []
    for _, row in df.iterrows():
        current_date   = row["date"]
        ticker         = row["ticker"]
        current_price  = row["price_usd"]

        # Target date (n_days calendar days forward)
        try:
            target_dt = datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=n_days)
            target_str = target_dt.strftime("%Y-%m-%d")
        except Exception:
            returns.append(float("nan"))
            continue

        # Find the nearest date at or after target_str
        ticker_dates = dates_by_ticker.get(ticker, [])
        future_date = None
        for d in ticker_dates:
            if d >= target_str:
                future_date = d
                break

        if future_date is None:
            returns.append(float("nan"))
            continue

        future_price = price_lookup.get((future_date, ticker))
        if future_price is None or current_price == 0:
            returns.append(float("nan"))
            continue

        ret = (future_price - current_price) / current_price * 100.0
        returns.append(ret)

    return pd.Series(returns, index=df.index)


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlations between Clayton Score and forward returns.

    Both per-stock (score vs own forward return) and portfolio-level
    (cross-stock, all observations pooled) correlations are computed.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned DataFrame with columns [date, ticker, score, rank, price_usd].

    Returns
    -------
    dict with keys:
        per_stock_1d, per_stock_5d, per_stock_21d : dict {ticker: r}
        portfolio_1d, portfolio_5d, portfolio_21d  : float (Pearson r)
        p_value_1d, p_value_5d, p_value_21d        : float
    """
    result: dict = {
        "per_stock_1d":  {},
        "per_stock_5d":  {},
        "per_stock_21d": {},
        "portfolio_1d":  None,
        "portfolio_5d":  None,
        "portfolio_21d": None,
        "p_value_1d":    None,
        "p_value_5d":    None,
        "p_value_21d":   None,
    }

    if df.empty:
        return result

    for n_days, key_per, key_port, key_p in [
        (1,  "per_stock_1d",  "portfolio_1d",  "p_value_1d"),
        (5,  "per_stock_5d",  "portfolio_5d",  "p_value_5d"),
        (21, "per_stock_21d", "portfolio_21d", "p_value_21d"),
    ]:
        fwd = _calc_forward_returns(df, n_days)
        temp = df.copy()
        temp["fwd_return"] = fwd.values

        # Portfolio-level: pool all (score, fwd_return) observations
        valid = temp.dropna(subset=["fwd_return"])
        if len(valid) >= 5:
            try:
                r, p = stats.pearsonr(valid["score"], valid["fwd_return"])
                result[key_port] = round(float(r), 4)
                result[key_p]    = round(float(p), 6)
            except Exception:
                pass

        # Per-stock: each ticker independently
        per_stock: dict = {}
        for ticker, grp in temp.groupby("ticker"):
            grp_valid = grp.dropna(subset=["fwd_return"])
            if len(grp_valid) < 5:
                continue
            try:
                r, _ = stats.pearsonr(grp_valid["score"], grp_valid["fwd_return"])
                per_stock[str(ticker)] = round(float(r), 4)
            except Exception:
                pass
        result[key_per] = per_stock

    return result


# ---------------------------------------------------------------------------
# Quintile performance
# ---------------------------------------------------------------------------

def compute_quintile_performance(df: pd.DataFrame) -> dict:
    """
    Divide stocks into quintiles by score on each date and calculate average
    forward returns per quintile.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned DataFrame.

    Returns
    -------
    dict with keys:
        quintile_avg_1d, quintile_avg_5d, quintile_avg_21d : list[float] (Q1→Q5)
        q5_minus_q1_1d, q5_minus_q1_5d, q5_minus_q1_21d   : float
    """
    empty_result = {
        "quintile_avg_1d":  [None] * 5,
        "quintile_avg_5d":  [None] * 5,
        "quintile_avg_21d": [None] * 5,
        "q5_minus_q1_1d":   None,
        "q5_minus_q1_5d":   None,
        "q5_minus_q1_21d":  None,
    }

    if df.empty:
        return empty_result

    result = {}
    for n_days, suffix in [(1, "1d"), (5, "5d"), (21, "21d")]:
        fwd = _calc_forward_returns(df, n_days)
        temp = df.copy()
        temp["fwd_return"] = fwd.values

        quintile_avgs = []
        for date_str, day_grp in temp.groupby("date"):
            day_valid = day_grp.dropna(subset=["fwd_return"])
            if len(day_valid) < 5:
                continue
            # Label quintiles by score
            day_valid = day_valid.copy()
            try:
                day_valid["quintile"] = pd.qcut(
                    day_valid["score"], q=5, labels=False, duplicates="drop"
                )
            except Exception:
                continue
            for q in range(5):
                q_grp = day_valid[day_valid["quintile"] == q]
                if q_grp.empty:
                    continue
                quintile_avgs.append({"quintile": q, "return": q_grp["fwd_return"].mean()})

        if not quintile_avgs:
            result[f"quintile_avg_{suffix}"] = [None] * 5
            result[f"q5_minus_q1_{suffix}"]  = None
            continue

        q_df = pd.DataFrame(quintile_avgs)
        avgs = []
        for q in range(5):
            vals = q_df[q_df["quintile"] == q]["return"]
            avgs.append(round(float(vals.mean()), 4) if not vals.empty else None)

        result[f"quintile_avg_{suffix}"] = avgs
        q1 = avgs[0]
        q5 = avgs[4]
        if q1 is not None and q5 is not None:
            result[f"q5_minus_q1_{suffix}"] = round(q5 - q1, 4)
        else:
            result[f"q5_minus_q1_{suffix}"] = None

    return result


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def compute_rolling_correlation(df: pd.DataFrame, window_days: int = 30) -> list:
    """
    Calculate rolling 30-day Pearson correlation between score and 5d forward return.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned DataFrame.
    window_days : int
        Rolling window in calendar days (default 30).

    Returns
    -------
    list[dict]
        List of {"date": str, "correlation": float} dicts sorted ascending by date.
    """
    if df.empty:
        return []

    fwd = _calc_forward_returns(df, 5)
    temp = df.copy()
    temp["fwd_return"] = fwd.values
    temp = temp.dropna(subset=["fwd_return"])

    if temp.empty:
        return []

    sorted_dates = sorted(temp["date"].unique())
    results = []

    for i, end_date in enumerate(sorted_dates):
        # Window: calendar days in [end_date - window_days, end_date]
        try:
            end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=window_days)
            start_str = start_dt.strftime("%Y-%m-%d")
        except Exception:
            continue

        window_df = temp[(temp["date"] >= start_str) & (temp["date"] <= end_date)]
        if len(window_df) < 5:
            continue

        try:
            r, _ = stats.pearsonr(window_df["score"], window_df["fwd_return"])
            results.append({"date": end_date, "correlation": round(float(r), 4)})
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Top-25 validation
# ---------------------------------------------------------------------------

def compute_top25_validation(df: pd.DataFrame, score_history: dict) -> dict:
    """
    For every (date, ticker) where rank <= 25, compute actual forward returns.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned DataFrame with rank column.
    score_history : dict
        Full score history (not used directly — rank comes from df).

    Returns
    -------
    dict with keys:
        avg_top25_1d, avg_top25_5d, avg_top25_21d : float or None
        avg_all_1d,   avg_all_5d,   avg_all_21d   : float or None
        event_count : int
    """
    empty_result = {
        "avg_top25_1d":  None, "avg_top25_5d":  None, "avg_top25_21d":  None,
        "avg_all_1d":    None, "avg_all_5d":    None, "avg_all_21d":    None,
        "event_count":   0,
    }

    if df.empty:
        return empty_result

    result = dict(empty_result)
    result["event_count"] = int((df["rank"] <= 25).sum())

    for n_days, suffix in [(1, "1d"), (5, "5d"), (21, "21d")]:
        fwd = _calc_forward_returns(df, n_days)
        temp = df.copy()
        temp["fwd_return"] = fwd.values
        valid = temp.dropna(subset=["fwd_return"])

        all_mean = round(float(valid["fwd_return"].mean()), 4) if not valid.empty else None
        top25    = valid[valid["rank"] <= 25]
        top25_mean = round(float(top25["fwd_return"].mean()), 4) if not top25.empty else None

        result[f"avg_top25_{suffix}"] = top25_mean
        result[f"avg_all_{suffix}"]   = all_mean

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_efficacy_analysis(ss, credentials_file: str) -> None:
    """
    Orchestrate efficacy analysis and write results to Google Sheets.

    Opens its own Sheets connection if ss is None; otherwise uses the provided
    gspread.Spreadsheet object.

    Parameters
    ----------
    ss : gspread.Spreadsheet
        Open spreadsheet object (passed in from price_tracker).
    credentials_file : str
        Path to Google Service Account JSON credentials.
    """
    from sheets_writer import write_efficacy_tab, write_efficacy_tab_insufficient

    price_history = _load_json(PRICE_HISTORY_FILE)
    score_history = _load_json(SCORE_HISTORY_FILE)

    # Count number of trading days (dates present in score_history)
    trading_days = len(_get_sorted_dates(score_history))
    logger.info(f"Efficacy analysis: {trading_days} trading days in score history.")

    if trading_days < MIN_TRADING_DAYS:
        logger.info(
            f"Insufficient data for efficacy analysis "
            f"({trading_days} < {MIN_TRADING_DAYS} required). "
            "Writing placeholder tab."
        )
        try:
            write_efficacy_tab_insufficient(ss, trading_days)
        except Exception as exc:
            logger.error(f"write_efficacy_tab_insufficient failed: {exc}", exc_info=True)
        return

    # Build aligned DataFrame
    df = _build_aligned_df(price_history, score_history)
    if df.empty:
        logger.warning("Aligned DataFrame is empty — cannot compute efficacy metrics.")
        try:
            write_efficacy_tab_insufficient(ss, trading_days)
        except Exception as exc:
            logger.error(f"write_efficacy_tab_insufficient failed: {exc}", exc_info=True)
        return

    logger.info(f"Aligned DataFrame: {len(df)} (date, ticker) observations.")

    # Compute all metrics
    correlations = compute_correlations(df)
    quintiles    = compute_quintile_performance(df)
    rolling_corr = compute_rolling_correlation(df)
    top25_val    = compute_top25_validation(df, score_history)

    metrics = {
        **correlations,
        **quintiles,
        "rolling_correlation": rolling_corr,
        **{f"top25_{k}": v for k, v in top25_val.items()},
    }

    try:
        write_efficacy_tab(ss, metrics, trading_days)
        logger.info("Efficacy tab written to Google Sheets.")
    except Exception as exc:
        logger.error(f"write_efficacy_tab failed: {exc}", exc_info=True)
