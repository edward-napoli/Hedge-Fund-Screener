"""
backtest.py — Clayton Score backtesting engine.

Simulates a long-only portfolio that rebalances on a chosen frequency,
holding the top-N stocks by Clayton Score computed from point-in-time
historical fundamentals.

Features:
  - Monthly / weekly / daily rebalancing
  - Transaction costs (configurable bps)
  - Liquidity filter (max % of average daily volume)
  - Market regime segmentation (bull / bear / sideways vs SPY)
  - Long-short variant (long top-N, short bottom-N)
  - Performance metrics: CAGR, Sharpe, Sortino, Calmar, max drawdown
  - Benchmark comparison (SPY buy-and-hold)
  - Risk parity weighting (inverse-volatility)
  - Regime filter (go partial-cash when SPY < 200-day MA)

CLI usage:
    python backtest.py --run
    python backtest.py --run --top-n 25 --rebalance monthly --long-short
    python backtest.py --run --weighting risk_parity --regime-filter
    python backtest.py --run --comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scorer import normalize_pe_pb_factors

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters (overridden by CLI / .env)
# ---------------------------------------------------------------------------
DEFAULT_TOP_N             = int(os.getenv("BACKTEST_TOP_N", "25"))
DEFAULT_RISK_FREE         = float(os.getenv("RISK_FREE_RATE", "0.04"))   # annual, e.g. 0.04 = 4%
DEFAULT_TC_BPS            = float(os.getenv("TRANSACTION_COST_BPS", "10"))  # one-way bps
DEFAULT_MAX_ADV           = float(os.getenv("MAX_ADV_PARTICIPATION", "0.05"))  # 5% of ADV
BACKTEST_START            = date(date.today().year - 10, 1, 1)
BACKTEST_END              = date.today()
RESULTS_DIR               = "cache/backtest"
os.makedirs(RESULTS_DIR, exist_ok=True)

# New env-driven constants for weighting / regime
DEFAULT_WEIGHTING         = os.getenv("WEIGHTING_SCHEME", "risk_parity")      # "equal" or "risk_parity"
DEFAULT_REGIME_FILTER     = os.getenv("REGIME_FILTER", "true").lower() == "true"
DEFAULT_CASH_FRACTION     = float(os.getenv("REGIME_CASH_FRACTION", "0.50"))
DEFAULT_VOL_LOOKBACK      = int(os.getenv("VOLATILITY_LOOKBACK_DAYS", "60"))
DEFAULT_MAX_SINGLE_POS    = float(os.getenv("MAX_SINGLE_POSITION", "0.15"))

SPY_CACHE_FILE            = "cache/spy_prices.json"

RebalanceFreq = Literal["monthly", "weekly", "daily"]


# ---------------------------------------------------------------------------
# SPY price fetching + caching
# ---------------------------------------------------------------------------

def fetch_spy_prices(
    start: date | None = None,
    end:   date | None = None,
) -> pd.Series | None:
    """
    Return SPY daily closing prices as pd.Series(date -> close).

    Loads from cache/spy_prices.json if the file is < 24 hours old and covers
    the requested period; otherwise re-fetches from yfinance and updates the cache.
    Fetches one extra year before `start` so the 200-day MA is available from day 1.
    """
    import time as _time

    if start is None:
        start = BACKTEST_START
    if end is None:
        end = BACKTEST_END

    os.makedirs("cache", exist_ok=True)

    # ── Try cache first ──────────────────────────────────────────────────────
    if os.path.exists(SPY_CACHE_FILE):
        age_hours = (_time.time() - os.path.getmtime(SPY_CACHE_FILE)) / 3600
        try:
            with open(SPY_CACHE_FILE, "r") as _f:
                cached = json.load(_f)
            prices_dict = cached.get("prices", {})
            if prices_dict:
                cached_start = min(prices_dict.keys())
                cached_end   = max(prices_dict.keys())
                # Accept cache if: < 24 h old AND covers full requested window
                covers_start = cached_start <= date(start.year - 1, 1, 1).isoformat()
                covers_end   = cached_end   >= (end - timedelta(days=7)).isoformat()
                if age_hours < 24 and covers_start and covers_end:
                    s = pd.Series(
                        {date.fromisoformat(k): float(v) for k, v in prices_dict.items()}
                    ).sort_index()
                    logger.info(
                        "SPY prices loaded from cache (%d days, %s to %s)",
                        len(s), cached_start, cached_end,
                    )
                    return s
        except Exception as exc:
            logger.warning("SPY cache read failed: %s", exc)

    # ── Fetch from yfinance ──────────────────────────────────────────────────
    try:
        import yfinance as yf
        fetch_start = date(start.year - 1, 1, 1)   # extra year for 200-day MA
        fetch_end   = end + timedelta(days=1)
        logger.info("Fetching SPY prices from yfinance (%s to %s)...", fetch_start, end)
        hist = yf.Ticker("SPY").history(
            start=fetch_start.isoformat(),
            end=fetch_end.isoformat(),
            auto_adjust=True,
        )
        if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            logger.error("SPY yfinance fetch returned empty or None result")
            return None

        # yfinance >= 0.2 sometimes wraps columns in a MultiIndex (field, ticker).
        # Flatten to single-level so ["Close"] always returns a Series.
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if "Close" not in hist.columns:
            logger.error("SPY fetch has no 'Close' column — found: %s", list(hist.columns))
            return None

        s = hist["Close"].copy()
        # Guard: if still a DataFrame (duplicate column names), take the first column.
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).date
        s = s.sort_index()

        prices_dict = {k.isoformat(): float(v) for k, v in s.items()}
        with open(SPY_CACHE_FILE, "w") as _f:
            json.dump({"prices": prices_dict, "fetched": date.today().isoformat()}, _f)
        logger.info("SPY prices cached to %s (%d trading days)", SPY_CACHE_FILE, len(s))
        return s

    except Exception as exc:
        logger.error("Failed to fetch SPY prices: %s", exc)
        return None


def _spy_price_on(spy_prices: pd.Series, as_of: date) -> float | None:
    """Return SPY closing price on or before `as_of`, or None if unavailable."""
    avail = [d for d in spy_prices.index if d <= as_of]
    if not avail:
        return None
    return float(spy_prices[max(avail)])


# ---------------------------------------------------------------------------
# Relative performance metrics (vs benchmark)
# ---------------------------------------------------------------------------

def compute_relative_metrics(
    port_rets:        pd.Series,
    bench_rets:       pd.Series,
    port_cagr:        float,
    bench_cagr:       float,
    risk_free:        float,
    periods_per_year: int,
) -> dict:
    """
    Compute portfolio-vs-benchmark relative performance metrics.

    All CAGR values must be in percentage points (e.g. 18.55, not 0.1855).

    Returns
    -------
    dict with keys:
        beta, alpha, correlation, excess_cagr, information_ratio
    """
    common = port_rets.index.intersection(bench_rets.index)
    if len(common) < 4:
        return {}

    p = port_rets.loc[common].values.astype(float)
    b = bench_rets.loc[common].values.astype(float)

    # Beta = Cov(port, bench) / Var(bench)
    cov_mat  = np.cov(p, b)
    var_b    = float(cov_mat[1, 1])
    beta     = float(cov_mat[0, 1] / var_b) if var_b > 0 else 0.0

    # Jensen's Alpha (annualised, in percentage points)
    rf_pct = risk_free * 100
    alpha  = port_cagr - rf_pct - beta * (bench_cagr - rf_pct)

    # Pearson correlation
    corr = float(np.corrcoef(p, b)[0, 1])

    # Excess CAGR (simple difference)
    excess_cagr = port_cagr - bench_cagr

    # Information Ratio = annualised mean excess return / tracking error
    excess_rets    = p - b
    tracking_error = float(np.std(excess_rets, ddof=1)) * np.sqrt(periods_per_year)
    if tracking_error > 0:
        ann_excess = float(np.mean(excess_rets)) * periods_per_year
        ir = ann_excess / tracking_error
    else:
        ir = 0.0

    return {
        "beta":              round(beta, 4),
        "alpha":             round(alpha, 2),
        "correlation":       round(corr, 4),
        "excess_cagr":       round(excess_cagr, 2),
        "information_ratio": round(ir, 4),
    }

# ---------------------------------------------------------------------------
# Scoring helper (inline, uses configurable weights)
# ---------------------------------------------------------------------------

def _safe(v) -> float:
    """Return float or 0.0."""
    import math
    if v is None:
        return 0.0
    try:
        f = float(v)
        return f if not (math.isnan(f) or math.isinf(f)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_row(row: dict, weights: dict) -> float:
    """
    Compute Clayton Score for a fundamental row using the given weights dict.

    weights keys:  E1, E3, E5, Ef, Ra, Re, Rc, C, Z, F, A, Y_outer, Pe_coef, Pb_coef
    Row keys match the historical snapshot + extra fields for valuation.
    """
    w = weights
    E1  = _safe(row.get("eps_1y_growth"))
    E3  = _safe(row.get("eps_3y_growth"))
    E5  = _safe(row.get("eps_5y_growth"))
    Ef  = _safe(row.get("eps_fwd_growth"))
    Ra  = _safe(row.get("roa"))
    Re  = _safe(row.get("roe"))
    Rc  = _safe(row.get("roic"))
    C   = _safe(row.get("current_ratio"))
    Z   = _safe(row.get("altman_z"))
    F   = _safe(row.get("piotroski_f"))
    A   = _safe(row.get("net_income_usd_m"))
    Y   = _safe(row.get("div_yield_pct"))
    Pr  = _safe(row.get("payout_ratio_pct"))
    Pe  = _safe(row.get("pe_ratio"))
    Pb  = _safe(row.get("pb_ratio"))

    growth   = w["E1"]*E1 + w["E3"]*E3 + w["E5"]*E5 + w["Ef"]*Ef
    quality  = Ra + Re + w["Rc"]*Rc
    liquid   = w["C"]*C + w["Z"]*Z + w["F"]*F
    income   = w["A"] * A
    div_val  = w["Y_outer"] * (Y * (2 - Pr/100) - (w["Pe_coef"]*Pe + w["Pb_coef"]*Pb))

    return round(growth + quality + liquid + income + div_val, 4)


# ---------------------------------------------------------------------------
# Rebalancing date generator
# ---------------------------------------------------------------------------

def get_rebalance_dates(
    start: date,
    end:   date,
    freq:  RebalanceFreq,
) -> list[date]:
    """Return a list of rebalancing dates between start and end."""
    dates = []
    cur   = start
    if freq == "monthly":
        while cur <= end:
            dates.append(cur)
            # advance to first day of next month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                cur = cur.replace(month=cur.month + 1, day=1)
    elif freq == "weekly":
        # every Monday
        # wind to nearest Monday
        offset = (0 - cur.weekday()) % 7
        cur = cur + timedelta(days=offset)
        while cur <= end:
            dates.append(cur)
            cur += timedelta(weeks=1)
    else:  # daily
        while cur <= end:
            dates.append(cur)
            cur += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------

def compute_market_regimes(spy_prices: pd.Series) -> pd.Series:
    """
    Label each trading day as 'bull', 'bear', or 'sideways' based on
    SPY's 200-day moving average and recent momentum.

    Returns a pd.Series indexed by date with string labels.
    """
    if spy_prices is None or spy_prices.empty:
        return pd.Series(dtype=str)

    df = spy_prices.to_frame("close").copy()
    df.index = pd.to_datetime(df.index)
    df["ma200"] = df["close"].rolling(200, min_periods=50).mean()
    df["ret_3m"] = df["close"].pct_change(63)  # ~3 months

    def _regime(row):
        if pd.isna(row["ma200"]) or pd.isna(row["ret_3m"]):
            return "unknown"
        if row["close"] > row["ma200"] and row["ret_3m"] > 0.05:
            return "bull"
        elif row["close"] < row["ma200"] and row["ret_3m"] < -0.05:
            return "bear"
        else:
            return "sideways"

    regimes = df.apply(_regime, axis=1)
    regimes.index = regimes.index.date
    return regimes


# ---------------------------------------------------------------------------
# ADV liquidity filter
# ---------------------------------------------------------------------------

def compute_adv(prices: dict[str, pd.Series], volumes: dict[str, pd.Series]) -> dict[str, float]:
    """
    Compute 20-day average daily volume (in dollar terms) for each ticker.
    Returns {ticker: avg_dollar_volume}.
    """
    adv: dict[str, float] = {}
    for ticker, vol_series in volumes.items():
        price_series = prices.get(ticker)
        if price_series is None or vol_series is None:
            continue
        # align
        common = vol_series.index.intersection(price_series.index)
        if len(common) < 5:
            continue
        recent = sorted(common)[-20:]
        dollar_vol = float(
            np.mean([
                float(vol_series.get(d, 0)) * float(price_series.get(d, 0))
                for d in recent
            ])
        )
        adv[ticker] = dollar_vol
    return adv


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

def apply_transaction_cost(portfolio_value: float, turnover: float, tc_bps: float) -> float:
    """
    Subtract one-way transaction costs.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio value before rebalancing.
    turnover : float
        Fraction of portfolio that changed (0–1).
    tc_bps : float
        One-way cost in basis points (e.g. 10 = 0.10%).

    Returns
    -------
    float
        Portfolio value after costs.
    """
    cost = portfolio_value * turnover * (tc_bps / 10_000)
    return portfolio_value - cost


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute standard performance metrics from a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily arithmetic returns (or whatever frequency).
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.04 = 4%).
    periods_per_year : int
        Trading periods in a year (252 for daily, 52 weekly, 12 monthly).

    Returns
    -------
    dict
        CAGR, Sharpe, Sortino, Calmar, max_drawdown, volatility, win_rate, etc.
    """
    if returns.empty or len(returns) < 5:
        return {}

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Cumulative
    cum    = (1 + returns).cumprod()
    total_return = float(cum.iloc[-1]) - 1
    n_years = len(returns) / periods_per_year
    cagr    = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Volatility
    vol = float(returns.std()) * np.sqrt(periods_per_year)

    # Sharpe
    excess = returns - rf_per_period
    sharpe = float(excess.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < rf_per_period] - rf_per_period
    down_dev = float(np.sqrt((downside ** 2).mean())) * np.sqrt(periods_per_year) if len(downside) > 0 else 0.001
    sortino  = float((returns.mean() - rf_per_period) * np.sqrt(periods_per_year) / down_dev) if down_dev > 0 else 0.0

    # Max drawdown
    roll_max    = cum.cummax()
    drawdowns   = (cum - roll_max) / roll_max
    max_dd      = float(drawdowns.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    win_rate = float((returns > 0).mean())

    # Best / worst period
    best   = float(returns.max())
    worst  = float(returns.min())

    return {
        "cagr":           round(cagr * 100, 2),
        "total_return":   round(total_return * 100, 2),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "calmar":         round(calmar, 3),
        "max_drawdown":   round(max_dd * 100, 2),
        "volatility":     round(vol * 100, 2),
        "win_rate":       round(win_rate * 100, 1),
        "best_period":    round(best * 100, 2),
        "worst_period":   round(worst * 100, 2),
        "n_periods":      len(returns),
    }


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def compute_realized_vol(
    prices: dict[str, pd.Series],
    ticker: str,
    as_of: date,
    lookback_days: int = 60,
    min_days: int = 30,
) -> float | None:
    """
    Compute annualized realized volatility for a ticker as of a given date.

    Parameters
    ----------
    prices : dict
        {ticker: pd.Series(date -> price)}
    ticker : str
        The ticker to compute vol for.
    as_of : date
        Use only prices on or before this date.
    lookback_days : int
        Number of daily returns to use (requires lookback_days+1 prices).
    min_days : int
        Minimum number of return observations required; returns None if fewer.

    Returns
    -------
    float or None
        Annualized volatility, or None if insufficient history.
    """
    series = prices.get(ticker)
    if series is None or series.empty:
        return None

    avail = sorted(d for d in series.index if d <= as_of)
    # Need lookback_days+1 prices to get lookback_days returns
    window = avail[-(lookback_days + 1):]
    if len(window) < min_days + 1:
        return None

    price_vals = [float(series[d]) for d in window]
    daily_rets = [
        (price_vals[i] - price_vals[i - 1]) / price_vals[i - 1]
        for i in range(1, len(price_vals))
        if price_vals[i - 1] > 0
    ]

    if len(daily_rets) < min_days:
        return None

    daily_std = float(np.std(daily_rets, ddof=1))
    return daily_std * np.sqrt(252)


# ---------------------------------------------------------------------------
# Risk parity weighting
# ---------------------------------------------------------------------------

def compute_risk_parity_weights(
    tickers: list[str],
    prices: dict[str, pd.Series],
    as_of: date,
    lookback_days: int = 60,
    min_days: int = 30,
    max_position: float = 0.15,
) -> dict[str, float]:
    """
    Compute inverse-volatility (risk parity) weights for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Tickers to weight.
    prices : dict
        {ticker: pd.Series(date -> price)}
    as_of : date
        Compute vol using only prices on or before this date.
    lookback_days : int
        Volatility lookback window in trading days.
    min_days : int
        Minimum observations to compute vol; missing tickers get average vol.
    max_position : float
        Maximum weight for any single position (e.g. 0.15 = 15%).

    Returns
    -------
    dict {ticker: weight}
        Weights summing to 1.0.
    """
    if not tickers:
        return {}

    # Compute raw vols
    vols: dict[str, float | None] = {}
    for t in tickers:
        vols[t] = compute_realized_vol(prices, t, as_of, lookback_days, min_days)

    # Fallback for tickers with insufficient history
    valid_vols = [v for v in vols.values() if v is not None and v > 0]
    fallback_vol = float(np.mean(valid_vols)) if valid_vols else 0.02 * np.sqrt(252)

    # Inverse-vol raw weights
    raw: dict[str, float] = {}
    for t in tickers:
        v = vols[t]
        if v is None or v <= 0:
            v = fallback_vol
        raw[t] = 1.0 / v

    # Normalize to sum to 1
    total = sum(raw.values())
    weights: dict[str, float] = {t: raw[t] / total for t in tickers}

    # Iterative position cap
    for _ in range(20):
        capped = {t: w for t, w in weights.items() if w >= max_position}
        uncapped = {t: w for t, w in weights.items() if w < max_position}

        if not capped:
            break  # no caps needed; stable

        # Fix capped weights at max_position; redistribute excess to uncapped
        excess = sum(w - max_position for w in capped.values())
        for t in capped:
            weights[t] = max_position

        if not uncapped or excess <= 0:
            break

        uncapped_total = sum(uncapped.values())
        if uncapped_total <= 0:
            break

        for t in uncapped:
            weights[t] += excess * (uncapped[t] / uncapped_total)

    # Final normalization (guard against floating-point drift)
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Regime filter (risk-on / risk-off based on SPY 200-day MA)
# ---------------------------------------------------------------------------

def is_risk_on(spy_prices: pd.Series, as_of: date, ma_window: int = 200) -> bool:
    """
    Determine whether the market is in a risk-on state as of the given date.

    Risk-on  = SPY closing price >= 200-day simple moving average.
    Risk-off = SPY closing price <  200-day simple moving average.

    Parameters
    ----------
    spy_prices : pd.Series
        SPY daily closing prices indexed by date.
    as_of : date
        Reference date.
    ma_window : int
        Number of trading days for the moving average (default 200).

    Returns
    -------
    bool
        True if risk-on (or insufficient data to determine regime).
    """
    if spy_prices is None or spy_prices.empty:
        return True

    avail = sorted(d for d in spy_prices.index if d <= as_of)
    if len(avail) < 50:
        return True  # default risk-on when insufficient history

    recent_price = float(spy_prices[avail[-1]])
    ma_window_prices = avail[-ma_window:]
    ma200 = float(np.mean([float(spy_prices[d]) for d in ma_window_prices]))

    return recent_price >= ma200


# ---------------------------------------------------------------------------
# Weighted period return (supports partial cash allocation)
# ---------------------------------------------------------------------------

def _compute_weighted_period_return(
    holdings_weights: dict[str, float],
    prices: dict[str, pd.Series],
    from_date: date,
    to_date: date,
    cash_return_per_period: float = 0.0,
) -> float:
    """
    Compute weighted portfolio return from from_date to to_date.

    Parameters
    ----------
    holdings_weights : dict {ticker: weight_fraction}
        Weights as fractions of total portfolio; need not sum to 1.
        The remainder (1 - sum(weights)) is treated as cash.
    prices : dict
        {ticker: pd.Series(date -> price)}
    from_date, to_date : date
        Period start and end.
    cash_return_per_period : float
        Return earned on the cash portion for this period (e.g. rf/periods_per_year).

    Returns
    -------
    float
        Total portfolio return for the period.
    """
    if not holdings_weights:
        return cash_return_per_period

    total_equity_weight = 0.0
    weighted_stock_return = 0.0

    for ticker, weight in holdings_weights.items():
        series = prices.get(ticker)
        if series is None:
            continue
        avail_to   = [d for d in series.index if d <= to_date]
        avail_from = [d for d in series.index if d <= from_date]
        if not avail_to or not avail_from:
            continue
        p_to   = float(series[max(avail_to)])
        p_from = float(series[max(avail_from)])
        if p_from > 0:
            stock_ret = (p_to - p_from) / p_from
            weighted_stock_return += weight * stock_ret
            total_equity_weight   += weight

    cash_weight = max(0.0, 1.0 - total_equity_weight)
    total_return = weighted_stock_return + cash_weight * cash_return_per_period
    return total_return


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    fundamentals_panel: dict[str, list[dict]],  # {ticker: [records]}
    prices:             dict[str, pd.Series],
    weights:            dict,
    top_n:              int  = DEFAULT_TOP_N,
    freq:               RebalanceFreq = "monthly",
    tc_bps:             float = DEFAULT_TC_BPS,
    risk_free:          float = DEFAULT_RISK_FREE,
    start:              date  = BACKTEST_START,
    end:                date  = BACKTEST_END,
    long_short:         bool  = False,
    spy_prices:         pd.Series | None = None,
    # New parameters (backward-compatible defaults)
    weighting_scheme:   str   = "equal",          # set to DEFAULT_WEIGHTING by CLI
    regime_filter:      bool  = False,             # set to DEFAULT_REGIME_FILTER by CLI
    cash_fraction:      float = DEFAULT_CASH_FRACTION,
    vol_lookback:       int   = DEFAULT_VOL_LOOKBACK,
    max_single_pos:     float = DEFAULT_MAX_SINGLE_POS,
) -> dict:
    """
    Run a full backtest simulation.

    Parameters
    ----------
    fundamentals_panel : dict
        {ticker: list of fundamental records (each a dict with 'period_end', 'filed_date', ...)}
    prices : dict
        {ticker: pd.Series(date -> closing price)}
    weights : dict
        Clayton Score weight dict (keys: E1, E3, E5, Ef, Ra, Re, Rc, C, Z, F, A, Y_outer, Pe_coef, Pb_coef)
    top_n : int
        Number of stocks to hold long (and short in long-short mode).
    freq : str
        Rebalancing frequency: 'monthly', 'weekly', 'daily'.
    tc_bps : float
        One-way transaction cost in basis points.
    risk_free : float
        Annual risk-free rate.
    start, end : date
        Backtest window.
    long_short : bool
        If True, also short the bottom-N stocks.
    spy_prices : pd.Series or None
        SPY closing prices for benchmark comparison and regime detection.
    weighting_scheme : str
        'equal' for equal-weight, 'risk_parity' for inverse-volatility weighting.
    regime_filter : bool
        If True, reduce equity exposure when SPY is below its 200-day MA.
    cash_fraction : float
        Fraction moved to cash during risk-off periods (e.g. 0.50 = hold 50% cash).
    vol_lookback : int
        Trading days of history used to compute realized volatility.
    max_single_pos : float
        Maximum weight for any single position in risk-parity mode.

    Returns
    -------
    dict
        {
          "portfolio_returns":  pd.Series,
          "benchmark_returns":  pd.Series,
          "metrics":            dict,
          "benchmark_metrics":  dict,
          "regime_metrics":     dict,
          "holdings_history":   list of {date, tickers, scores, weighting_scheme,
                                         equity_fraction, weights},
          "turnover_history":   list of floats,
          "regime_counts":      {"risk_on": int, "risk_off": int},
          "params":             dict,
        }
    """
    from historical_data import get_fundamental_snapshot, compute_eps_growth

    rebal_dates   = get_rebalance_dates(start, end, freq)
    spy_regimes   = compute_market_regimes(spy_prices) if spy_prices is not None else pd.Series(dtype=str)

    portfolio_val  = 1_000_000.0   # starting NAV
    # Holdings now stored as {ticker: weight_fraction} instead of position_value
    holdings: dict[str, float] = {}
    short_holdings: dict[str, float] = {}

    portfolio_values: list[tuple[date, float]] = []
    turnover_history: list[float] = []
    holdings_history: list[dict]  = []
    regime_counts: dict[str, int] = {"risk_on": 0, "risk_off": 0}

    periods_per_year = {"monthly": 12, "weekly": 52, "daily": 252}[freq]
    cash_return_per_period = (1 + risk_free) ** (1 / periods_per_year) - 1

    tickers_all = list(fundamentals_panel.keys())

    for i, rebal_date in enumerate(rebal_dates):
        # --- Step 1: Build fundamental snapshots for all tickers ---
        snaps: list[tuple[str, dict]] = []
        for ticker in tickers_all:
            snap = get_fundamental_snapshot(ticker, rebal_date, use_quarterly=True)
            if not snap:
                continue
            # Add EPS growth fields (computed from history)
            snap["eps_1y_growth"] = compute_eps_growth(ticker, rebal_date, 1)
            snap["eps_3y_growth"] = compute_eps_growth(ticker, rebal_date, 3)
            snap["eps_5y_growth"] = compute_eps_growth(ticker, rebal_date, 5)
            snap["eps_fwd_growth"] = None   # no forward estimates in history
            # Net income in USD M
            ni  = snap.get("net_income")
            snap["net_income_usd_m"] = ni / 1_000_000 if ni else None

            # ROIC proxy: net_income / (total_equity + total_debt)
            te  = snap.get("total_equity") or 0.0
            td  = snap.get("total_debt")   or 0.0
            ic  = te + td
            snap["roic"] = round(ni / ic * 100, 2) if (ni and ic > 0) else None

            # Valuation fields computed from price + cached per-share / aggregate data
            price_series  = prices.get(ticker)
            price_on_date: float | None = None
            if price_series is not None:
                avail = [d for d in price_series.index if d <= rebal_date]
                if avail:
                    price_on_date = float(price_series[max(avail)])

            eps = snap.get("eps_basic")

            # P/E = price / EPS (both in same currency per share)
            if price_on_date and eps and eps > 0:
                snap["pe_ratio"] = round(price_on_date / eps, 2)
            else:
                snap["pe_ratio"] = None

            # P/B: requires shares outstanding — not in cache schema, leave None
            snap["pb_ratio"] = None

            # Payout ratio = |dividends_paid| / net_income * 100
            div_paid = snap.get("dividends_paid")
            if div_paid is not None and ni and ni > 0:
                snap["payout_ratio_pct"] = round(abs(div_paid) / ni * 100, 2)
            else:
                snap["payout_ratio_pct"] = None

            # Div yield: estimate via shares proxy = net_income / eps_basic
            # dps = |dividends_paid| / shares; yield = dps / price * 100
            if div_paid is not None and eps and eps > 0 and ni and ni != 0 and price_on_date:
                shares_est = ni / eps
                if shares_est > 0:
                    dps = abs(div_paid) / shares_est
                    snap["div_yield_pct"] = round(dps / price_on_date * 100, 2)
                else:
                    snap["div_yield_pct"] = None
            else:
                snap["div_yield_pct"] = None

            # altman_z and piotroski_f are computed by get_fundamental_snapshot()
            snaps.append((ticker, snap))

        # --- Step 1b: Normalize P/E and P/B cross-sectionally for this date ---
        # Only stocks available as of rebal_date are included — no look-ahead bias.
        normalize_pe_pb_factors(snaps)

        # --- Step 1c: Score all tickers with normalized valuation metrics ---
        scored: list[tuple[str, float]] = []
        for ticker, snap in snaps:
            s = score_row(snap, weights)
            if s is not None:
                scored.append((ticker, s))

        if not scored:
            continue

        scored.sort(key=lambda x: -x[1])
        top_tickers     = [t for t, _ in scored[:top_n]]
        bottom_tickers  = [t for t, _ in scored[-top_n:]] if long_short else []

        # --- Regime filter: determine equity fraction ---
        if regime_filter and spy_prices is not None:
            risk_on = is_risk_on(spy_prices, rebal_date)
            equity_fraction = 1.0 if risk_on else (1.0 - cash_fraction)
            regime_label = "RISK-ON" if risk_on else "RISK-OFF"
            if risk_on:
                regime_counts["risk_on"] += 1
            else:
                regime_counts["risk_off"] += 1
        else:
            equity_fraction = 1.0
            risk_on = True
            regime_label = "RISK-ON"
            regime_counts["risk_on"] += 1

        logger.debug(
            f"  {rebal_date}: {regime_label} (equity={equity_fraction:.0%})"
        )

        # --- Compute position weights based on weighting_scheme ---
        if top_tickers:
            if weighting_scheme == "risk_parity":
                rp_weights = compute_risk_parity_weights(
                    top_tickers, prices, rebal_date,
                    lookback_days=vol_lookback,
                    min_days=30,
                    max_position=max_single_pos,
                )
                new_holdings: dict[str, float] = {
                    t: w * equity_fraction for t, w in rp_weights.items()
                }
            else:  # equal
                per_stock_weight = equity_fraction / len(top_tickers)
                new_holdings = {t: per_stock_weight for t in top_tickers}
        else:
            new_holdings = {}

        # --- Step 2: Compute portfolio return since last rebalance ---
        if i > 0:
            prev_date  = rebal_dates[i - 1]
            period_ret = _compute_weighted_period_return(
                holdings, prices, prev_date, rebal_date,
                cash_return_per_period=cash_return_per_period,
            )
            if long_short:
                short_ret  = _compute_weighted_period_return(
                    short_holdings, prices, prev_date, rebal_date,
                    cash_return_per_period=0.0,
                )
                period_ret = (period_ret - short_ret) / 2   # net L/S return
            portfolio_val *= (1 + period_ret)

        # --- Step 3: Compute turnover (weight-based) ---
        all_tickers_union = set(holdings.keys()) | set(new_holdings.keys())
        if holdings:
            turnover = sum(
                abs(new_holdings.get(t, 0.0) - holdings.get(t, 0.0))
                for t in all_tickers_union
            ) / 2.0
        else:
            turnover = 1.0

        # --- Step 4: Apply transaction costs ---
        portfolio_val = apply_transaction_cost(portfolio_val, turnover, tc_bps)

        # --- Step 5: Update holdings ---
        holdings = new_holdings
        if long_short and bottom_tickers:
            short_per = equity_fraction / len(bottom_tickers)
            short_holdings = {t: short_per for t in bottom_tickers}

        portfolio_values.append((rebal_date, portfolio_val))
        turnover_history.append(turnover)
        holdings_history.append({
            "date":             rebal_date.isoformat(),
            "tickers":          top_tickers,
            "scores":           [s for _, s in scored[:top_n]],
            "weighting_scheme": weighting_scheme,
            "equity_fraction":  equity_fraction,
            "weights":          {t: round(w, 6) for t, w in holdings.items()},
        })

    if len(portfolio_values) < 3:
        return {"error": "Insufficient data to run backtest."}

    # --- Build return series ---
    pv_series  = pd.Series(
        {d: v for d, v in portfolio_values},
        name="portfolio_value",
    )
    port_rets  = pv_series.pct_change().dropna()

    # Benchmark: SPY buy-and-hold, aligned to rebalancing dates via nearest-day lookup
    bench_rets = pd.Series(dtype=float)
    if spy_prices is not None:
        spy_nav: dict[date, float] = {}
        for d in pv_series.index:
            p = _spy_price_on(spy_prices, d)
            if p is not None:
                spy_nav[d] = p
        if len(spy_nav) >= 2:
            spy_sub    = pd.Series(spy_nav).sort_index()
            bench_rets = spy_sub.pct_change().dropna()

    metrics_port  = compute_metrics(port_rets,  risk_free, periods_per_year)
    metrics_bench = compute_metrics(bench_rets, risk_free, periods_per_year)

    # Relative metrics (beta, alpha, correlation, excess CAGR, IR)
    relative_metrics: dict = {}
    if not bench_rets.empty and "cagr" in metrics_port and "cagr" in metrics_bench:
        relative_metrics = compute_relative_metrics(
            port_rets, bench_rets,
            port_cagr=float(metrics_port["cagr"]),
            bench_cagr=float(metrics_bench["cagr"]),
            risk_free=risk_free,
            periods_per_year=periods_per_year,
        )

    # --- Regime breakdown ---
    regime_returns: dict[str, list] = {"bull": [], "bear": [], "sideways": [], "unknown": []}
    if not spy_regimes.empty:
        for d, r in zip(port_rets.index, port_rets.values):
            regime = str(spy_regimes.get(d, "unknown"))
            regime_returns[regime].append(r)

    regime_metrics: dict[str, dict] = {}
    for regime, rets in regime_returns.items():
        if rets:
            s = pd.Series(rets)
            regime_metrics[regime] = compute_metrics(s, risk_free, periods_per_year)

    return {
        "portfolio_returns":  port_rets,
        "portfolio_values":   pv_series,
        "benchmark_returns":  bench_rets,
        "metrics":            metrics_port,
        "benchmark_metrics":  metrics_bench,
        "relative_metrics":   relative_metrics,
        "regime_metrics":     regime_metrics,
        "holdings_history":   holdings_history,
        "turnover_history":   turnover_history,
        "regime_counts":      regime_counts,
        "params": {
            "top_n":            top_n,
            "freq":             freq,
            "tc_bps":           tc_bps,
            "risk_free":        risk_free,
            "long_short":       long_short,
            "start":            start.isoformat(),
            "end":              end.isoformat(),
            "weighting_scheme": weighting_scheme,
            "regime_filter":    regime_filter,
            "cash_fraction":    cash_fraction,
        },
    }


def _compute_period_return(
    holdings: dict[str, float],
    prices:   dict[str, pd.Series],
    from_date: date,
    to_date:   date,
) -> float:
    """Compute the equal-weighted return of `holdings` from from_date to to_date."""
    if not holdings:
        return 0.0
    returns = []
    for ticker in holdings:
        series = prices.get(ticker)
        if series is None:
            continue
        avail = [d for d in series.index if d <= to_date]
        avail_from = [d for d in series.index if d <= from_date]
        if not avail or not avail_from:
            continue
        p_to   = float(series[max(avail)])
        p_from = float(series[max(avail_from)])
        if p_from > 0:
            returns.append((p_to - p_from) / p_from)
    return float(np.mean(returns)) if returns else 0.0


# ---------------------------------------------------------------------------
# Comparison backtest (all 4 weighting × regime variants)
# ---------------------------------------------------------------------------

def run_comparison_backtest(
    fundamentals_panel: dict[str, list[dict]],
    prices: dict[str, pd.Series],
    weights: dict,
    top_n: int,
    freq: RebalanceFreq,
    tc_bps: float,
    risk_free: float,
    start: date,
    end: date,
    spy_prices: pd.Series | None,
    vol_lookback: int,
    max_single_pos: float,
    cash_fraction: float,
) -> dict[str, dict]:
    """
    Run all 4 combinations of weighting scheme × regime filter and return results.

    Returns
    -------
    dict {variant_name: results_dict}
    """
    variants = [
        ("Equal Weight, No Filter",     "equal",       False),
        ("Equal Weight, Regime Filter", "equal",       True),
        ("Risk Parity, No Filter",      "risk_parity", False),
        ("Risk Parity, Regime Filter",  "risk_parity", True),
    ]

    comparison: dict[str, dict] = {}
    for name, scheme, use_regime in variants:
        logger.info(f"Running variant: {name} ...")
        result = run_backtest(
            fundamentals_panel=fundamentals_panel,
            prices=prices,
            weights=weights,
            top_n=top_n,
            freq=freq,
            tc_bps=tc_bps,
            risk_free=risk_free,
            start=start,
            end=end,
            spy_prices=spy_prices,
            weighting_scheme=scheme,
            regime_filter=use_regime,
            cash_fraction=cash_fraction,
            vol_lookback=vol_lookback,
            max_single_pos=max_single_pos,
        )
        comparison[name] = result

    return comparison


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_comparison_table(comparison_results: dict) -> None:
    """
    Print a formatted side-by-side table for all 4 backtest variants.

    Columns: CAGR%, Sharpe, Sortino, Max DD%, Calmar, Win Rate%
    Annotates 'Risk Parity, Regime Filter' as (recommended).
    """
    RECOMMENDED = "Risk Parity, Regime Filter"

    variant_names = list(comparison_results.keys())
    col_w = 22  # width of each data column

    # Header
    print("\n" + "=" * (28 + col_w * len(variant_names)))
    print("  BACKTEST COMPARISON — ALL VARIANTS")
    print("=" * (28 + col_w * len(variant_names)))

    # Variant header row
    header = f"  {'Metric':<24}"
    for name in variant_names:
        display = name if name != RECOMMENDED else f"{name} (*)"
        header += f"  {display:>{col_w - 2}}"
    print(header)
    print(f"  {'-'*24}" + (f"  {'-'*(col_w-2)}" * len(variant_names)))

    def _fmt(val, suffix="", fmt=".2f"):
        if isinstance(val, (int, float)):
            return f"{val:{fmt}}{suffix}"
        return "n/a"

    metrics_rows = [
        ("CAGR",         "cagr",         ".2f", "%"),
        ("Sharpe",       "sharpe",       ".3f", ""),
        ("Sortino",      "sortino",      ".3f", ""),
        ("Max Drawdown", "max_drawdown", ".2f", "%"),
        ("Calmar",       "calmar",       ".3f", ""),
        ("Win Rate",     "win_rate",     ".1f", "%"),
    ]

    for label, key, fmt, suffix in metrics_rows:
        row_str = f"  {label:<24}"
        for name in variant_names:
            m = comparison_results[name].get("metrics", {})
            val = m.get(key, "n/a")
            cell = _fmt(val, suffix=suffix, fmt=fmt)
            row_str += f"  {cell:>{col_w - 2}}"
        print(row_str)

    # Regime counts row
    print(f"  {'-'*24}" + (f"  {'-'*(col_w-2)}" * len(variant_names)))
    row_str = f"  {'Risk-On Periods':<24}"
    for name in variant_names:
        rc = comparison_results[name].get("regime_counts", {})
        cell = str(rc.get("risk_on", "n/a"))
        row_str += f"  {cell:>{col_w - 2}}"
    print(row_str)

    row_str = f"  {'Risk-Off Periods':<24}"
    for name in variant_names:
        rc = comparison_results[name].get("regime_counts", {})
        cell = str(rc.get("risk_off", "n/a"))
        row_str += f"  {cell:>{col_w - 2}}"
    print(row_str)

    print("=" * (28 + col_w * len(variant_names)))
    print("  (*) Recommended variant: Risk Parity + Regime Filter\n")


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def _serialise_value(v):
    """
    Recursively convert a value to something json.dump can handle.
    Converts date/datetime keys in dicts and date/datetime values to ISO strings.
    """
    import datetime as _dt
    if isinstance(v, dict):
        return {
            (k.isoformat() if isinstance(k, (_dt.date, _dt.datetime)) else k): _serialise_value(val)
            for k, val in v.items()
        }
    if isinstance(v, list):
        return [_serialise_value(i) for i in v]
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None   # replace NaN/inf with null
    return v


def save_backtest_results(results: dict, label: str = "default") -> str:
    """Persist backtest results (excluding raw return series) to JSON."""
    path = os.path.join(RESULTS_DIR, f"backtest_{label}.json")

    # Drop raw pd.Series objects (too large / not JSON-serialisable)
    serialisable = {k: v for k, v in results.items() if not isinstance(v, pd.Series)}

    # Convert portfolio_values Series: date keys to ISO string keys
    pv = results.get("portfolio_values")
    if isinstance(pv, pd.Series):
        serialisable["portfolio_values"] = {
            (k.isoformat() if hasattr(k, "isoformat") else str(k)): float(v)
            for k, v in pv.items()
        }

    # Recursively sanitise any remaining date keys or values
    serialisable = _serialise_value(serialisable)

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Backtest results saved to {path}")
    return path


def load_backtest_results(label: str = "default") -> dict | None:
    path = os.path.join(RESULTS_DIR, f"backtest_{label}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_backtest_summary(results: dict) -> None:
    """Print a formatted backtest performance summary."""
    m  = results.get("metrics", {})
    bm = results.get("benchmark_metrics", {})
    rm = results.get("relative_metrics", {})
    p  = results.get("params", {})
    rc = results.get("regime_counts", {})

    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS — CLAYTON SCORE STRATEGY")
    print("=" * 65)
    print(f"  Period:        {p.get('start')} to {p.get('end')}")
    print(f"  Rebalancing:   {p.get('freq')}")
    print(f"  Portfolio:     Top {p.get('top_n')} stocks  |  "
          f"{'Long-Short' if p.get('long_short') else 'Long-Only'}")
    print(f"  Costs:         {p.get('tc_bps')} bps one-way")
    print(f"  Weighting:     {p.get('weighting_scheme', 'equal')}")
    regime_status = "ON" if p.get("regime_filter", False) else "OFF"
    print(f"  Regime Filter: {regime_status}", end="")
    if rc:
        print(
            f"  (Risk-On: {rc.get('risk_on', 0)} periods, "
            f"Risk-Off: {rc.get('risk_off', 0)} periods)",
            end="",
        )
    print()
    print("-" * 65)
    print(f"  {'Metric':<22} {'Strategy':>12} {'Benchmark (SPY)':>16}")
    print(f"  {'-'*22} {'-'*12} {'-'*16}")

    def _fv(d, key, fmt=".2f", suffix=""):
        v = d.get(key, "n/a")
        return f"{v:{fmt}}{suffix}" if isinstance(v, (int, float)) else "n/a"

    def row(label, key, fmt=".2f", suffix=""):
        print(f"  {label:<22} {_fv(m, key, fmt, suffix):>12} {_fv(bm, key, fmt, suffix):>16}")

    row("CAGR",          "cagr",         suffix="%")
    row("Total Return",  "total_return",  suffix="%")
    row("Sharpe Ratio",  "sharpe",        fmt=".3f")
    row("Sortino Ratio", "sortino",       fmt=".3f")
    row("Calmar Ratio",  "calmar",        fmt=".3f")
    row("Max Drawdown",  "max_drawdown",  suffix="%")
    row("Volatility",    "volatility",    suffix="%")
    row("Win Rate",      "win_rate",      suffix="%")
    row("Best Period",   "best_period",   suffix="%")
    row("Worst Period",  "worst_period",  suffix="%")

    # Relative metrics (only shown when benchmark data is available)
    if rm:
        print(f"\n  {'--- vs SPY Benchmark ---'}")
        print(f"  {'-'*50}")
        print(f"  {'Excess CAGR':<22} {_fv(rm, 'excess_cagr', suffix='%'):>12}")
        print(f"  {'Beta':<22} {_fv(rm, 'beta', fmt='.4f'):>12}")
        print(f"  {'Jensen Alpha (ann%)':<22} {_fv(rm, 'alpha', suffix='%'):>12}")
        print(f"  {'Correlation':<22} {_fv(rm, 'correlation', fmt='.4f'):>12}")
        print(f"  {'Information Ratio':<22} {_fv(rm, 'information_ratio', fmt='.4f'):>12}")

    # Regime breakdown
    regime_m = results.get("regime_metrics", {})
    if regime_m:
        print(f"\n  Regime Breakdown")
        print(f"  {'-'*40}")
        for regime, rm_reg in regime_m.items():
            sharpe   = rm_reg.get("sharpe", "n/a")
            cagr     = rm_reg.get("cagr", "n/a")
            n        = rm_reg.get("n_periods", 0)
            cagr_s   = f"{cagr:.1f}%" if isinstance(cagr, float) else "n/a"
            sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, float) else "n/a"
            print(f"  {regime.capitalize():<12}  CAGR: {cagr_s:>8}  Sharpe: {sharpe_s:>6}  n={n}")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from historical_data import load_all_fundamentals, load_prices
    from config import WEIGHTS as DEFAULT_WEIGHTS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Clayton Score backtest engine")
    parser.add_argument("--run",        action="store_true",       help="Run backtest")
    parser.add_argument("--top-n",      type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance",  default="monthly",         choices=["monthly", "weekly", "daily"])
    parser.add_argument("--tc-bps",     type=float, default=DEFAULT_TC_BPS)
    parser.add_argument("--long-short", action="store_true")
    parser.add_argument("--start",      default=BACKTEST_START.isoformat())
    parser.add_argument("--end",        default=BACKTEST_END.isoformat())
    # New weighting / regime CLI flags
    parser.add_argument(
        "--weighting",
        choices=["equal", "risk_parity"],
        default=DEFAULT_WEIGHTING,
        help="Position weighting scheme (default from WEIGHTING_SCHEME env var)",
    )
    regime_group = parser.add_mutually_exclusive_group()
    regime_group.add_argument(
        "--regime-filter",
        dest="regime_filter",
        action="store_true",
        default=DEFAULT_REGIME_FILTER,
        help="Enable regime filter: go partial-cash when SPY < 200-day MA",
    )
    regime_group.add_argument(
        "--no-regime-filter",
        dest="regime_filter",
        action="store_false",
        help="Disable regime filter (always fully invested)",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run all 4 weighting × regime variants and print comparison table",
    )
    args = parser.parse_args()

    if not args.run and not args.comparison:
        parser.print_help()
    else:
        logger.info("Loading historical data...")
        fund_all   = load_all_fundamentals()
        fund_panel = {t: d.get("records", []) for t, d in fund_all.items()}

        from historical_data import load_all_prices
        prices = load_all_prices()

        spy = load_prices("SPY")

        start_dt = date.fromisoformat(args.start)
        end_dt   = date.fromisoformat(args.end)

        if args.comparison:
            logger.info("Running all 4 comparison variants...")
            comparison = run_comparison_backtest(
                fundamentals_panel=fund_panel,
                prices=prices,
                weights=DEFAULT_WEIGHTS,
                top_n=args.top_n,
                freq=args.rebalance,
                tc_bps=args.tc_bps,
                risk_free=DEFAULT_RISK_FREE,
                start=start_dt,
                end=end_dt,
                spy_prices=spy,
                vol_lookback=DEFAULT_VOL_LOOKBACK,
                max_single_pos=DEFAULT_MAX_SINGLE_POS,
                cash_fraction=DEFAULT_CASH_FRACTION,
            )
            print_comparison_table(comparison)
        else:
            logger.info(
                f"Running backtest: {len(fund_panel)} tickers, "
                f"{args.rebalance} rebalancing, "
                f"weighting={args.weighting}, "
                f"regime_filter={args.regime_filter} ..."
            )
            results = run_backtest(
                fundamentals_panel=fund_panel,
                prices=prices,
                weights=DEFAULT_WEIGHTS,
                top_n=args.top_n,
                freq=args.rebalance,
                tc_bps=args.tc_bps,
                long_short=args.long_short,
                start=start_dt,
                end=end_dt,
                spy_prices=spy,
                weighting_scheme=args.weighting,
                regime_filter=args.regime_filter,
                cash_fraction=DEFAULT_CASH_FRACTION,
                vol_lookback=DEFAULT_VOL_LOOKBACK,
                max_single_pos=DEFAULT_MAX_SINGLE_POS,
            )

            print_backtest_summary(results)
            save_backtest_results(
                results,
                label=f"{args.rebalance}_top{args.top_n}_{args.weighting}"
                      f"{'_regime' if args.regime_filter else ''}",
            )
