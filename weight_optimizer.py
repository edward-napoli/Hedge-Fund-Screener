"""
weight_optimizer.py — Walk-forward weight optimization for the Clayton Score.

Uses scipy.optimize.differential_evolution to maximise Sharpe ratio on
a rolling training window, then evaluates out-of-sample on the following
test window.

Speed optimizations vs v1:
  - Factor matrix pre-computed ONCE before the optimization loop (eliminates
    all per-evaluation disk I/O and EPS-growth re-computation)
  - Period returns pre-computed ONCE per window (eliminates price lookups)
  - ThreadPoolExecutor used as workers= so objective evaluations run in
    parallel across all CPU cores without pickling overhead (Windows-safe)
  - Progress logged every 10 iterations via callback

Walk-forward schedule:
  --optimize-fast : popsize=5,  maxiter=30  (~minutes)
  default         : popsize=8,  maxiter=50  (~tens of minutes)
  --optimize-full : popsize=12, maxiter=80  (~hours, thorough)

CLI usage:
    python weight_optimizer.py --run
    python weight_optimizer.py --optimize-fast
    python weight_optimizer.py --optimize-full
    python weight_optimizer.py --show-weights
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import differential_evolution, OptimizeResult

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR          = "cache/backtest"
OPTIMAL_WEIGHTS_FILE = os.path.join(RESULTS_DIR, "optimal_weights.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Weight keys and bounds
# ---------------------------------------------------------------------------

WEIGHT_KEYS = [
    "E1", "E3", "E5", "Ef",
    "Ra", "Re", "Rc",
    "C", "Z", "F",
    "A",
    "Y_outer", "Pe_coef", "Pb_coef",
]

WEIGHT_BOUNDS = {
    "E1":      (0.0, 2.0),
    "E3":      (0.0, 3.0),
    "E5":      (0.0, 4.0),
    "Ef":      (0.0, 6.0),
    "Ra":      (0.0, 3.0),
    "Re":      (0.0, 3.0),
    "Rc":      (0.0, 5.0),
    "C":       (0.0, 4.0),
    "Z":       (0.0, 8.0),
    "F":       (0.0, 8.0),
    "A":       (0.0, 1.0),
    "Y_outer": (0.0, 8.0),
    "Pe_coef": (0.0, 10.0),
    "Pb_coef": (0.0, 8.0),
}


def _vec_to_weights(vec: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, vec)}


def _weights_to_vec(weights: dict) -> np.ndarray:
    return np.array([weights[k] for k in WEIGHT_KEYS], dtype=float)


def _bounds_list() -> list[tuple[float, float]]:
    return [WEIGHT_BOUNDS[k] for k in WEIGHT_KEYS]


# ---------------------------------------------------------------------------
# In-memory fundamental helpers (bypass disk I/O)
# ---------------------------------------------------------------------------

def _snapshot_from_records(
    records: list[dict],
    as_of: date,
    use_quarterly: bool = True,
) -> dict | None:
    """
    Return the most recent fundamental record available on `as_of`,
    working entirely from an already-loaded records list (no disk I/O).
    """
    best = None
    for rec in records:
        ptype = rec.get("period_type", "unknown")
        if not use_quarterly and ptype == "quarterly":
            continue

        filed_raw = rec.get("filed_date") or rec.get("period_end")
        if not filed_raw:
            continue
        try:
            filed = date.fromisoformat(filed_raw[:10])
        except ValueError:
            continue

        # If no filed_date, assume 45-day reporting lag
        if not rec.get("filed_date"):
            try:
                period_end = date.fromisoformat(rec["period_end"])
                filed = period_end + timedelta(days=45)
            except (ValueError, KeyError):
                continue

        if filed > as_of:
            continue

        if best is None or rec.get("period_end", "") > best.get("period_end", ""):
            best = rec

    return best


def _prior_snapshot_from_records(
    records: list[dict],
    current_best: dict,
) -> dict | None:
    """
    Find the prior-year annual record needed for Piotroski F delta criteria.
    Returns the most recent record whose period_end is at least 9 months
    before current_best's period_end, or None if no such record exists.
    """
    if not current_best or not current_best.get("period_end"):
        return None
    best_end = date.fromisoformat(current_best["period_end"][:10])
    cutoff = best_end - timedelta(days=274)  # ~9 months back
    candidates = [
        r for r in records
        if r is not current_best
        and r.get("period_end")
        and date.fromisoformat(r["period_end"][:10]) <= cutoff
    ]
    return max(candidates, key=lambda r: r["period_end"]) if candidates else None


def _eps_growth_from_records(records: list[dict], as_of: date, years: int) -> float | None:
    """
    Compute annualised EPS growth over `years` years from in-memory records.
    No disk I/O — records must already be loaded.
    """
    annual = [
        r for r in records
        if r.get("period_type") == "annual"
        and r.get("eps_basic") is not None
        and r.get("period_end")
    ]
    annual = [r for r in annual if date.fromisoformat(r["period_end"]) <= as_of]
    annual.sort(key=lambda r: r["period_end"])

    if len(annual) < years + 1:
        return None

    eps_recent = annual[-1]["eps_basic"]
    eps_past   = annual[-(years + 1)]["eps_basic"]

    if eps_past is None or eps_past == 0:
        return None
    try:
        growth = ((eps_recent / eps_past) ** (1 / years) - 1) * 100
        return round(growth, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Factor matrix pre-computation
# ---------------------------------------------------------------------------

#: {date_iso: [(ticker, factor_dict), ...]}
FactorMatrix = dict[str, list[tuple[str, dict]]]

#: [{ticker: period_return}, ...] — index matches rebal_dates
PriceReturns = list[dict[str, float]]


def build_factor_matrix(
    fundamentals_panel: dict[str, dict],
    prices:             dict[str, pd.Series],
    rebal_dates:        list[date],
) -> FactorMatrix:
    """
    Pre-compute all factor values for every ticker on every rebalancing date.

    This is called ONCE before the optimisation loop.  All subsequent
    objective function evaluations read from this in-memory structure,
    eliminating all disk I/O and EPS-growth re-computation.

    Parameters
    ----------
    fundamentals_panel : dict
        {ticker: full fundamentals dict (with "records" list)} — already in memory.
    prices : dict
        {ticker: pd.Series(date -> close)}
    rebal_dates : list[date]
        Rebalancing dates covering the window.

    Returns
    -------
    FactorMatrix
        {date_iso: [(ticker, factor_dict), ...]}
    """
    from historical_data import _compute_altman_z, _compute_piotroski_f

    matrix: FactorMatrix = {}
    n = len(rebal_dates)
    t0 = time.monotonic()

    for idx, rebal_date in enumerate(rebal_dates):
        date_str = rebal_date.isoformat()
        entries: list[tuple[str, dict]] = []

        for ticker, fund_data in fundamentals_panel.items():
            records = fund_data.get("records", [])
            snap = _snapshot_from_records(records, rebal_date)
            if not snap:
                continue

            prior = _prior_snapshot_from_records(records, snap)

            ni  = snap.get("net_income")
            te  = snap.get("total_equity") or 0.0
            td  = snap.get("total_debt")   or 0.0
            ic  = te + td
            eps = snap.get("eps_basic")

            # Price on rebalancing date (for P/E and div yield)
            price_on_date: float | None = None
            price_series = prices.get(ticker)
            if price_series is not None:
                avail = [d for d in price_series.index if d <= rebal_date]
                if avail:
                    price_on_date = float(price_series[max(avail)])

            # ROIC proxy: net_income / (total_equity + total_debt)
            roic = round(ni / ic * 100, 2) if (ni and ic > 0) else None

            # P/E = price / eps_basic
            pe_ratio = (
                round(price_on_date / eps, 2)
                if (price_on_date and eps and eps > 0)
                else None
            )

            # Payout ratio and div yield via shares proxy (net_income / eps_basic)
            div_paid = snap.get("dividends_paid")
            if div_paid is not None and ni and ni > 0:
                payout_ratio_pct = round(abs(div_paid) / ni * 100, 2)
            else:
                payout_ratio_pct = None

            if div_paid is not None and eps and eps > 0 and ni and ni != 0 and price_on_date:
                shares_est = ni / eps
                if shares_est > 0:
                    dps = abs(div_paid) / shares_est
                    div_yield_pct = round(dps / price_on_date * 100, 2)
                else:
                    div_yield_pct = None
            else:
                div_yield_pct = None

            factors: dict = {
                "eps_1y_growth":    _eps_growth_from_records(records, rebal_date, 1),
                "eps_3y_growth":    _eps_growth_from_records(records, rebal_date, 3),
                "eps_5y_growth":    _eps_growth_from_records(records, rebal_date, 5),
                "eps_fwd_growth":   None,
                "roa":              snap.get("roa"),
                "roe":              snap.get("roe"),
                "roic":             roic,
                "current_ratio":    snap.get("current_ratio"),
                "altman_z":         _compute_altman_z(snap),
                "piotroski_f":      _compute_piotroski_f(snap, prior),
                "net_income_usd_m": ni / 1_000_000 if ni else None,
                "div_yield_pct":    div_yield_pct,
                "payout_ratio_pct": payout_ratio_pct,
                "pe_ratio":         pe_ratio,
                "pb_ratio":         None,
            }
            entries.append((ticker, factors))

        matrix[date_str] = entries

        if (idx + 1) % 12 == 0 or idx == n - 1:
            elapsed = time.monotonic() - t0
            logger.info(
                f"  Factor matrix: {idx + 1}/{n} dates "
                f"({len(entries)} stocks on {date_str})  [{elapsed:.0f}s]"
            )

    total_obs = sum(len(v) for v in matrix.values())
    logger.info(
        f"Factor matrix built: {n} dates x avg "
        f"{total_obs // max(n, 1)} stocks = {total_obs:,} observations "
        f"in {time.monotonic() - t0:.1f}s"
    )

    # Diagnostic: confirm key factors contain real non-zero data
    def _fstats(key: str) -> str:
        vals = [
            float(f[key])
            for entries in matrix.values()
            for _, f in entries
            if f.get(key) is not None
        ]
        if not vals:
            return "0 obs  mean=n/a  std=n/a"
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return "0 finite obs"
        return f"{len(arr):,} obs  mean={arr.mean():.3f}  std={arr.std():.3f}"

    for fname in ["roic", "pe_ratio", "div_yield_pct", "altman_z", "piotroski_f"]:
        logger.info(f"  Factor [{fname}]: {_fstats(fname)}")

    return matrix


def build_price_returns(
    prices:      dict[str, pd.Series],
    rebal_dates: list[date],
) -> PriceReturns:
    """
    Pre-compute period returns between consecutive rebalancing dates for
    every ticker.  Index i holds returns from rebal_dates[i-1] to
    rebal_dates[i]; index 0 is always empty.

    Returns
    -------
    PriceReturns
        List of {ticker: float} dicts, one per rebalancing date.
    """
    result: PriceReturns = [{}]   # index 0 — no prior period

    for i in range(1, len(rebal_dates)):
        prev_date = rebal_dates[i - 1]
        curr_date = rebal_dates[i]
        period: dict[str, float] = {}

        for ticker, series in prices.items():
            idx_prev = [d for d in series.index if d <= prev_date]
            idx_curr = [d for d in series.index if d <= curr_date]
            if not idx_prev or not idx_curr:
                continue
            p_from = float(series[max(idx_prev)])
            p_to   = float(series[max(idx_curr)])
            if p_from > 0:
                period[ticker] = (p_to - p_from) / p_from

        result.append(period)

    return result


# ---------------------------------------------------------------------------
# Fast in-memory Sharpe computation (the hot inner loop)
# ---------------------------------------------------------------------------

def _safe_f(v) -> float:
    """Return float or 0.0, handling None / NaN / inf."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        import math
        return f if not (math.isnan(f) or math.isinf(f)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _score_factors(factors: dict, w: dict) -> float:
    """Apply Clayton Score formula to a pre-computed factor dict."""
    E1  = _safe_f(factors.get("eps_1y_growth"))
    E3  = _safe_f(factors.get("eps_3y_growth"))
    E5  = _safe_f(factors.get("eps_5y_growth"))
    Ef  = _safe_f(factors.get("eps_fwd_growth"))
    Ra  = _safe_f(factors.get("roa"))
    Re  = _safe_f(factors.get("roe"))
    Rc  = _safe_f(factors.get("roic"))
    C   = _safe_f(factors.get("current_ratio"))
    Z   = _safe_f(factors.get("altman_z"))
    F   = _safe_f(factors.get("piotroski_f"))
    A   = _safe_f(factors.get("net_income_usd_m"))
    Y   = _safe_f(factors.get("div_yield_pct"))
    Pr  = _safe_f(factors.get("payout_ratio_pct"))
    Pe  = _safe_f(factors.get("pe_ratio"))
    Pb  = _safe_f(factors.get("pb_ratio"))

    return (
        w["E1"]*E1 + w["E3"]*E3 + w["E5"]*E5 + w["Ef"]*Ef
        + Ra + Re + w["Rc"]*Rc
        + w["C"]*C + w["Z"]*Z + w["F"]*F
        + w["A"]*A
        + w["Y_outer"] * (Y * (2 - Pr / 100) - (w["Pe_coef"]*Pe + w["Pb_coef"]*Pb))
    )


def _fast_sharpe(
    vec:             np.ndarray,
    factor_matrix:   FactorMatrix,
    price_returns:   PriceReturns,
    rebal_dates:     list[date],
    top_n:           int,
    tc_bps:          float,
    risk_free:       float,
    periods_per_year: int,
) -> float:
    """
    Compute Sharpe ratio for one weight vector using pre-computed data.
    Pure arithmetic — no disk I/O, no file reads.  Called millions of times.
    """
    w = _vec_to_weights(vec)
    portfolio_val = 1_000_000.0
    holdings: list[str] = []
    pv_list: list[float] = [portfolio_val]

    for i, rebal_date in enumerate(rebal_dates):
        date_str = rebal_date.isoformat()
        entries  = factor_matrix.get(date_str, [])

        # Score and rank
        scored = [(ticker, _score_factors(factors, w)) for ticker, factors in entries]
        if not scored:
            pv_list.append(portfolio_val)
            continue
        scored.sort(key=lambda x: -x[1])
        new_holdings = [t for t, _ in scored[:top_n]]

        # Portfolio return since last rebalance
        if i > 0 and holdings:
            period_rets = price_returns[i]
            rets = [period_rets.get(t, 0.0) for t in holdings]
            portfolio_val *= 1.0 + (float(np.mean(rets)) if rets else 0.0)

        # Transaction costs
        if holdings:
            old_set = set(holdings)
            new_set = set(new_holdings)
            turnover = len(old_set.symmetric_difference(new_set)) / max(len(old_set), len(new_set))
        else:
            turnover = 1.0
        portfolio_val *= 1.0 - turnover * tc_bps / 10_000.0

        holdings = new_holdings
        pv_list.append(portfolio_val)

    if len(pv_list) < 3:
        return 0.0

    arr  = np.array(pv_list, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    std  = float(rets.std())
    if std == 0.0:
        return 0.0

    rf_per = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    sharpe = float((rets.mean() - rf_per) / std * np.sqrt(periods_per_year))
    return sharpe


# ---------------------------------------------------------------------------
# Single-window optimisation (with pre-computed cache + thread workers)
# ---------------------------------------------------------------------------

def optimise_window(
    factor_matrix:    FactorMatrix,
    price_returns:    PriceReturns,
    rebal_dates:      list[date],
    top_n:            int   = 25,
    tc_bps:           float = 10.0,
    risk_free:        float = 0.04,
    freq:             str   = "monthly",
    popsize:          int   = 8,
    maxiter:          int   = 50,
    seed:             int   = 42,
    n_workers:        int   = -1,
) -> tuple[dict, OptimizeResult]:
    """
    Optimise Clayton Score weights on a pre-computed factor matrix.

    Parameters
    ----------
    factor_matrix : FactorMatrix
        Pre-computed {date_iso: [(ticker, factor_dict), ...]} for the window.
    price_returns : PriceReturns
        Pre-computed period returns matching rebal_dates.
    rebal_dates : list[date]
        Rebalancing dates for the training window.
    top_n, tc_bps, risk_free, freq : see run_backtest().
    popsize : int
        Population size per dimension for differential_evolution.
    maxiter : int
        Maximum number of generations.
    seed : int
        Random seed.
    n_workers : int
        Number of parallel workers. -1 = all CPUs.

    Returns
    -------
    (optimal_weights_dict, scipy_result)
    """
    periods_per_year = {"monthly": 12, "weekly": 52, "daily": 252}[freq]
    n_cpu = (os.cpu_count() or 1) if n_workers == -1 else max(1, n_workers)

    logger.info(
        f"Optimising window ({len(rebal_dates)} periods, "
        f"popsize={popsize}, maxiter={maxiter}, workers={n_cpu})"
    )

    # Progress tracking (mutable container for closure)
    _iter    = [0]
    _best    = [float("inf")]
    _t_start = [time.monotonic()]

    def _progress_callback(xk, convergence=0.0):
        _iter[0] += 1
        cur = -_fast_sharpe(xk, factor_matrix, price_returns, rebal_dates,
                            top_n, tc_bps, risk_free, periods_per_year)
        if cur < _best[0]:
            _best[0] = cur
        if _iter[0] % 10 == 0:
            elapsed = time.monotonic() - _t_start[0]
            logger.info(
                f"  Iter {_iter[0]:>4}/{maxiter}  "
                f"best Sharpe: {-_best[0]:.4f}  "
                f"convergence: {convergence:.4f}  "
                f"elapsed: {elapsed:.0f}s"
            )

    # Build thread-safe objective (threads share memory, no pickling needed)
    def obj(vec: np.ndarray) -> float:
        return -_fast_sharpe(
            vec, factor_matrix, price_returns, rebal_dates,
            top_n, tc_bps, risk_free, periods_per_year,
        )

    bounds = _bounds_list()

    with ThreadPoolExecutor(max_workers=n_cpu) as executor:
        result = differential_evolution(
            obj,
            bounds=bounds,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            tol=1e-4,
            workers=executor.map,   # thread pool — no pickling, Windows-safe
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            disp=False,
            callback=_progress_callback,
        )

    optimal = _vec_to_weights(result.x)
    logger.info(
        f"  Done — best Sharpe = {-result.fun:.4f}  "
        f"({result.nit} iters, {result.nfev} evaluations)"
    )
    return optimal, result


# ---------------------------------------------------------------------------
# Walk-forward optimisation
# ---------------------------------------------------------------------------

def walk_forward_optimise(
    fundamentals_panel: dict,
    prices:             dict,
    universe_start:     date,
    universe_end:       date,
    train_years:        int   = 7,
    test_years:         int   = 1,
    step_years:         int   = 1,
    top_n:              int   = 25,
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
    freq:               str   = "monthly",
    popsize:            int   = 8,
    maxiter:            int   = 50,
    n_workers:          int   = -1,
) -> dict:
    """
    Run walk-forward weight optimisation.

    For each window:
      - Pre-compute factor matrix and price returns for the training window
      - Optimise weights via differential_evolution on that window
      - Evaluate out-of-sample on the test window

    Returns the same result structure as before:
      {windows, average_optimal_weights, oos_combined_metrics, params}
    """
    from backtest import run_backtest, compute_metrics, get_rebalance_dates
    from config import WEIGHTS as DEFAULT_WEIGHTS

    windows: list[dict] = []
    all_oos_returns: list[pd.Series] = []
    periods_per_year = {"monthly": 12, "weekly": 52, "daily": 252}[freq]

    cur_year = universe_start.year
    end_year = universe_end.year - train_years - test_years + 1

    if cur_year > end_year:
        logger.warning("Not enough history for walk-forward. Reduce train/test years.")
        return {"error": "Insufficient history for walk-forward optimisation."}

    window_num = 0
    while cur_year <= end_year:
        window_num += 1
        train_start = date(cur_year, 1, 1)
        train_end   = date(cur_year + train_years, 12, 31)
        test_start  = date(cur_year + train_years + 1, 1, 1)
        test_end    = date(
            min(cur_year + train_years + test_years, universe_end.year), 12, 31
        )

        if test_end > universe_end:
            break

        logger.info(
            f"\n[Window {window_num}] Train: {train_start} to {train_end} | "
            f"Test: {test_start} to {test_end}"
        )

        # Pre-compute factor matrix for training window
        train_dates = get_rebalance_dates(train_start, train_end, freq)
        logger.info(f"  Pre-computing factor matrix ({len(train_dates)} rebal dates)...")
        t_fm = time.monotonic()
        factor_matrix  = build_factor_matrix(fundamentals_panel, prices, train_dates)
        price_returns  = build_price_returns(prices, train_dates)
        logger.info(f"  Factor matrix ready in {time.monotonic() - t_fm:.1f}s")

        # Optimise
        opt_weights, scipy_res = optimise_window(
            factor_matrix=factor_matrix,
            price_returns=price_returns,
            rebal_dates=train_dates,
            top_n=top_n,
            tc_bps=tc_bps,
            risk_free=risk_free,
            freq=freq,
            popsize=popsize,
            maxiter=maxiter,
            n_workers=n_workers,
        )

        # Train Sharpe (re-use pre-computed matrix, no extra backtest run)
        train_sharpe = _fast_sharpe(
            _weights_to_vec(opt_weights),
            factor_matrix, price_returns, train_dates,
            top_n, tc_bps, risk_free, periods_per_year,
        )

        # Evaluate on test window
        test_res = run_backtest(
            fundamentals_panel=fundamentals_panel,
            prices=prices,
            weights=opt_weights,
            top_n=top_n,
            freq=freq,
            tc_bps=tc_bps,
            risk_free=risk_free,
            start=test_start,
            end=test_end,
        )
        default_res = run_backtest(
            fundamentals_panel=fundamentals_panel,
            prices=prices,
            weights=DEFAULT_WEIGHTS,
            top_n=top_n,
            freq=freq,
            tc_bps=tc_bps,
            risk_free=risk_free,
            start=test_start,
            end=test_end,
        )

        window_entry = {
            "train_start":         train_start.isoformat(),
            "train_end":           train_end.isoformat(),
            "test_start":          test_start.isoformat(),
            "test_end":            test_end.isoformat(),
            "optimal_weights":     opt_weights,
            "train_sharpe":        round(train_sharpe, 3),
            "test_sharpe":         round(test_res.get("metrics", {}).get("sharpe", 0.0), 3),
            "default_test_sharpe": round(default_res.get("metrics", {}).get("sharpe", 0.0), 3),
            "test_metrics":        test_res.get("metrics", {}),
            "default_metrics":     default_res.get("metrics", {}),
            "n_evaluations":       scipy_res.nfev,
        }
        windows.append(window_entry)

        logger.info(
            f"  Window {window_num} done — train Sharpe={train_sharpe:.3f}, "
            f"test Sharpe={window_entry['test_sharpe']:.3f} "
            f"(default: {window_entry['default_test_sharpe']:.3f})"
        )

        if isinstance(test_res.get("portfolio_returns"), pd.Series):
            all_oos_returns.append(test_res["portfolio_returns"])

        cur_year += step_years

    # Aggregate
    avg_weights: dict[str, float] = {}
    if windows:
        for key in WEIGHT_KEYS:
            avg_weights[key] = round(
                float(np.mean([w["optimal_weights"].get(key, 0) for w in windows])), 4
            )

    oos_metrics: dict = {}
    if all_oos_returns:
        combined    = pd.concat(all_oos_returns).sort_index()
        oos_metrics = compute_metrics(combined, risk_free, periods_per_year)

    results = {
        "windows":                 windows,
        "average_optimal_weights": avg_weights,
        "oos_combined_metrics":    oos_metrics,
        "params": {
            "train_years": train_years,
            "test_years":  test_years,
            "step_years":  step_years,
            "top_n":       top_n,
            "tc_bps":      tc_bps,
            "freq":        freq,
            "popsize":     popsize,
            "maxiter":     maxiter,
        },
    }

    _save_optimal_weights(avg_weights)
    return results


# ---------------------------------------------------------------------------
# Optimal weights persistence
# ---------------------------------------------------------------------------

def _save_optimal_weights(weights: dict) -> None:
    data = {"weights": weights, "generated": date.today().isoformat()}
    with open(OPTIMAL_WEIGHTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Optimal weights saved to {OPTIMAL_WEIGHTS_FILE}")


def load_optimal_weights() -> dict | None:
    if not os.path.exists(OPTIMAL_WEIGHTS_FILE):
        return None
    try:
        with open(OPTIMAL_WEIGHTS_FILE, "r") as f:
            return json.load(f).get("weights")
    except Exception:
        return None


def check_weights_staleness() -> str | None:
    """
    Compare the mtime of optimal_weights.json against the most recent
    backtest results file in cache/backtest/.

    Returns a warning string if weights pre-date the newest backtest results,
    or None if weights are up to date or no comparison is possible.
    """
    if not os.path.exists(OPTIMAL_WEIGHTS_FILE):
        return None

    weights_mtime = os.path.getmtime(OPTIMAL_WEIGHTS_FILE)

    # Find the newest non-weights JSON file in the results directory
    try:
        backtest_files = [
            os.path.join(RESULTS_DIR, f)
            for f in os.listdir(RESULTS_DIR)
            if f.endswith(".json") and f != "optimal_weights.json"
        ]
    except FileNotFoundError:
        return None

    if not backtest_files:
        return None

    newest_backtest      = max(backtest_files, key=os.path.getmtime)
    newest_backtest_mtime = os.path.getmtime(newest_backtest)

    if newest_backtest_mtime > weights_mtime:
        from datetime import datetime
        w_dt = datetime.fromtimestamp(weights_mtime).strftime("%Y-%m-%d %H:%M")
        b_dt = datetime.fromtimestamp(newest_backtest_mtime).strftime("%Y-%m-%d %H:%M")
        newest_name = os.path.basename(newest_backtest)
        return (
            f"[WARN] Cached weights ({w_dt}) pre-date the newest backtest results "
            f"'{newest_name}' ({b_dt}). "
            f"Weights may be stale — consider re-running: python main.py --optimize-full"
        )

    return None


def get_active_weights() -> dict:
    """Return optimal weights if USE_OPTIMIZED_WEIGHTS=true, else defaults."""
    from config import WEIGHTS
    if os.getenv("USE_OPTIMIZED_WEIGHTS", "false").lower() == "true":
        opt = load_optimal_weights()
        if opt:
            logger.info("Using optimised weights.")
            return opt
    return WEIGHTS


# ---------------------------------------------------------------------------
# Pretty-print walk-forward summary
# ---------------------------------------------------------------------------

def print_wf_summary(results: dict) -> None:
    p = results.get("params", {})
    windows = results.get("windows", [])

    print("\n" + "=" * 70)
    print("  WALK-FORWARD OPTIMISATION RESULTS")
    print("=" * 70)
    print(
        f"  Train: {p.get('train_years')}yr  |  Test: {p.get('test_years')}yr  |  "
        f"Step: {p.get('step_years')}yr  |  Top-{p.get('top_n')}  |  {p.get('freq')}  |  "
        f"popsize={p.get('popsize')}  maxiter={p.get('maxiter')}"
    )
    print(f"  {'Window':<25} {'Train Sharpe':>13} {'Test Sharpe':>11} {'Default Sharpe':>15} {'Evals':>7}")
    print(f"  {'-'*25} {'-'*13} {'-'*11} {'-'*15} {'-'*7}")
    for w in windows:
        label = f"{w['train_start'][:7]}->{w['test_end'][:7]}"
        print(
            f"  {label:<25} {w['train_sharpe']:>13.3f} "
            f"{w['test_sharpe']:>11.3f} {w['default_test_sharpe']:>15.3f} "
            f"{w.get('n_evaluations', '?'):>7}"
        )

    oos = results.get("oos_combined_metrics", {})
    if oos:
        print(
            f"\n  Combined OOS  CAGR={oos.get('cagr','n/a')}%  "
            f"Sharpe={oos.get('sharpe','n/a')}  "
            f"MaxDD={oos.get('max_drawdown','n/a')}%"
        )

    avg = results.get("average_optimal_weights", {})
    if avg:
        from config import WEIGHTS as DW
        print(f"\n  {'Weight Key':<12} {'Default':>10} {'Optimised':>12} {'Change':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
        for k in WEIGHT_KEYS:
            d = DW.get(k, 0)
            o = avg.get(k, 0)
            print(f"  {k:<12} {d:>10.3f} {o:>12.3f} {o - d:>+10.3f}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from historical_data import load_all_fundamentals, load_all_prices
    from backtest import BACKTEST_START, BACKTEST_END

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Clayton Score weight optimizer")
    parser.add_argument("--run",           action="store_true", help="Run with default settings (popsize=8, maxiter=50)")
    parser.add_argument("--optimize-fast", action="store_true", help="Fast approximate run (popsize=5, maxiter=30)")
    parser.add_argument("--optimize-full", action="store_true", help="Thorough run (popsize=12, maxiter=80)")
    parser.add_argument("--show-weights",  action="store_true", help="Print current optimal weights")
    parser.add_argument("--train-years",   type=int, default=7)
    parser.add_argument("--test-years",    type=int, default=1)
    parser.add_argument("--step-years",    type=int, default=1)
    parser.add_argument("--top-n",         type=int, default=25)
    parser.add_argument("--workers",       type=int, default=-1, help="CPU workers (-1=all)")
    # Override popsize/maxiter manually if desired
    parser.add_argument("--popsize",       type=int, default=None)
    parser.add_argument("--maxiter",       type=int, default=None)
    args = parser.parse_args()

    if args.show_weights:
        opt = load_optimal_weights()
        if opt:
            from config import WEIGHTS as DW
            print("\nCurrent optimal weights:")
            for k, v in opt.items():
                print(f"  {k:<12}: {v:.4f}  (default: {DW.get(k, 0):.4f})")
        else:
            print("No optimal weights saved. Run --run first.")

    if args.run or args.optimize_fast or args.optimize_full:
        # Resolve popsize/maxiter from flag hierarchy
        if args.optimize_fast:
            popsize, maxiter = 5, 30
        elif args.optimize_full:
            popsize, maxiter = 12, 80
        else:
            popsize, maxiter = 8, 50
        # Manual overrides take precedence
        if args.popsize is not None:
            popsize = args.popsize
        if args.maxiter is not None:
            maxiter = args.maxiter

        logger.info("Loading historical data into memory...")
        fund_all   = load_all_fundamentals()
        prices     = load_all_prices()
        logger.info(f"Loaded {len(fund_all)} tickers, {len(prices)} price series.")

        results = walk_forward_optimise(
            fundamentals_panel=fund_all,
            prices=prices,
            universe_start=BACKTEST_START,
            universe_end=BACKTEST_END,
            train_years=args.train_years,
            test_years=args.test_years,
            step_years=args.step_years,
            top_n=args.top_n,
            popsize=popsize,
            maxiter=maxiter,
            n_workers=args.workers,
        )

        print_wf_summary(results)

        path = os.path.join(RESULTS_DIR, "walk_forward_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Full WF results saved to {path}")
