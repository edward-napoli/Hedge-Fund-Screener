"""
factor_analysis.py — Per-factor contribution analysis for the Clayton Score.

Analyses:
  1. Factor-to-forward-return correlation (IC: Information Coefficient)
  2. Leave-one-out contribution: Sharpe drop when each factor is zeroed
  3. Factor decay: IC at 1w, 2w, 1m, 3m, 6m, 12m forward horizons
  4. Factor correlation matrix (to detect redundancy)

CLI usage:
    python factor_analysis.py --run
    python factor_analysis.py --run --horizons 1m 3m 6m 12m
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR  = "cache/backtest"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Factor names and the fundamental record fields they map to
FACTOR_MAP = {
    "EPS 1Y Growth":    "eps_1y_growth",
    "EPS 3Y Growth":    "eps_3y_growth",
    "EPS 5Y Growth":    "eps_5y_growth",
    "ROA":              "roa",
    "ROE":              "roe",
    "ROIC":             "roic",
    "Current Ratio":    "current_ratio",
    "Altman Z-Score":   "altman_z",
    "Piotroski F":      "piotroski_f",
    "Net Income USD M": "net_income_usd_m",
    "P/E Ratio":        "pe_ratio",
    "Div Yield":        "div_yield_pct",
    "Payout Ratio":     "payout_ratio_pct",
}

# Forward-return horizons in trading days
HORIZON_MAP = {
    "1w":  5,
    "2w":  10,
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "12m": 252,
}


# ---------------------------------------------------------------------------
# Helper: compute forward returns
# ---------------------------------------------------------------------------

def _forward_return(
    prices: dict[str, pd.Series],
    ticker: str,
    from_date: date,
    horizon_days: int,
) -> float | None:
    """Return the price return for `ticker` over `horizon_days` trading days from `from_date`."""
    series = prices.get(ticker)
    if series is None:
        return None
    avail = sorted(d for d in series.index if d >= from_date)
    if len(avail) <= horizon_days:
        return None
    p_start = float(series[avail[0]])
    p_end   = float(series[avail[horizon_days]])
    if p_start <= 0:
        return None
    return (p_end - p_start) / p_start


# ---------------------------------------------------------------------------
# Build factor panel (one row per ticker-date observation)
# ---------------------------------------------------------------------------

def build_factor_panel(
    prices:    dict[str, pd.Series],
    sample_dates: list[date],
    horizons:  list[str] = None,
) -> pd.DataFrame:
    """
    Build a panel DataFrame of factor values and forward returns.

    Parameters
    ----------
    prices : dict
        {ticker: pd.Series(date -> price)}
    sample_dates : list[date]
        Dates at which to sample factors (e.g. monthly rebalancing dates).
    horizons : list[str]
        Forward-return horizon labels (e.g. ['1m', '3m', '6m']).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, <factor_fields>, fwd_ret_<horizon>
    """
    from historical_data import get_fundamental_snapshot, compute_eps_growth

    if horizons is None:
        horizons = ["1m", "3m", "6m", "12m"]

    tickers = list(prices.keys())
    rows    = []

    for snap_date in sample_dates:
        logger.debug(f"Building panel: {snap_date}")
        for ticker in tickers:
            snap = get_fundamental_snapshot(ticker, snap_date)
            if not snap:
                continue

            row: dict = {"ticker": ticker, "date": snap_date}

            # Factor values
            row["eps_1y_growth"]    = compute_eps_growth(ticker, snap_date, 1)
            row["eps_3y_growth"]    = compute_eps_growth(ticker, snap_date, 3)
            row["eps_5y_growth"]    = compute_eps_growth(ticker, snap_date, 5)
            row["roa"]              = snap.get("roa")
            row["roe"]              = snap.get("roe")
            row["current_ratio"]    = snap.get("current_ratio")
            row["altman_z"]         = snap.get("altman_z")    # computed by get_fundamental_snapshot()
            row["piotroski_f"]      = snap.get("piotroski_f") # computed by get_fundamental_snapshot()
            ni  = snap.get("net_income")
            row["net_income_usd_m"] = ni / 1_000_000 if ni else None

            # ROIC proxy: net_income / (total_equity + total_debt)
            te = snap.get("total_equity") or 0.0
            td = snap.get("total_debt")   or 0.0
            ic = te + td
            row["roic"] = round(ni / ic * 100, 2) if (ni and ic > 0) else None

            # P/E from price series
            eps = snap.get("eps_basic")
            price_series = prices.get(ticker)
            price_on_date: float | None = None
            if price_series is not None:
                avail = [d for d in price_series.index if d <= snap_date]
                if avail:
                    price_on_date = float(price_series[max(avail)])
            row["pe_ratio"] = round(price_on_date / eps, 2) if (price_on_date and eps and eps > 0) else None

            # Payout ratio and div yield
            div_paid = snap.get("dividends_paid")
            row["payout_ratio_pct"] = round(abs(div_paid) / ni * 100, 2) if (div_paid is not None and ni and ni > 0) else None
            if div_paid is not None and eps and eps > 0 and ni and ni != 0 and price_on_date:
                shares_est = ni / eps
                row["div_yield_pct"] = round(abs(div_paid) / shares_est / price_on_date * 100, 2) if shares_est > 0 else None
            else:
                row["div_yield_pct"] = None

            # Forward returns
            for h in horizons:
                hd = HORIZON_MAP.get(h, 21)
                row[f"fwd_ret_{h}"] = _forward_return(prices, ticker, snap_date, hd)

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Information Coefficient (IC) analysis
# ---------------------------------------------------------------------------

def compute_ic(panel: pd.DataFrame, horizons: list[str] = None) -> pd.DataFrame:
    """
    Compute Rank IC (Spearman correlation) between each factor and each
    forward-return horizon.

    Returns a DataFrame with factors as index, horizons as columns,
    showing IC mean, IC std, t-stat, and p-value.
    """
    if horizons is None:
        horizons = [c.replace("fwd_ret_", "") for c in panel.columns if c.startswith("fwd_ret_")]

    factor_fields = list(FACTOR_MAP.values())
    results = {}

    for factor_name, field in FACTOR_MAP.items():
        if field not in panel.columns:
            continue
        row_data: dict = {}
        for h in horizons:
            col = f"fwd_ret_{h}"
            if col not in panel.columns:
                continue
            sub = panel[[field, col]].dropna()
            if len(sub) < 20:
                continue
            rho, pval = stats.spearmanr(sub[field], sub[col])
            row_data[f"{h}_IC"]     = round(rho, 4)
            row_data[f"{h}_pval"]   = round(pval, 4)
            row_data[f"{h}_tstat"]  = round(rho * np.sqrt(len(sub) - 2) / np.sqrt(1 - rho**2 + 1e-9), 3)
        results[factor_name] = row_data

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# 2. Factor decay analysis
# ---------------------------------------------------------------------------

def compute_factor_decay(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how IC decays over horizons for each factor.

    Returns a DataFrame: factors × horizons with IC values.
    """
    horizons = sorted(HORIZON_MAP.keys(), key=lambda h: HORIZON_MAP[h])
    ic_data  = {}

    for factor_name, field in FACTOR_MAP.items():
        if field not in panel.columns:
            continue
        ics = []
        for h in horizons:
            col = f"fwd_ret_{h}"
            if col not in panel.columns:
                ics.append(None)
                continue
            sub = panel[[field, col]].dropna()
            if len(sub) < 20:
                ics.append(None)
                continue
            rho, _ = stats.spearmanr(sub[field], sub[col])
            ics.append(round(rho, 4))
        ic_data[factor_name] = ics

    return pd.DataFrame(ic_data, index=horizons).T


# ---------------------------------------------------------------------------
# 3. Leave-one-out Sharpe contribution
# ---------------------------------------------------------------------------

def compute_loo_contribution(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
) -> pd.DataFrame:
    """
    For each factor, zero out its weight(s) and compute the resulting
    Sharpe ratio drop vs baseline.

    Returns a DataFrame with columns: factor, baseline_sharpe, loo_sharpe, sharpe_contribution.
    """
    from backtest import run_backtest, BACKTEST_START, BACKTEST_END

    # Baseline
    logger.info("Computing baseline Sharpe for LOO analysis...")
    baseline_res = run_backtest(
        fundamentals_panel=fundamentals_panel,
        prices=prices,
        weights=weights,
        top_n=top_n,
        freq=freq,
        tc_bps=tc_bps,
        risk_free=risk_free,
        start=BACKTEST_START,
        end=BACKTEST_END,
    )
    baseline_sharpe = baseline_res.get("metrics", {}).get("sharpe", 0.0)
    logger.info(f"Baseline Sharpe: {baseline_sharpe:.3f}")

    # Factor to weight keys mapping
    factor_weights = {
        "EPS 1Y Growth":    ["E1"],
        "EPS 3Y Growth":    ["E3"],
        "EPS 5Y Growth":    ["E5"],
        "Future EPS Growth":["Ef"],
        "ROA":              ["Ra"],
        "ROE":              ["Re"],
        "ROIC":             ["Rc"],
        "Current Ratio":    ["C"],
        "Altman Z-Score":   ["Z"],
        "Piotroski F":      ["F"],
        "Net Income":       ["A"],
        "Dividend/Valuation":["Y_outer", "Pe_coef", "Pb_coef"],
    }

    rows = []
    for factor_name, wkeys in factor_weights.items():
        loo_weights = deepcopy_weights(weights)
        for wk in wkeys:
            loo_weights[wk] = 0.0

        logger.info(f"LOO: removing {factor_name}...")
        try:
            res = run_backtest(
                fundamentals_panel=fundamentals_panel,
                prices=prices,
                weights=loo_weights,
                top_n=top_n,
                freq=freq,
                tc_bps=tc_bps,
                risk_free=risk_free,
                start=BACKTEST_START,
                end=BACKTEST_END,
            )
            loo_sharpe = res.get("metrics", {}).get("sharpe", 0.0)
        except Exception as exc:
            logger.warning(f"LOO failed for {factor_name}: {exc}")
            loo_sharpe = 0.0

        rows.append({
            "Factor":               factor_name,
            "Baseline Sharpe":      round(baseline_sharpe, 3),
            "LOO Sharpe":           round(loo_sharpe, 3),
            "Sharpe Contribution":  round(baseline_sharpe - loo_sharpe, 3),
        })

    df = pd.DataFrame(rows).sort_values("Sharpe Contribution", ascending=False)
    return df


def deepcopy_weights(weights: dict) -> dict:
    return {k: float(v) for k, v in weights.items()}


# ---------------------------------------------------------------------------
# 4. Factor correlation matrix
# ---------------------------------------------------------------------------

def compute_factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation matrix between factor values.
    High correlations (>0.7) indicate redundancy.
    """
    factor_fields = [f for f in FACTOR_MAP.values() if f in panel.columns]
    sub = panel[factor_fields].dropna(how="all")
    corr = sub.corr(method="spearman")
    corr.index   = list(FACTOR_MAP.keys())[:len(corr)]
    corr.columns = list(FACTOR_MAP.keys())[:len(corr)]
    return corr


# ---------------------------------------------------------------------------
# Save and print results
# ---------------------------------------------------------------------------

def save_factor_results(results: dict, label: str = "factor_analysis") -> str:
    path = os.path.join(RESULTS_DIR, f"{label}.json")
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            serialisable[k] = v.to_dict(orient="index")
        else:
            serialisable[k] = v
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info(f"Factor analysis saved to {path}")
    return path


def print_factor_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  FACTOR ANALYSIS — INFORMATION COEFFICIENTS")
    print("=" * 70)

    ic_df = results.get("ic_table")
    if ic_df is not None and not ic_df.empty:
        horizons = ["1m", "3m", "6m", "12m"]
        header = f"  {'Factor':<22}" + "".join(f"{'IC '+h:>10}" for h in horizons)
        print(header)
        print("  " + "-" * (22 + 10 * len(horizons)))
        for factor in ic_df.index:
            row = f"  {factor:<22}"
            for h in horizons:
                col = f"{h}_IC"
                v = ic_df.at[factor, col] if col in ic_df.columns else None
                row += f"{v:>10.4f}" if isinstance(v, float) else f"{'n/a':>10}"
            print(row)

    loo_df = results.get("loo_contribution")
    if loo_df is not None and not loo_df.empty:
        print(f"\n  {'LEAVE-ONE-OUT FACTOR CONTRIBUTION':}")
        print(f"  {'Factor':<25} {'Baseline':>10} {'LOO Sharpe':>11} {'Contribution':>13}")
        print("  " + "-" * 62)
        for _, row in loo_df.iterrows():
            print(
                f"  {row['Factor']:<25} "
                f"{row['Baseline Sharpe']:>10.3f} "
                f"{row['LOO Sharpe']:>11.3f} "
                f"{row['Sharpe Contribution']:>13.3f}"
            )

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_factor_analysis(
    prices:             dict,
    fundamentals_panel: dict,
    weights:            dict,
    horizons:           list[str] | None = None,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
) -> dict:
    """
    Run the full factor analysis suite.

    Returns a results dict with: ic_table, decay_table, loo_contribution, factor_corr.
    """
    from backtest import get_rebalance_dates, BACKTEST_START, BACKTEST_END

    if horizons is None:
        horizons = ["1m", "3m", "6m", "12m"]

    # Sample at monthly rebalancing dates
    sample_dates = get_rebalance_dates(BACKTEST_START, BACKTEST_END, "monthly")
    # Limit to dates with sufficient forward data (need 1 year ahead)
    cutoff = date(BACKTEST_END.year - 1, BACKTEST_END.month, BACKTEST_END.day)
    sample_dates = [d for d in sample_dates if d <= cutoff]

    logger.info(f"Building factor panel across {len(sample_dates)} dates × {len(prices)} tickers...")
    panel = build_factor_panel(prices, sample_dates, horizons)
    logger.info(f"Panel shape: {panel.shape}")

    ic_table   = compute_ic(panel, horizons)
    decay_table = compute_factor_decay(panel)
    factor_corr = compute_factor_correlation(panel)

    logger.info("Running LOO Sharpe contribution analysis...")
    loo_df = compute_loo_contribution(
        fundamentals_panel=fundamentals_panel,
        prices=prices,
        weights=weights,
        top_n=top_n,
        freq=freq,
        tc_bps=tc_bps,
        risk_free=risk_free,
    )

    return {
        "ic_table":         ic_table,
        "decay_table":      decay_table,
        "loo_contribution": loo_df,
        "factor_corr":      factor_corr,
        "n_observations":   len(panel),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from historical_data import load_all_fundamentals, load_all_prices
    from config import WEIGHTS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Clayton Score factor analysis")
    parser.add_argument("--run",      action="store_true")
    parser.add_argument("--horizons", nargs="+", default=["1m", "3m", "6m", "12m"],
                        choices=list(HORIZON_MAP.keys()))
    parser.add_argument("--top-n",   type=int, default=25)
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
    else:
        fund_all   = load_all_fundamentals()
        fund_panel = {t: d.get("records", []) for t, d in fund_all.items()}
        prices     = load_all_prices()

        results = run_factor_analysis(
            prices=prices,
            fundamentals_panel=fund_panel,
            weights=WEIGHTS,
            horizons=args.horizons,
            top_n=args.top_n,
        )

        print_factor_summary(results)
        save_factor_results(results)
