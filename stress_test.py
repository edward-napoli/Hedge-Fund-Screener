"""
stress_test.py — Stress testing and Monte Carlo simulation for the Clayton Score.

Tests:
  1. Historical stress periods (2008-09 GFC, 2020 COVID, 2022 rate shock, etc.)
  2. Remove-one-factor sensitivity (Sharpe vs factor-zeroed portfolios)
  3. Monte Carlo weight perturbation (robustness of current weights)
  4. Transaction cost sensitivity (tc_bps sweep)
  5. Universe size sensitivity (top-N sweep)

CLI usage:
    python stress_test.py --run
    python stress_test.py --run --mc-runs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR = "cache/backtest"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pre-defined stress periods {label: (start, end)}
STRESS_PERIODS = {
    "GFC 2008-09":      (date(2008, 1, 1),  date(2009, 3, 31)),
    "Euro Crisis 2011": (date(2011, 5, 1),  date(2012, 6, 30)),
    "China Vol 2015-16":(date(2015, 6, 1),  date(2016, 2, 29)),
    "COVID Crash 2020": (date(2020, 2, 1),  date(2020, 6, 30)),
    "Rate Shock 2022":  (date(2022, 1, 1),  date(2022, 12, 31)),
    "Drawdown 2018 Q4": (date(2018, 10, 1), date(2019, 1, 31)),
}


# ---------------------------------------------------------------------------
# 1. Historical stress periods
# ---------------------------------------------------------------------------

def run_stress_periods(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
    spy_prices:         pd.Series | None = None,
) -> pd.DataFrame:
    """
    Run backtests confined to each stress period.

    Returns a DataFrame with metrics for strategy vs benchmark per period.
    """
    from backtest import run_backtest

    rows = []
    for label, (start, end) in STRESS_PERIODS.items():
        logger.info(f"Stress period: {label} ({start} to {end})")
        try:
            res = run_backtest(
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
            )
            m  = res.get("metrics", {})
            bm = res.get("benchmark_metrics", {})
            rows.append({
                "Period":           label,
                "Start":            start.isoformat(),
                "End":              end.isoformat(),
                "Strat CAGR%":      m.get("cagr", "n/a"),
                "Strat MaxDD%":     m.get("max_drawdown", "n/a"),
                "Strat Sharpe":     m.get("sharpe", "n/a"),
                "Bench CAGR%":      bm.get("cagr", "n/a"),
                "Bench MaxDD%":     bm.get("max_drawdown", "n/a"),
                "Bench Sharpe":     bm.get("sharpe", "n/a"),
            })
        except Exception as exc:
            logger.warning(f"  Failed: {exc}")
            rows.append({"Period": label, "Start": start.isoformat(), "End": end.isoformat()})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Monte Carlo weight perturbation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    fundamentals_panel: dict,
    prices:             dict,
    base_weights:       dict,
    n_runs:             int   = 300,
    noise_std:          float = 0.3,   # fractional std of each weight
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
    seed:               int   = 42,
) -> dict:
    """
    Perturb each weight by Gaussian noise and measure Sharpe distribution.

    noise_std = 0.3 means ±30% noise on each weight value.

    Returns:
        {
          "sharpe_distribution": list of floats,
          "cagr_distribution":   list of floats,
          "pct_beat_default":    float,
          "mean_sharpe":         float,
          "p5_sharpe":           float,
          "p95_sharpe":          float,
        }
    """
    from backtest import run_backtest, BACKTEST_START, BACKTEST_END
    from weight_optimizer import WEIGHT_KEYS, WEIGHT_BOUNDS

    rng = np.random.default_rng(seed)
    sharpes: list[float] = []
    cagrs:   list[float] = []

    base_vec = np.array([base_weights.get(k, 0.0) for k in WEIGHT_KEYS])

    logger.info(f"Monte Carlo: {n_runs} runs with noise_std={noise_std}...")
    for i in range(n_runs):
        # Perturb weights
        noise    = rng.normal(0, noise_std * np.abs(base_vec))
        perturbed = np.clip(base_vec + noise, 0, None)

        # Clip to bounds
        for j, k in enumerate(WEIGHT_KEYS):
            lo, hi = WEIGHT_BOUNDS[k]
            perturbed[j] = float(np.clip(perturbed[j], lo, hi))

        w = {k: float(v) for k, v in zip(WEIGHT_KEYS, perturbed)}

        try:
            res = run_backtest(
                fundamentals_panel=fundamentals_panel,
                prices=prices,
                weights=w,
                top_n=top_n,
                freq=freq,
                tc_bps=tc_bps,
                risk_free=risk_free,
                start=BACKTEST_START,
                end=BACKTEST_END,
            )
            sharpes.append(res.get("metrics", {}).get("sharpe", 0.0))
            cagrs.append(res.get("metrics", {}).get("cagr", 0.0))
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            logger.info(f"  MC run {i + 1}/{n_runs}  mean Sharpe so far: {np.mean(sharpes):.3f}")

    # Baseline sharpe
    try:
        base_res = run_backtest(
            fundamentals_panel=fundamentals_panel,
            prices=prices,
            weights=base_weights,
            top_n=top_n,
            freq=freq,
            tc_bps=tc_bps,
            risk_free=risk_free,
            start=BACKTEST_START,
            end=BACKTEST_END,
        )
        base_sharpe = base_res.get("metrics", {}).get("sharpe", 0.0)
    except Exception:
        base_sharpe = 0.0

    return {
        "sharpe_distribution": [round(s, 4) for s in sharpes],
        "cagr_distribution":   [round(c, 4) for c in cagrs],
        "base_sharpe":         round(base_sharpe, 3),
        "mean_sharpe":         round(float(np.mean(sharpes)), 3) if sharpes else 0.0,
        "median_sharpe":       round(float(np.median(sharpes)), 3) if sharpes else 0.0,
        "p5_sharpe":           round(float(np.percentile(sharpes, 5)), 3) if sharpes else 0.0,
        "p95_sharpe":          round(float(np.percentile(sharpes, 95)), 3) if sharpes else 0.0,
        "pct_positive_sharpe": round(float(np.mean([s > 0 for s in sharpes]) * 100), 1) if sharpes else 0.0,
        "pct_beat_default":    round(float(np.mean([s >= base_sharpe for s in sharpes]) * 100), 1) if sharpes else 0.0,
        "n_runs":              len(sharpes),
    }


# ---------------------------------------------------------------------------
# 3. Transaction cost sensitivity sweep
# ---------------------------------------------------------------------------

def run_tc_sensitivity(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    tc_bps_range:       list[float] | None = None,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    risk_free:          float = 0.04,
) -> pd.DataFrame:
    """
    Sweep transaction costs from 0 to 50 bps and measure performance impact.

    Returns a DataFrame indexed by tc_bps with performance metrics.
    """
    from backtest import run_backtest, BACKTEST_START, BACKTEST_END

    if tc_bps_range is None:
        tc_bps_range = [0, 5, 10, 15, 20, 30, 50]

    rows = []
    for tc in tc_bps_range:
        logger.info(f"TC sensitivity: {tc} bps")
        try:
            res = run_backtest(
                fundamentals_panel=fundamentals_panel,
                prices=prices,
                weights=weights,
                top_n=top_n,
                freq=freq,
                tc_bps=tc,
                risk_free=risk_free,
                start=BACKTEST_START,
                end=BACKTEST_END,
            )
            m = res.get("metrics", {})
            rows.append({
                "TC (bps)":     tc,
                "CAGR%":        m.get("cagr"),
                "Sharpe":       m.get("sharpe"),
                "Max DD%":      m.get("max_drawdown"),
                "Volatility%":  m.get("volatility"),
            })
        except Exception as exc:
            logger.warning(f"  TC={tc} failed: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Universe size (top-N) sensitivity sweep
# ---------------------------------------------------------------------------

def run_topn_sensitivity(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    topn_range:         list[int] | None = None,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
) -> pd.DataFrame:
    """
    Sweep top-N from 5 to 100 and measure performance.

    Returns a DataFrame indexed by top_n.
    """
    from backtest import run_backtest, BACKTEST_START, BACKTEST_END

    if topn_range is None:
        topn_range = [5, 10, 15, 20, 25, 35, 50, 75, 100]

    rows = []
    for n in topn_range:
        logger.info(f"Top-N sensitivity: N={n}")
        try:
            res = run_backtest(
                fundamentals_panel=fundamentals_panel,
                prices=prices,
                weights=weights,
                top_n=n,
                freq=freq,
                tc_bps=tc_bps,
                risk_free=risk_free,
                start=BACKTEST_START,
                end=BACKTEST_END,
            )
            m = res.get("metrics", {})
            rows.append({
                "Top N":        n,
                "CAGR%":        m.get("cagr"),
                "Sharpe":       m.get("sharpe"),
                "Sortino":      m.get("sortino"),
                "Max DD%":      m.get("max_drawdown"),
            })
        except Exception as exc:
            logger.warning(f"  Top-N={n} failed: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Remove-one-factor sensitivity (focused on performance impact)
# ---------------------------------------------------------------------------

def run_factor_removal_sensitivity(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
) -> pd.DataFrame:
    """
    Similar to LOO in factor_analysis, but measures full metrics (not just Sharpe).
    """
    from backtest import run_backtest, BACKTEST_START, BACKTEST_END

    factor_groups = {
        "EPS Growth (all)":    ["E1", "E3", "E5", "Ef"],
        "EPS 1Y Only":         ["E1"],
        "EPS Future Only":     ["Ef"],
        "ROA":                 ["Ra"],
        "ROE":                 ["Re"],
        "ROIC":                ["Rc"],
        "All Quality":         ["Ra", "Re", "Rc"],
        "Current Ratio":       ["C"],
        "Altman Z-Score":      ["Z"],
        "Piotroski F":         ["F"],
        "Net Income":          ["A"],
        "Dividend/Valuation":  ["Y_outer", "Pe_coef", "Pb_coef"],
    }

    # Baseline
    logger.info("Baseline run...")
    base_res = run_backtest(
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
    base_m = base_res.get("metrics", {})

    rows = [{
        "Factor Removed":  "— Baseline (all factors) —",
        "CAGR%":           base_m.get("cagr"),
        "Sharpe":          base_m.get("sharpe"),
        "Sortino":         base_m.get("sortino"),
        "Max DD%":         base_m.get("max_drawdown"),
        "Sharpe Chg":      0.0,
        "CAGR Chg":        0.0,
    }]

    for label, wkeys in factor_groups.items():
        loo_w = {k: float(v) for k, v in weights.items()}
        for wk in wkeys:
            if wk in loo_w:
                loo_w[wk] = 0.0

        logger.info(f"Removing {label}...")
        try:
            res = run_backtest(
                fundamentals_panel=fundamentals_panel,
                prices=prices,
                weights=loo_w,
                top_n=top_n,
                freq=freq,
                tc_bps=tc_bps,
                risk_free=risk_free,
                start=BACKTEST_START,
                end=BACKTEST_END,
            )
            m = res.get("metrics", {})
            rows.append({
                "Factor Removed":  label,
                "CAGR%":           m.get("cagr"),
                "Sharpe":          m.get("sharpe"),
                "Sortino":         m.get("sortino"),
                "Max DD%":         m.get("max_drawdown"),
                "Sharpe Chg":      round((m.get("sharpe", 0) or 0) - (base_m.get("sharpe", 0) or 0), 3),
                "CAGR Chg":        round((m.get("cagr", 0) or 0) - (base_m.get("cagr", 0) or 0), 2),
            })
        except Exception as exc:
            logger.warning(f"  {label} failed: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save and print
# ---------------------------------------------------------------------------

def save_stress_results(all_results: dict) -> str:
    path = os.path.join(RESULTS_DIR, "stress_test_results.json")
    serialisable = {}
    for k, v in all_results.items():
        if isinstance(v, pd.DataFrame):
            serialisable[k] = v.to_dict(orient="records")
        elif isinstance(v, dict):
            serialisable[k] = v
        else:
            serialisable[k] = str(v)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info(f"Stress test results saved to {path}")
    return path


def print_stress_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  STRESS TEST RESULTS")
    print("=" * 70)

    # Stress periods
    sp = results.get("stress_periods")
    if sp is not None and not sp.empty:
        print("\n  Historical Stress Periods:")
        print(f"  {'Period':<22} {'Strat CAGR':>11} {'Strat DD':>10} {'Bench CAGR':>11} {'Bench DD':>10}")
        print("  " + "-" * 66)
        for _, row in sp.iterrows():
            sc = row.get("Strat CAGR%", "n/a")
            sd = row.get("Strat MaxDD%", "n/a")
            bc = row.get("Bench CAGR%", "n/a")
            bd = row.get("Bench MaxDD%", "n/a")
            sc_s = f"{sc:.1f}%" if isinstance(sc, float) else "n/a"
            sd_s = f"{sd:.1f}%" if isinstance(sd, float) else "n/a"
            bc_s = f"{bc:.1f}%" if isinstance(bc, float) else "n/a"
            bd_s = f"{bd:.1f}%" if isinstance(bd, float) else "n/a"
            print(f"  {row['Period']:<22} {sc_s:>11} {sd_s:>10} {bc_s:>11} {bd_s:>10}")

    # Monte Carlo
    mc = results.get("monte_carlo", {})
    if mc:
        print(f"\n  Monte Carlo ({mc.get('n_runs', 0)} runs, ±30% weight noise):")
        print(f"    Base Sharpe:       {mc.get('base_sharpe', 'n/a')}")
        print(f"    Mean Sharpe:       {mc.get('mean_sharpe', 'n/a')}")
        print(f"    5th-95th Pct:      {mc.get('p5_sharpe', 'n/a')} to {mc.get('p95_sharpe', 'n/a')}")
        print(f"    % Positive Sharpe: {mc.get('pct_positive_sharpe', 'n/a')}%")

    # TC sensitivity
    tc = results.get("tc_sensitivity")
    if tc is not None and not tc.empty:
        print(f"\n  Transaction Cost Sensitivity:")
        print(f"  {'TC (bps)':>10} {'CAGR%':>10} {'Sharpe':>10} {'Max DD%':>10}")
        print("  " + "-" * 44)
        for _, row in tc.iterrows():
            print(
                f"  {row.get('TC (bps)', '?'):>10.0f} "
                f"{row.get('CAGR%', 0):>10.2f} "
                f"{row.get('Sharpe', 0):>10.3f} "
                f"{row.get('Max DD%', 0):>10.2f}"
            )

    # Top-N sensitivity
    tn = results.get("topn_sensitivity")
    if tn is not None and not tn.empty:
        print(f"\n  Top-N Sensitivity:")
        print(f"  {'Top N':>7} {'CAGR%':>10} {'Sharpe':>10} {'Max DD%':>10}")
        print("  " + "-" * 40)
        for _, row in tn.iterrows():
            print(
                f"  {int(row.get('Top N', 0)):>7} "
                f"{row.get('CAGR%', 0):>10.2f} "
                f"{row.get('Sharpe', 0):>10.3f} "
                f"{row.get('Max DD%', 0):>10.2f}"
            )

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_stress_tests(
    fundamentals_panel: dict,
    prices:             dict,
    weights:            dict,
    top_n:              int   = 25,
    freq:               str   = "monthly",
    tc_bps:             float = 10.0,
    risk_free:          float = 0.04,
    mc_runs:            int   = 300,
    spy_prices:         pd.Series | None = None,
) -> dict:
    """Run the full stress test suite and return all results."""
    results: dict = {}

    logger.info("Running stress period tests...")
    results["stress_periods"] = run_stress_periods(
        fundamentals_panel, prices, weights,
        top_n=top_n, freq=freq, tc_bps=tc_bps, risk_free=risk_free,
        spy_prices=spy_prices,
    )

    logger.info("Running factor removal sensitivity...")
    results["factor_sensitivity"] = run_factor_removal_sensitivity(
        fundamentals_panel, prices, weights,
        top_n=top_n, freq=freq, tc_bps=tc_bps, risk_free=risk_free,
    )

    logger.info(f"Running Monte Carlo ({mc_runs} runs)...")
    results["monte_carlo"] = run_monte_carlo(
        fundamentals_panel, prices, weights,
        n_runs=mc_runs, top_n=top_n, freq=freq, tc_bps=tc_bps, risk_free=risk_free,
    )

    logger.info("Running TC sensitivity sweep...")
    results["tc_sensitivity"] = run_tc_sensitivity(
        fundamentals_panel, prices, weights,
        top_n=top_n, freq=freq, risk_free=risk_free,
    )

    logger.info("Running top-N sensitivity sweep...")
    results["topn_sensitivity"] = run_topn_sensitivity(
        fundamentals_panel, prices, weights,
        freq=freq, tc_bps=tc_bps, risk_free=risk_free,
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from historical_data import load_all_fundamentals, load_all_prices, load_prices
    from config import WEIGHTS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Clayton Score stress tests")
    parser.add_argument("--run",      action="store_true")
    parser.add_argument("--mc-runs",  type=int, default=300)
    parser.add_argument("--top-n",    type=int, default=25)
    parser.add_argument("--tc-bps",   type=float, default=10.0)
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
    else:
        fund_all   = load_all_fundamentals()
        fund_panel = {t: d.get("records", []) for t, d in fund_all.items()}
        prices     = load_all_prices()
        spy        = load_prices("SPY")

        results = run_all_stress_tests(
            fundamentals_panel=fund_panel,
            prices=prices,
            weights=WEIGHTS,
            top_n=args.top_n,
            tc_bps=args.tc_bps,
            mc_runs=args.mc_runs,
            spy_prices=spy,
        )

        print_stress_summary(results)
        save_stress_results(results)
