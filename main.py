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

# Score history file written after each run
_SCORE_HISTORY_FILE = "cache/score_history.json"

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
# Score history
# ---------------------------------------------------------------------------

def save_score_history(df: pd.DataFrame, date_str: str, run_type: str) -> None:
    """
    Append today's Clayton Scores to cache/score_history.json.

    Priority rule: afternoon entries overwrite morning entries for the same
    date; morning entries do NOT overwrite an existing afternoon entry.

    Parameters
    ----------
    df : pd.DataFrame
        Fully scored and ranked DataFrame (must contain 'Ticker',
        'Composite Score', and 'Rank' columns).
    date_str : str
        Date key in "YYYY-MM-DD" format.
    run_type : str
        "morning" or "afternoon".
    """
    path = _SCORE_HISTORY_FILE
    history: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    # Afternoon takes priority; don't overwrite afternoon with morning
    existing = history.get(date_str, {})
    if existing.get("run_type") == "afternoon" and run_type == "morning":
        logger.info(
            f"Score history for {date_str} already has an afternoon entry — "
            "skipping morning overwrite."
        )
        return

    stocks_data: dict = {}
    for _, row in df.iterrows():
        try:
            stocks_data[str(row["Ticker"])] = {
                "score": float(row["Composite Score"]),
                "rank":  int(row["Rank"]),
            }
        except (TypeError, ValueError):
            pass

    history[date_str] = {
        "run_type":  run_type,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "stocks":    stocks_data,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Score history saved: {len(stocks_data)} stocks for {date_str}.")
    except Exception as exc:
        logger.error(f"Could not save score history: {exc}")


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
# Run validation
# ---------------------------------------------------------------------------

_ERRORS_LOG = "logs/errors.log"


def _validate_run(df: "pd.DataFrame", is_partial: bool) -> list:
    """
    Validate data quality after each screener run.

    Checks
    ------
    1. At least PARTIAL_RUN_THRESHOLD stocks scored (covered by is_partial).
    2. Top-25 rank shifts no more than 15 positions on average vs previous run.
    3. No stock's Composite Score is more than 5 standard deviations from mean.

    Returns a list of warning strings (empty = all clear).
    Failures are also written to logs/errors.log.
    """
    import math

    warnings: list = []

    # Check 1: partial run
    if is_partial:
        msg = (
            f"Partial run: {len(df)} stocks scored "
            f"(threshold: {PARTIAL_RUN_THRESHOLD})"
        )
        warnings.append(msg)

    # Check 2: large top-25 rank shift (>15 positions average)
    if "Rank Delta" in df.columns and not df.empty:
        top25_deltas = df.head(25)["Rank Delta"].dropna()
        if not top25_deltas.empty:
            avg_shift = float(top25_deltas.abs().mean())
            if avg_shift > 15:
                msg = (
                    f"Large top-25 rank shift: avg {avg_shift:.1f} positions "
                    "(threshold: 15) — possible data error"
                )
                warnings.append(msg)

    # Check 3: score outliers (>5 std from mean)
    if "Composite Score" in df.columns and len(df) > 10:
        scores = df["Composite Score"].dropna()
        mean_s = float(scores.mean())
        std_s  = float(scores.std())
        if std_s > 0:
            outliers = df[
                (df["Composite Score"] - mean_s).abs() > 5 * std_s
            ]["Ticker"].tolist()
            if outliers:
                sample = [str(t) for t in outliers[:10]]
                msg = (
                    f"Score outliers (>5 std from mean={mean_s:.1f}, "
                    f"std={std_s:.1f}): {', '.join(sample)}"
                )
                warnings.append(msg)

    # Write failures to errors.log
    if warnings:
        try:
            with open(_ERRORS_LOG, "a", encoding="utf-8") as f:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                for w in warnings:
                    f.write(f"{ts} [VALIDATION] {w}\n")
        except Exception as exc:
            logger.warning(f"Could not write validation failures to errors.log: {exc}")

    return warnings


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
    # 5b. Diagnostic: top-25 regional concentration
    # ------------------------------------------------------------------
    if "Country" in df.columns:
        top25 = df.head(25)
        region_counts = (
            top25["Country"]
            .replace("N/A", "Unknown")
            .value_counts()
        )
        logger.info("\n[Diagnostic] Top-25 stocks by region:")
        for country, count in region_counts.items():
            logger.info(f"  {country}: {count}")

    # ------------------------------------------------------------------
    # 6. Compute deltas vs previous run
    # ------------------------------------------------------------------
    df = compute_deltas(df, previous)

    # ------------------------------------------------------------------
    # 6b. Persist Clayton Scores to score_history.json
    # ------------------------------------------------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        save_score_history(df, today_str, run_type)
    except Exception as exc:
        logger.error(f"save_score_history failed: {exc}")

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
    # 7b. Run validation (outliers, rank shifts, coverage)
    # ------------------------------------------------------------------
    validation_warnings = _validate_run(df, is_partial)
    if validation_warnings:
        for w in validation_warnings:
            logger.warning(f"[VALIDATION] {w}")
        try:
            from alerts import send_slack_validation_alert
            send_slack_validation_alert(validation_warnings)
        except Exception as exc:
            logger.error(f"Could not send validation Slack alert: {exc}")

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
    # 9b. Market regime detection (SPY 200-day MA filter)
    # ------------------------------------------------------------------
    regime_label = "UNKNOWN"
    try:
        from backtest import is_risk_on, fetch_spy_prices
        spy_series = fetch_spy_prices()
        if spy_series is not None:
            risk_on = is_risk_on(spy_series, datetime.now().date())
            regime_label = "RISK-ON" if risk_on else "RISK-OFF"
        else:
            regime_label = "RISK-ON (no SPY data)"
    except Exception as exc:
        logger.warning(f"Regime detection failed: {exc}")
        regime_label = "UNKNOWN"

    weighting_scheme = os.getenv("WEIGHTING_SCHEME", "risk_parity")
    regime_filter_on = os.getenv("REGIME_FILTER", "true").lower() == "true"
    print(
        f"[INFO] Market Regime: {regime_label}  |  "
        f"Weighting: {weighting_scheme}  |  "
        f"Regime filter: {'ON' if regime_filter_on else 'OFF'}"
    )

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
                regime_label=regime_label,
                weighting_scheme=weighting_scheme,
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
            validation_warnings=validation_warnings,
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
        "timestamp":        timestamp,
        "run_type":         run_type,
        "duration_s":       round(elapsed, 1),
        "stocks_fetched":   total_found,
        "stocks_skipped":   total_skipped,
        "errors":           error_count,
        "is_partial":       is_partial,
        "regime":           regime_label,
        "weighting_scheme": weighting_scheme,
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
# Backtesting / analysis CLI handlers
# ---------------------------------------------------------------------------

def _cmd_fetch_history(args) -> None:
    """Bootstrap 10-year historical data."""
    from data_fetcher import get_stock_universe
    from historical_data import bootstrap_historical_data, print_coverage_report
    tickers = get_stock_universe()
    print(f"[INFO] Fetching 10-year history for {len(tickers):,} tickers...")
    bootstrap_historical_data(tickers, force_refresh=args.force)
    print_coverage_report()


def _cmd_coverage_report(_args) -> None:
    from historical_data import print_coverage_report
    print_coverage_report()


def _cmd_diagnose_zf(_args) -> None:
    from historical_data import print_zf_diagnostic
    print_zf_diagnostic()


def _cmd_backtest(args) -> None:
    """Run Clayton Score backtest and optionally write to Sheets."""
    from historical_data import load_all_fundamentals, load_all_prices
    from backtest import (
        run_backtest, run_comparison_backtest,
        print_backtest_summary, print_comparison_table,
        save_backtest_results, fetch_spy_prices,
        BACKTEST_START, BACKTEST_END,
        DEFAULT_WEIGHTING, DEFAULT_REGIME_FILTER,
        DEFAULT_CASH_FRACTION, DEFAULT_VOL_LOOKBACK, DEFAULT_MAX_SINGLE_POS,
    )
    from weight_optimizer import get_active_weights, check_weights_staleness

    # Warn if cached weights are older than the most recent backtest results
    stale_warning = check_weights_staleness()
    if stale_warning:
        print(stale_warning)

    weights = get_active_weights()
    print("[INFO] Loading historical data...")
    fund_all = load_all_fundamentals()
    prices   = load_all_prices()
    spy      = fetch_spy_prices(start=BACKTEST_START, end=BACKTEST_END)

    if not fund_all:
        print("[ERROR] No historical data found. Run: python main.py --fetch-history")
        return

    # Resolve weighting / regime from CLI flags → .env defaults
    weighting = getattr(args, "weighting", DEFAULT_WEIGHTING) or DEFAULT_WEIGHTING
    use_regime = not getattr(args, "no_regime_filter", False)
    if getattr(args, "regime_filter", False):
        use_regime = True

    comparison = getattr(args, "comparison", False)

    common_kwargs = dict(
        fundamentals_panel=fund_all,
        prices=prices,
        weights=weights,
        top_n=args.top_n,
        freq=args.rebalance,
        tc_bps=args.tc_bps,
        risk_free=DEFAULT_RISK_FREE if hasattr(args, "risk_free") else float(os.getenv("RISK_FREE_RATE", "0.04")),
        start=BACKTEST_START,
        end=BACKTEST_END,
        spy_prices=spy,
        cash_fraction=DEFAULT_CASH_FRACTION,
        vol_lookback=DEFAULT_VOL_LOOKBACK,
        max_single_pos=DEFAULT_MAX_SINGLE_POS,
    )

    if comparison:
        print("[INFO] Running all 4 strategy variants for comparison...")
        comp_results = run_comparison_backtest(**common_kwargs)
        print_comparison_table(comp_results)
        _write_backtest_to_sheets(comparison_results=comp_results)
        # Also save each variant individually
        for variant_name, res in comp_results.items():
            label = variant_name.lower().replace(" ", "_").replace(",", "")
            save_backtest_results(res, label=label)
        print("[INFO] All 4 variant results saved to cache/backtest/")
    else:
        print(
            f"[INFO] Running backtest ({args.rebalance}, top-{args.top_n}, "
            f"weighting={weighting}, regime_filter={use_regime})..."
        )
        results = run_backtest(
            **common_kwargs,
            long_short=args.long_short,
            weighting_scheme=weighting,
            regime_filter=use_regime,
        )
        print_backtest_summary(results)
        label = f"{args.rebalance}_top{args.top_n}_{weighting}"
        path  = save_backtest_results(results, label=label)
        print(f"[INFO] Results saved to {path}")
        _write_backtest_to_sheets(backtest_results=results)


def _cmd_optimize(args) -> None:
    """Run walk-forward weight optimisation."""
    import os as _os
    from historical_data import load_all_fundamentals, load_all_prices
    from weight_optimizer import (
        walk_forward_optimise, print_wf_summary,
        OPTIMAL_WEIGHTS_FILE, check_weights_staleness,
    )
    from backtest import BACKTEST_START, BACKTEST_END

    # --force: delete cached weights so optimisation truly starts from scratch
    if getattr(args, "force", False):
        if _os.path.exists(OPTIMAL_WEIGHTS_FILE):
            _os.remove(OPTIMAL_WEIGHTS_FILE)
            print(f"[INFO] --force: deleted cached weights ({OPTIMAL_WEIGHTS_FILE}). "
                  f"Running full optimisation from scratch.")
        else:
            print("[INFO] --force: no cached weights file found, nothing to clear.")

    # Resolve popsize/maxiter from speed flag, then manual overrides
    if getattr(args, "optimize_fast", False):
        popsize, maxiter = 5, 30
    elif getattr(args, "optimize_full", False):
        popsize, maxiter = 12, 80
    else:
        popsize, maxiter = 8, 50
    if args.popsize is not None:
        popsize = args.popsize
    if args.maxiter is not None:
        maxiter = args.maxiter

    print(f"[INFO] Loading historical data...")
    fund_all = load_all_fundamentals()
    prices   = load_all_prices()

    if not fund_all:
        print("[ERROR] No historical data found. Run: python main.py --fetch-history")
        return

    print(
        f"[INFO] Walk-forward optimisation — "
        f"train={args.train_years}yr, test={args.test_years}yr, "
        f"popsize={popsize}, maxiter={maxiter}"
    )
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
    )

    print_wf_summary(results)

    if args.write_sheets:
        _write_backtest_to_sheets(wf_results=results)


def _cmd_factor_analysis(args) -> None:
    """Run factor contribution and IC analysis."""
    from historical_data import load_all_fundamentals, load_all_prices
    from factor_analysis import run_factor_analysis, print_factor_summary, save_factor_results
    from weight_optimizer import get_active_weights

    fund_all   = load_all_fundamentals()
    fund_panel = {t: d.get("records", []) for t, d in fund_all.items()}
    prices     = load_all_prices()

    if not fund_panel:
        print("[ERROR] No historical data found. Run: python main.py --fetch-history")
        return

    print("[INFO] Running factor analysis...")
    results = run_factor_analysis(
        prices=prices,
        fundamentals_panel=fund_panel,
        weights=get_active_weights(),
        horizons=args.horizons,
        top_n=args.top_n,
    )

    print_factor_summary(results)
    save_factor_results(results)

    if args.write_sheets:
        _write_backtest_to_sheets(factor_results=results)


def _cmd_stress_test(args) -> None:
    """Run stress tests and Monte Carlo simulation."""
    from historical_data import load_all_fundamentals, load_all_prices, load_prices
    from stress_test import run_all_stress_tests, print_stress_summary, save_stress_results
    from weight_optimizer import get_active_weights

    fund_all   = load_all_fundamentals()
    fund_panel = {t: d.get("records", []) for t, d in fund_all.items()}
    prices     = load_all_prices()
    spy        = load_prices("SPY")

    if not fund_panel:
        print("[ERROR] No historical data found. Run: python main.py --fetch-history")
        return

    print(f"[INFO] Running stress tests ({args.mc_runs} MC runs)...")
    results = run_all_stress_tests(
        fundamentals_panel=fund_panel,
        prices=prices,
        weights=get_active_weights(),
        top_n=args.top_n,
        tc_bps=args.tc_bps,
        mc_runs=args.mc_runs,
        spy_prices=spy,
    )

    print_stress_summary(results)
    save_stress_results(results)

    if args.write_sheets:
        _write_backtest_to_sheets(stress_results=results)


def _write_backtest_to_sheets(
    backtest_results=None,
    wf_results=None,
    stress_results=None,
    factor_results=None,
    comparison_results=None,
) -> None:
    """Write whichever results are available to the Backtest Results tab."""
    from sheets_writer import write_backtest_tab
    import gspread
    from google.oauth2.service_account import Credentials

    sheet_id      = os.getenv("GOOGLE_SHEET_ID", "").strip() or None
    credentials_f = os.getenv("CREDENTIALS_FILE", "credentials.json")

    if not os.path.exists(credentials_f):
        logger.warning("No credentials file found at '%s' — skipping Backtest Results tab write.", credentials_f)
        return

    if not sheet_id:
        logger.warning("GOOGLE_SHEET_ID not set — skipping Backtest Results tab write.")
        return

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(credentials_f, scopes=scopes)
        gc    = gspread.authorize(creds)
        ss    = gc.open_by_key(sheet_id)
        write_backtest_tab(
            ss,
            backtest_results=backtest_results,
            wf_results=wf_results,
            stress_results=stress_results,
            factor_results=factor_results,
            comparison_results=comparison_results,
        )
        logger.info("Backtest Results tab successfully written: %s", ss.url)
        print(f"[INFO] Backtest Results tab updated: {ss.url}")
    except Exception as exc:
        logger.error("Backtest Results tab write failed: %s", exc, exc_info=True)
        print(f"[ERROR] Backtest Results tab write failed: {exc}")


# ---------------------------------------------------------------------------
# Live status and scheduler status CLI commands
# ---------------------------------------------------------------------------

def _cmd_live_status(_args) -> None:
    """Show live data accumulation progress from run_history.json."""
    history: list = []
    if os.path.exists(RUN_HISTORY_FILE):
        try:
            with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as exc:
            print(f"[ERROR] Could not read {RUN_HISTORY_FILE}: {exc}")
            return

    successful = [
        r for r in history
        if r.get("status") != "FAILED" and r.get("stocks_fetched")
    ]

    if not successful:
        print("\n[INFO] No successful live runs recorded yet.")
        print(f"       Start the scheduler with start_scheduler.bat, or run:\n"
              f"       python main.py\n")
        return

    # Unique trading days (by date prefix of timestamp)
    trading_days = sorted({r["timestamp"][:10] for r in successful if r.get("timestamp")})
    n_days       = len(trading_days)
    first_run    = trading_days[0]
    last_run     = trading_days[-1]

    # Most recent run metadata (latest timestamp)
    last_meta   = max(successful, key=lambda r: r.get("timestamp", ""))
    n_stocks    = last_meta.get("stocks_fetched", 0)
    top5        = last_meta.get("top10", [])[:5]
    regime      = last_meta.get("regime", "unknown")

    days_to_21  = max(0, 21 - n_days)
    days_to_63  = max(0, 63 - n_days)

    print("\n" + "=" * 58)
    print("  LIVE DATA ACCUMULATION STATUS")
    print("=" * 58)
    print(f"  Live trading days recorded:  {n_days}")
    print(f"  Stocks tracked (last run):   {n_stocks:,}")
    print(f"  First live run:              {first_run}")
    print(f"  Most recent run:             {last_run}")
    print(f"  Market regime (last run):    {regime}")
    print()
    if days_to_21 > 0:
        print(f"  Days to 21-day threshold:    {days_to_21}  (efficacy analysis unlocks)")
    else:
        print(f"  Efficacy analysis:           UNLOCKED (>= 21 days)")
    if days_to_63 > 0:
        print(f"  Days to 63-day threshold:    {days_to_63}  (backtest validation unlocks)")
    else:
        print(f"  Backtest validation:         UNLOCKED (>= 63 days)")
    print()
    print("  Top 5 by Clayton Score (most recent run):")
    for i, stock in enumerate(top5, 1):
        print(f"    {i}. {str(stock.get('ticker','?')):<8}  Score: {stock.get('score', 0):,.2f}")
    print("=" * 58 + "\n")


def _cmd_scheduler_status(_args) -> None:
    """Show scheduler process status and last 10 log lines."""
    import subprocess

    log_file = os.path.join("logs", "scheduler.log")
    pid_file = os.path.join("logs", "scheduler.pid")

    # ── Check if process is running via PID file ─────────────────────
    proc_running = False
    running_pid  = None
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r", encoding="utf-8") as f:
                pid_str = f.read().strip()
            if pid_str.isdigit():
                running_pid = int(pid_str)
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {running_pid}", "/FO", "CSV"],
                    capture_output=True, text=True, timeout=5,
                )
                proc_running = str(running_pid) in result.stdout
        except Exception:
            pass

    status_label = f"RUNNING (PID {running_pid})" if proc_running else "NOT RUNNING"
    print(f"\n[INFO] Scheduler process: {status_label}")

    # ── Last 10 log lines ────────────────────────────────────────────
    print(f"\n[INFO] Last 10 lines of {log_file}:")
    print("-" * 65)
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
        except Exception as exc:
            print(f"[ERROR] Could not read log file: {exc}")
    else:
        print("  (log file not found — scheduler has not been started yet)")
    print("-" * 65)

    # ── Today's scheduled times ──────────────────────────────────────
    try:
        from scheduler import _get_schedule_times, _should_run
        from datetime import date as _date
        today = _date.today()
        m, a, p = _get_schedule_times(today)
        runs_today = _should_run(today)
        market_status = "MARKET OPEN" if runs_today else "MARKET CLOSED / HOLIDAY"
        print(f"\n[INFO] Today's schedule ({today}) - {market_status}:")
        print(f"  Morning run:    {m.strftime('%H:%M UTC')}")
        print(f"  Afternoon run:  {a.strftime('%H:%M UTC')}")
        print(f"  Price fetch:    {p.strftime('%H:%M UTC')}")
    except Exception as exc:
        print(f"[WARN] Could not compute next run times: {exc}")

    if not proc_running:
        print("\n  To start: start_scheduler.bat  (or: python scheduler.py)")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hedge Fund Global Stock Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                              # morning screener run\n"
            "  python main.py --run-type afternoon         # afternoon run\n"
            "  python main.py --fetch-history              # bootstrap 10yr data\n"
            "  python main.py --coverage-report            # data coverage stats\n"
            "  python main.py --backtest                   # run backtest\n"
            "  python main.py --backtest --write-sheets    # backtest + Sheets tab\n"
            "  python main.py --optimize-fast              # quick weight opt (~mins)\n"
            "  python main.py --optimize                   # standard weight opt\n"
            "  python main.py --optimize-full              # thorough weight opt (~hrs)\n"
            "  python main.py --factor-analysis            # factor IC + LOO analysis\n"
            "  python main.py --stress-test                # stress tests + Monte Carlo\n"
        ),
    )

    # Screener mode
    parser.add_argument(
        "--run-type",
        choices=["morning", "afternoon"],
        default="morning",
        help="Screener run type (default: morning)",
    )

    # Historical data
    parser.add_argument("--fetch-history",    action="store_true", help="Bootstrap 10-year historical data")
    parser.add_argument("--coverage-report",  action="store_true", help="Print data coverage stats")
    parser.add_argument("--diagnose-zf",      action="store_true", help="Print Altman Z / Piotroski F diagnostic")
    parser.add_argument("--force",            action="store_true", help="Force re-fetch / ignore cached weights (--fetch-history, --optimize*)")

    # Live status / scheduler status
    parser.add_argument("--live-status",      action="store_true", help="Show live data accumulation progress")
    parser.add_argument("--scheduler-status", action="store_true", help="Show scheduler process status and next run times")

    # Backtesting
    parser.add_argument("--backtest",       action="store_true", help="Run Clayton Score backtest")
    parser.add_argument("--comparison",     action="store_true", help="Run all 4 strategy variants side-by-side")
    parser.add_argument("--rebalance",      default="monthly",   choices=["monthly", "weekly", "daily"])
    parser.add_argument("--top-n",          type=int, default=int(os.getenv("BACKTEST_TOP_N", "25")))
    parser.add_argument("--tc-bps",         type=float, default=float(os.getenv("TRANSACTION_COST_BPS", "10")))
    parser.add_argument("--long-short",     action="store_true", help="Long-short variant")
    parser.add_argument("--write-sheets",   action="store_true", help="Write results to Backtest Results tab")
    # Weighting and regime
    parser.add_argument("--weighting",      default=None,        choices=["equal", "risk_parity"],
                        help="Position sizing: equal or risk_parity (default: from WEIGHTING_SCHEME env)")
    _regime_grp = parser.add_mutually_exclusive_group()
    _regime_grp.add_argument("--regime-filter",    dest="regime_filter",    action="store_true",
                             help="Enable SPY 200-day MA regime filter")
    _regime_grp.add_argument("--no-regime-filter", dest="no_regime_filter", action="store_true",
                             help="Disable SPY 200-day MA regime filter")

    # Optimisation — three speed tiers
    parser.add_argument("--optimize",        action="store_true", help="Walk-forward optimisation (popsize=8, maxiter=50)")
    parser.add_argument("--optimize-fast",   action="store_true", help="Fast approximate optimisation (popsize=5, maxiter=30)")
    parser.add_argument("--optimize-full",   action="store_true", help="Thorough optimisation (popsize=12, maxiter=80)")
    parser.add_argument("--train-years",     type=int, default=7)
    parser.add_argument("--test-years",      type=int, default=1)
    parser.add_argument("--step-years",      type=int, default=1)
    # Optional manual overrides — default None so speed-tier defaults apply
    parser.add_argument("--popsize",         type=int, default=None, help="Override popsize (optional)")
    parser.add_argument("--maxiter",         type=int, default=None, help="Override maxiter (optional)")

    # Factor analysis
    parser.add_argument("--factor-analysis", action="store_true", help="Factor IC + LOO analysis")
    parser.add_argument("--horizons",        nargs="+", default=["1m", "3m", "6m", "12m"])

    # Stress tests
    parser.add_argument("--stress-test",    action="store_true", help="Stress tests + Monte Carlo")
    parser.add_argument("--mc-runs",        type=int, default=300)

    args = parser.parse_args()

    # Route to correct handler
    if args.live_status:
        _cmd_live_status(args)
    elif args.scheduler_status:
        _cmd_scheduler_status(args)
    elif args.fetch_history:
        _cmd_fetch_history(args)
    elif args.coverage_report:
        _cmd_coverage_report(args)
    elif getattr(args, "diagnose_zf", False):
        _cmd_diagnose_zf(args)
    elif args.backtest:
        _cmd_backtest(args)
    elif args.optimize or args.optimize_fast or args.optimize_full:
        _cmd_optimize(args)
    elif args.factor_analysis:
        _cmd_factor_analysis(args)
    elif args.stress_test:
        _cmd_stress_test(args)
    else:
        run_screener(run_type=args.run_type)
