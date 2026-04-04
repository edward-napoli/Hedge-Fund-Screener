"""
scorer.py — Composite score calculation and ranking logic.

Formula:
  Score = 0.3(E1) + 0.8(E3) + 1.4(E5) + 2.5(Ef)
        + Ra + Re + 2(Rc)
        + 1.4(C) + 3.25(Z) + 2.65(F) + 0.2(A)
        + 3.25 * (Y*(2 - Pr/100) - (5*Pe + 3*Pb))

Pe and Pb are cross-sectionally normalized to percentile ranks (0–100)
within the screened universe on each run. 0 = cheapest, 100 = most expensive.
All other metrics remain on their absolute scales.
Missing values default to 0 (or 50 for Pe/Pb after normalization).
"""

import logging
import math
import numpy as np
import pandas as pd
from config import WEIGHTS, FORMULA_METRICS, MAX_MISSING_METRICS

logger = logging.getLogger(__name__)


def _safe(value, default: float = 0.0) -> float:
    """Return float value, substituting default for None / NaN / inf."""
    if value is None:
        return default
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def count_missing(row: dict) -> int:
    """Count how many formula metrics are missing (None, NaN, or 'N/A')."""
    missing = 0
    for col in FORMULA_METRICS:
        val = row.get(col)
        if val is None or val == "N/A":
            missing += 1
        else:
            try:
                if math.isnan(float(val)):
                    missing += 1
            except (TypeError, ValueError):
                missing += 1
    return missing


def normalize_pe_pb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally normalize 'P/E Ratio' and 'P/B Ratio' columns to
    percentile ranks (0–100) across the full screened universe.

    0 = cheapest, 100 = most expensive. Stocks with missing or non-positive
    values receive the neutral default of 50. Modifies df in-place and returns it.

    Also logs 10th / 50th / 90th percentile boundaries of the raw distribution
    so the caller can verify the data looks reasonable.
    """
    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B"}
    for col, label in col_map.items():
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        valid_mask = raw.notna() & (raw > 0) & np.isfinite(raw)
        valid_vals = raw[valid_mask]
        n = len(valid_vals)
        if n < 2:
            df[col] = 50.0
            logger.info(f"[Normalization] {label}: only {n} valid values — all set to 50")
            continue
        p10 = float(valid_vals.quantile(0.10))
        p50 = float(valid_vals.quantile(0.50))
        p90 = float(valid_vals.quantile(0.90))
        logger.info(
            f"[Normalization] {label} raw — "
            f"10th pct: {p10:.2f}  median: {p50:.2f}  90th pct: {p90:.2f}  "
            f"({n} valid stocks)"
        )
        pct_ranks = valid_vals.rank(pct=True) * 100
        result = pd.Series(50.0, index=df.index)
        result[valid_mask] = pct_ranks.round(2)
        df[col] = result
    return df


def normalize_pe_pb_factors(entries: list) -> None:
    """
    Cross-sectionally normalize 'pe_ratio' and 'pb_ratio' in a list of
    (ticker, factor_dict) tuples to percentile ranks (0–100).

    Modifies factor dicts in-place. Missing / non-positive values get 50.
    Used by backtest.py and weight_optimizer.py for point-in-time normalization.
    """
    for key in ("pe_ratio", "pb_ratio"):
        valid_pairs: list[tuple[int, float]] = []
        for i, (_, factors) in enumerate(entries):
            v = factors.get(key)
            try:
                fv = float(v)
                if math.isfinite(fv) and fv > 0:
                    valid_pairs.append((i, fv))
            except (TypeError, ValueError):
                pass

        n = len(valid_pairs)
        if n < 2:
            for _, factors in entries:
                factors[key] = 50.0
            continue

        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
        assigned: dict[int, float] = {}
        for rank_pos, (idx, _) in enumerate(sorted_pairs):
            assigned[idx] = round(rank_pos / (n - 1) * 100, 2)

        for i, (_, factors) in enumerate(entries):
            factors[key] = assigned.get(i, 50.0)


def calculate_composite_score(row: dict) -> float:
    """
    Calculate the composite score for a single stock row.

    Parameters
    ----------
    row : dict
        Dict keyed by column names from config.COLUMNS.

    Returns
    -------
    float
        Composite score (2 decimal places).
    """
    w = WEIGHTS

    E1  = _safe(row.get("1-Year EPS Growth %"))
    E3  = _safe(row.get("3-Year EPS Growth %"))
    E5  = _safe(row.get("5-Year EPS Growth %"))
    Ef  = _safe(row.get("Future EPS Growth Est. %"))
    Ra  = _safe(row.get("ROA %"))
    Re  = _safe(row.get("ROE %"))
    Rc  = _safe(row.get("ROIC %"))
    C   = _safe(row.get("Current Ratio"))
    Z   = _safe(row.get("Altman Z-Score"))
    F   = _safe(row.get("Piotroski F-Score"))
    A   = _safe(row.get("Annual Net Income (USD M)"))
    Y   = _safe(row.get("Dividend Yield %"))
    Pr  = _safe(row.get("Payout Ratio %"))
    Pe  = _safe(row.get("P/E Ratio"))
    Pb  = _safe(row.get("P/B Ratio"))

    growth_term    = w["E1"]*E1 + w["E3"]*E3 + w["E5"]*E5 + w["Ef"]*Ef
    quality_term   = Ra + Re + w["Rc"]*Rc
    liquidity_term = w["C"]*C + w["Z"]*Z + w["F"]*F
    income_term    = w["A"] * A
    div_val_term   = w["Y_outer"] * (Y * (2 - Pr/100) - (w["Pe_coef"]*Pe + w["Pb_coef"]*Pb))

    score = growth_term + quality_term + liquidity_term + income_term + div_val_term
    return round(score, 2)


def score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Composite Score' and 'Rank' columns to a DataFrame of stocks.

    P/E and P/B are cross-sectionally normalized to percentile ranks (0–100)
    before scoring so regional valuation differences do not structurally bias
    the ranking. All other factors remain on their absolute scales.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data without score/rank columns.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by Rank ascending (best stock first).
    """
    df = df.copy()

    # Save raw P/E and P/B for post-rank diagnostic display
    df["_raw_pe"] = pd.to_numeric(df.get("P/E Ratio"), errors="coerce")
    df["_raw_pb"] = pd.to_numeric(df.get("P/B Ratio"), errors="coerce")

    # Cross-sectional percentile normalization for P/E and P/B
    df = normalize_pe_pb(df)

    df["Composite Score"] = df.apply(
        lambda row: calculate_composite_score(row.to_dict()), axis=1
    )
    df["Rank"] = df["Composite Score"].rank(ascending=False, method="min").astype(int)
    df.sort_values("Rank", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Diagnostic: raw vs percentile for top 10 ranked stocks
    logger.info(
        "\n[Diagnostic] Raw vs Percentile-Adjusted P/E and P/B — Top 10 Ranked Stocks:"
    )
    logger.info(
        f"  {'Ticker':<10} {'Rank':>5} {'Raw P/E':>9} {'Pct P/E':>9} "
        f"{'Raw P/B':>9} {'Pct P/B':>9}"
    )
    for _, row in df.head(10).iterrows():
        ticker = str(row.get("Ticker", "?"))
        rank   = row.get("Rank", "?")
        rpe    = row.get("_raw_pe")
        rpb    = row.get("_raw_pb")
        ppe    = row.get("P/E Ratio")
        ppb    = row.get("P/B Ratio")
        rpe_s  = f"{rpe:.1f}" if pd.notna(rpe) else "N/A"
        rpb_s  = f"{rpb:.2f}" if pd.notna(rpb) else "N/A"
        ppe_s  = f"{ppe:.1f}" if isinstance(ppe, float) else str(ppe)
        ppb_s  = f"{ppb:.1f}" if isinstance(ppb, float) else str(ppb)
        logger.info(
            f"  {ticker:<10} {str(rank):>5} {rpe_s:>9} {ppe_s:>9} "
            f"{rpb_s:>9} {ppb_s:>9}"
        )

    df.drop(columns=["_raw_pe", "_raw_pb"], inplace=True)
    return df
