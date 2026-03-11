"""
scorer.py — Composite score calculation and ranking logic.

Formula:
  Score = 0.3(E1) + 0.8(E3) + 1.4(E5) + 2.5(Ef)
        + Ra + Re + 2(Rc)
        + 1.4(C) + 3.25(Z) + 2.65(F) + 0.2(A)
        + 3.25 * (Y*(2 - Pr) - (5*Pe + 3*Pb))

All percentage values kept as raw numbers (15% = 15).
Missing values are treated as 0 so the formula never crashes.
"""

import math
import pandas as pd
from config import WEIGHTS, FORMULA_METRICS, MAX_MISSING_METRICS


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
    df["Composite Score"] = df.apply(
        lambda row: calculate_composite_score(row.to_dict()), axis=1
    )
    df["Rank"] = df["Composite Score"].rank(ascending=False, method="min").astype(int)
    df.sort_values("Rank", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
