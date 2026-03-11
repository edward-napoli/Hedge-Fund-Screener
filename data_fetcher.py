"""
data_fetcher.py — Fetches stock universe and all financial metrics.

Data sources (in priority order):
  1. yfinance  — price ratios, returns, balance sheet, earnings history
  2. yahooquery — forward EPS growth, supplemental fundamentals
  3. Calculated in-house — ROIC, Altman Z-Score, Piotroski F-Score
"""

import io
import logging
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from config import (
    COLUMNS, EXCHANGE_MAP, ALL_INTERNATIONAL,
    MAX_WORKERS, RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_BACKOFF_BASE,
    MIN_MARKET_CAP, MAX_MISSING_METRICS, LOG_FILE,
)
from scorer import count_missing

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Thread-local rate-limit tracker
_thread_local = threading.local()

# ---------------------------------------------------------------------------
# Shared Yahoo Finance session (one crumb/cookie set for all threads)
# ---------------------------------------------------------------------------
_yahoo_session = None
_yahoo_session_lock = threading.Lock()


def _init_yahoo_session():
    """
    Create a curl_cffi Session (required by yfinance 1.x) that impersonates
    Chrome, pre-loaded with Yahoo Finance cookies and crumb.
    Called once before the parallel fetch begins; shared across all threads.
    """
    global _yahoo_session
    with _yahoo_session_lock:
        if _yahoo_session is not None:
            return _yahoo_session

        from curl_cffi.requests import Session as CurlSession
        session = CurlSession(impersonate="chrome124")

        # Step 1: visit Yahoo Finance to receive session cookies
        try:
            session.get("https://finance.yahoo.com", timeout=15)
        except Exception as e:
            logger.warning(f"Yahoo Finance cookie fetch failed: {e}")

        # Step 2: fetch the crumb
        crumb = None
        for crumb_url in [
            "https://query1.finance.yahoo.com/v1/test/csrfToken",
            "https://query2.finance.yahoo.com/v1/test/csrfToken",
        ]:
            try:
                resp = session.get(crumb_url, timeout=10)
                if resp.status_code == 200 and resp.text.strip():
                    crumb = resp.text.strip()
                    break
            except Exception:
                continue

        if crumb:
            session.headers.update({"X-Yahoo-Finance-Crumb": crumb})
            logger.info("Yahoo Finance session initialised (crumb acquired)")
        else:
            logger.warning("Could not acquire Yahoo Finance crumb — proceeding without it")

        _yahoo_session = session
        return _yahoo_session


def _get_yahoo_session():
    """Return the shared Yahoo Finance session, initialising it if needed."""
    global _yahoo_session
    if _yahoo_session is None:
        return _init_yahoo_session()
    return _yahoo_session


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
def with_retry(max_retries: int = MAX_RETRIES, backoff_base: int = RETRY_BACKOFF_BASE):
    """Decorator that retries a function on exception with exponential back-off."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        wait = backoff_base ** attempt
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


def _rate_limit():
    """Enforce per-thread rate limiting."""
    now = time.monotonic()
    last = getattr(_thread_local, "last_request", 0.0)
    elapsed = now - last
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _thread_local.last_request = time.monotonic()


# ---------------------------------------------------------------------------
# Stock universe helpers
# ---------------------------------------------------------------------------
_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _wiki_tables(url: str) -> list:
    """Fetch Wikipedia page with a browser User-Agent and parse HTML tables."""
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = _wiki_tables(url)
        sp500 = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"  S&P 500: {len(sp500)} tickers loaded")
        return sp500
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 from Wikipedia: {e}")
        return []


def _fetch_nasdaq100_tickers() -> list[str]:
    """Fetch NASDAQ-100 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = _wiki_tables(url)
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            if "ticker" in cols or "symbol" in cols:
                col = next(c for c in table.columns if str(c).lower() in ("ticker", "symbol"))
                tickers = table[col].dropna().tolist()
                logger.info(f"  NASDAQ-100: {len(tickers)} tickers loaded")
                return tickers
    except Exception as e:
        logger.warning(f"Could not fetch NASDAQ-100 from Wikipedia: {e}")
    return []


def _fetch_russell1000_tickers() -> list[str]:
    """Fetch Russell 1000 tickers from Wikipedia (best-effort)."""
    try:
        url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
        tables = _wiki_tables(url)
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            if any(k in " ".join(cols) for k in ["ticker", "symbol"]):
                col_name = next(
                    c for c in table.columns
                    if str(c).lower() in ("ticker", "symbol")
                )
                tickers = table[col_name].dropna().tolist()
                if len(tickers) > 50:
                    logger.info(f"  Russell 1000: {len(tickers)} tickers loaded")
                    return tickers
    except Exception as e:
        logger.warning(f"Could not fetch Russell 1000 from Wikipedia: {e}")
    return []


def get_stock_universe() -> list[str]:
    """
    Build a deduplicated global ticker list.

    Sources:
      - S&P 500, NASDAQ-100, Russell 1000 (fetched dynamically)
      - International hardcoded lists from config.py

    Returns
    -------
    list[str]
        Unique ticker symbols.
    """
    logger.info("Loading stock universe...")
    tickers: list[str] = []

    tickers.extend(_fetch_sp500_tickers())
    tickers.extend(_fetch_nasdaq100_tickers())
    tickers.extend(_fetch_russell1000_tickers())
    tickers.extend(ALL_INTERNATIONAL)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tickers:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logger.info(f"[INFO] Loading stock universe... {len(unique):,} stocks found")
    return unique


# ---------------------------------------------------------------------------
# Balance sheet / income statement key helpers
# (yfinance key names vary slightly by version; try multiple variants)
# ---------------------------------------------------------------------------
_BS_KEY_VARIANTS = {
    "total_assets":           ["Total Assets", "TotalAssets"],
    "current_assets":         ["Current Assets", "Total Current Assets", "TotalCurrentAssets"],
    "current_liabilities":    ["Current Liabilities", "Total Current Liabilities", "TotalCurrentLiabilities"],
    "retained_earnings":      ["Retained Earnings", "RetainedEarnings"],
    "total_liabilities":      ["Total Liabilities Net Minority Interest", "Total Liab",
                                "TotalLiabilities", "Total Liabilities"],
    "long_term_debt":         ["Long Term Debt", "LongTermDebt",
                                "Long Term Debt And Capital Lease Obligation"],
    "stockholders_equity":    ["Stockholders Equity", "Total Stockholder Equity",
                                "StockholdersEquity", "Common Stock Equity"],
    "common_stock":           ["Common Stock", "CommonStock", "Ordinary Shares Number"],
    "cash":                   ["Cash And Cash Equivalents", "Cash",
                                "Cash Cash Equivalents And Short Term Investments"],
    "inventory":              ["Inventory", "Inventories"],
}

_FS_KEY_VARIANTS = {
    "total_revenue":    ["Total Revenue", "TotalRevenue", "Revenue"],
    "operating_income": ["Operating Income", "Ebit", "EBIT", "OperatingIncome"],
    "net_income":       ["Net Income", "NetIncome", "Net Income Common Stockholders"],
    "gross_profit":     ["Gross Profit", "GrossProfit"],
    "income_tax":       ["Income Tax Expense", "Tax Provision", "IncomeTaxExpense"],
    "interest_expense": ["Interest Expense", "InterestExpense"],
}

_CF_KEY_VARIANTS = {
    "operating_cashflow": ["Operating Cash Flow", "Total Cash From Operating Activities",
                            "Cash From Operations", "OperatingCashFlow"],
    "capex":              ["Capital Expenditure", "Capital Expenditures", "CapitalExpenditures"],
}


# ---------------------------------------------------------------------------
# yfinance 1.x compatibility helpers
# (yfinance >=1.0 renamed: financials→income_stmt, cashflow→cash_flow)
# ---------------------------------------------------------------------------
def _get_financials(stock: yf.Ticker) -> Optional[pd.DataFrame]:
    """Return annual income statement DataFrame, compatible with yfinance 0.x and 1.x."""
    for attr in ("income_stmt", "financials"):
        try:
            df = getattr(stock, attr, None)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


def _get_cashflow(stock: yf.Ticker) -> Optional[pd.DataFrame]:
    """Return annual cash flow DataFrame, compatible with yfinance 0.x and 1.x."""
    for attr in ("cash_flow", "cashflow"):
        try:
            df = getattr(stock, attr, None)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


def _get_balance_sheet(stock: yf.Ticker) -> Optional[pd.DataFrame]:
    """Return annual balance sheet DataFrame."""
    for attr in ("balance_sheet", "quarterly_balance_sheet"):
        try:
            df = getattr(stock, attr, None)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


def _get_df_val(df: Optional[pd.DataFrame], key_group: str,
                variants_dict: dict, col_idx: int = 0) -> Optional[float]:
    """
    Safely extract a scalar from a yfinance DataFrame using known key variants.
    Returns None if not found or value is NaN/inf.
    """
    if df is None or df.empty:
        return None
    variants = variants_dict.get(key_group, [])
    for key in variants:
        if key in df.index:
            try:
                val = df.loc[key].iloc[col_idx]
                if pd.notna(val) and not math.isinf(float(val)):
                    return float(val)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Advanced metric calculators
# ---------------------------------------------------------------------------
def _calculate_roic(stock: yf.Ticker, info: dict) -> Optional[float]:
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = Operating Income * (1 - effective tax rate)
    Invested Capital = Total Equity + Total Debt - Cash
    """
    try:
        bs = _get_balance_sheet(stock)
        fs = _get_financials(stock)

        op_income = _get_df_val(fs, "operating_income", _FS_KEY_VARIANTS)
        tax_exp   = _get_df_val(fs, "income_tax",       _FS_KEY_VARIANTS)
        revenue   = _get_df_val(fs, "total_revenue",    _FS_KEY_VARIANTS)

        if op_income is None:
            return None

        # Effective tax rate
        tax_rate = 0.21  # default US rate
        if tax_exp is not None and op_income != 0:
            computed = abs(tax_exp) / abs(op_income)
            if 0 < computed < 0.6:
                tax_rate = computed

        nopat = op_income * (1 - tax_rate)

        equity   = _get_df_val(bs, "stockholders_equity", _BS_KEY_VARIANTS)
        lt_debt  = _get_df_val(bs, "long_term_debt",      _BS_KEY_VARIANTS) or 0
        cash     = _get_df_val(bs, "cash",                _BS_KEY_VARIANTS) or 0

        if equity is None:
            return None

        invested_capital = equity + lt_debt - cash
        if invested_capital <= 0:
            return None

        return round((nopat / invested_capital) * 100, 2)
    except Exception:
        return None


def _calculate_altman_z(stock: yf.Ticker, info: dict) -> Optional[float]:
    """
    Altman Z-Score for public companies:
      Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    """
    try:
        bs = _get_balance_sheet(stock)
        fs = _get_financials(stock)

        total_assets = _get_df_val(bs, "total_assets",        _BS_KEY_VARIANTS)
        if not total_assets or total_assets == 0:
            return None

        curr_assets   = _get_df_val(bs, "current_assets",      _BS_KEY_VARIANTS) or 0
        curr_liab     = _get_df_val(bs, "current_liabilities",  _BS_KEY_VARIANTS) or 0
        retained_earn = _get_df_val(bs, "retained_earnings",   _BS_KEY_VARIANTS) or 0
        total_liab    = _get_df_val(bs, "total_liabilities",   _BS_KEY_VARIANTS) or 0
        op_income     = _get_df_val(fs, "operating_income",    _FS_KEY_VARIANTS) or 0
        revenue       = _get_df_val(fs, "total_revenue",       _FS_KEY_VARIANTS) or 0
        market_cap    = float(info.get("marketCap") or 0)

        working_capital = curr_assets - curr_liab

        x1 = working_capital  / total_assets
        x2 = retained_earn    / total_assets
        x3 = op_income        / total_assets
        x4 = market_cap / total_liab if total_liab != 0 else 0
        x5 = revenue          / total_assets

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        return round(z, 2)
    except Exception:
        return None


def _calculate_piotroski(stock: yf.Ticker, info: dict) -> Optional[int]:
    """
    Piotroski F-Score (0–9).
      Profitability (4):  ROA>0, OCF>0, ΔROA>0, Accruals(OCF/A > ROA)
      Leverage (3):       ΔLeverage<0, ΔCurrentRatio>0, No new shares
      Efficiency (2):     ΔGrossMargin>0, ΔAssetTurnover>0
    """
    try:
        bs = _get_balance_sheet(stock)
        fs = _get_financials(stock)
        cf = _get_cashflow(stock)

        if any(x is None or (hasattr(x, "empty") and x.empty) for x in [bs, fs, cf]):
            return None
        if bs.shape[1] < 2 or fs.shape[1] < 2:
            return None  # need two years

        score = 0

        # --- helpers ---
        def bv(key, yr=0):  return _get_df_val(bs, key, _BS_KEY_VARIANTS, yr)
        def fv(key, yr=0):  return _get_df_val(fs, key, _FS_KEY_VARIANTS, yr)
        def cv(key, yr=0):  return _get_df_val(cf, key, _CF_KEY_VARIANTS, yr)

        ta0, ta1  = bv("total_assets", 0), bv("total_assets", 1)
        ni0, ni1  = fv("net_income",   0), fv("net_income",   1)
        ocf0      = cv("operating_cashflow", 0)

        if ta0 and ta0 != 0 and ni0 is not None:
            roa0 = ni0 / ta0
            roa1 = (ni1 / ta1) if (ni1 is not None and ta1 and ta1 != 0) else 0

            if roa0 > 0:         score += 1   # F1
            if ocf0 and ocf0 > 0: score += 1  # F2
            if roa0 > roa1:      score += 1   # F3
            if ocf0 and (ocf0 / ta0) > roa0:  score += 1  # F4

        # Leverage
        ltd0 = bv("long_term_debt", 0) or 0
        ltd1 = bv("long_term_debt", 1) or 0
        ca0  = bv("current_assets",      0)
        ca1  = bv("current_assets",      1)
        cl0  = bv("current_liabilities", 0)
        cl1  = bv("current_liabilities", 1)
        cs0  = bv("common_stock", 0) or 0
        cs1  = bv("common_stock", 1) or 0

        if ta0 and ta0 != 0 and ta1 and ta1 != 0:
            lev0 = ltd0 / ta0
            lev1 = ltd1 / ta1
            if lev0 < lev1:   score += 1  # F5

        if ca0 and cl0 and ca1 and cl1 and cl0 != 0 and cl1 != 0:
            cr0 = ca0 / cl0
            cr1 = ca1 / cl1
            if cr0 > cr1:     score += 1  # F6

        if cs0 <= cs1:        score += 1  # F7 — no new shares

        # Operating efficiency
        rev0 = fv("total_revenue", 0)
        rev1 = fv("total_revenue", 1)
        gp0  = fv("gross_profit",  0)
        gp1  = fv("gross_profit",  1)

        if gp0 and rev0 and rev0 != 0 and gp1 and rev1 and rev1 != 0:
            gm0, gm1 = gp0 / rev0, gp1 / rev1
            if gm0 > gm1:     score += 1  # F8

        if rev0 and ta0 and ta0 != 0 and rev1 and ta1 and ta1 != 0:
            at0, at1 = rev0 / ta0, rev1 / ta1
            if at0 > at1:     score += 1  # F9

        return score
    except Exception:
        return None


def _calculate_eps_cagr(stock: yf.Ticker, years: int, info: Optional[dict] = None) -> Optional[float]:
    """
    Compute trailing EPS CAGR over `years`.

    Strategy (yfinance 1.x compatible):
      1. Try stock.earnings_history (yfinance >=0.2 some builds)
      2. Derive basic EPS from annual income_stmt net income ÷ shares outstanding
      3. Fall back to stock.earnings (yfinance 0.x)
    Returns percentage (e.g. 12.5 for 12.5%).
    """
    try:
        eps_series: Optional[pd.Series] = None

        # --- Attempt 1: earnings_history DataFrame (some yfinance builds) ---
        eh = getattr(stock, "earnings_history", None)
        if eh is not None and isinstance(eh, pd.DataFrame) and not eh.empty:
            for col in ("epsActual", "eps", "EPS"):
                if col in eh.columns:
                    tmp = eh[col].dropna()
                    if len(tmp) >= years + 1:
                        eps_series = tmp.sort_index()
                        break

        # --- Attempt 2: derive from net income / shares (yfinance 1.x) ---
        if eps_series is None:
            fs = _get_financials(stock)
            _info = info or {}
            shares = _info.get("sharesOutstanding") or _info.get("impliedSharesOutstanding")
            if fs is not None and not fs.empty and shares and shares > 0:
                ni_vals = None
                for key in ("Net Income", "NetIncome", "Net Income Common Stockholders"):
                    if key in fs.index:
                        ni_vals = fs.loc[key].dropna()
                        break
                if ni_vals is not None and len(ni_vals) >= years + 1:
                    eps_series = (ni_vals / shares).sort_index(ascending=True)

        # --- Attempt 3: legacy stock.earnings (yfinance 0.x) ---
        if eps_series is None:
            legacy = getattr(stock, "earnings", None)
            if legacy is not None and isinstance(legacy, pd.DataFrame) and not legacy.empty:
                for col in ("Earnings", "EPS"):
                    if col in legacy.columns:
                        tmp = legacy[col].dropna()
                        if len(tmp) >= years + 1:
                            eps_series = tmp.sort_index()
                            break

        if eps_series is None or len(eps_series) < years + 1:
            return None

        eps_end   = float(eps_series.iloc[-1])
        eps_start = float(eps_series.iloc[-(years + 1)])

        if eps_start == 0 or (eps_start < 0 and eps_end < 0):
            return None

        cagr = (abs(eps_end) / abs(eps_start)) ** (1 / years) - 1
        sign = 1 if eps_end >= eps_start else -1
        return round(sign * cagr * 100, 2)
    except Exception:
        return None


def _get_forward_eps_growth(ticker: str) -> Optional[float]:
    """
    Try to get analyst forward EPS growth estimate via yahooquery.
    Returns percentage (e.g. 12.5 for 12.5%).
    """
    try:
        from yahooquery import Ticker as YQTicker
        yqt = YQTicker(ticker)
        trend = yqt.earnings_trend

        if isinstance(trend, str):   # error string
            return None
        if isinstance(trend, dict) and ticker in trend:
            trend_data = trend[ticker]
        else:
            trend_data = trend

        if isinstance(trend_data, pd.DataFrame):
            row = trend_data[trend_data.get("period", pd.Series()) == "+5y"]
            if not row.empty and "growth" in row.columns:
                val = row["growth"].iloc[0]
                if pd.notna(val):
                    return round(float(val) * 100, 2)
        return None
    except Exception:
        return None


def _exchange_from_ticker(ticker: str) -> tuple[str, str]:
    """Infer exchange label and country from ticker suffix."""
    for suffix, (exchange, country) in EXCHANGE_MAP.items():
        if ticker.endswith(suffix):
            return exchange, country
    return "NYSE/NASDAQ", "United States"


# ---------------------------------------------------------------------------
# Main per-stock fetcher
# ---------------------------------------------------------------------------
@with_retry()
def _fetch_info(stock: yf.Ticker) -> dict:
    """Fetch yfinance info dict with retry."""
    _rate_limit()
    return stock.info


def fetch_stock_data(ticker: str) -> Optional[dict]:
    """
    Fetch all financial metrics for a single stock.

    Returns None if:
      - Market cap < MIN_MARKET_CAP
      - More than MAX_MISSING_METRICS formula fields are missing
    Returns a dict keyed by config.COLUMNS on success (N/A for unavailable fields).
    """
    row: dict = {col: None for col in COLUMNS}
    row["Ticker"] = ticker

    try:
        session = _get_yahoo_session()
        stock = yf.Ticker(ticker, session=session)
        info  = _fetch_info(stock)

        if not info or not isinstance(info, dict):
            logger.debug(f"No info for {ticker}")
            return None

        market_cap = info.get("marketCap") or 0
        if market_cap < MIN_MARKET_CAP:
            return None

        # ---- Descriptive fields ----------------------------------------
        row["Company Name"] = (
            info.get("longName") or info.get("shortName") or "N/A"
        )
        exch_inf = info.get("exchange") or ""
        country_inf = info.get("country") or ""
        fallback_exch, fallback_country = _exchange_from_ticker(ticker)
        row["Exchange"] = exch_inf if exch_inf else fallback_exch
        row["Country"]  = country_inf if country_inf else fallback_country
        row["Sector"]   = info.get("sector")   or "N/A"
        row["Industry"] = info.get("industry") or "N/A"

        # ---- Valuation ---------------------------------------------------
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        row["P/E Ratio"] = round(float(pe), 2) if pe and not math.isinf(float(pe)) else None
        row["P/B Ratio"] = round(float(pb), 2) if pb and not math.isinf(float(pb)) else None

        # ---- Net income --------------------------------------------------
        ni = info.get("netIncomeToCommon")
        if ni is not None:
            row["Annual Net Income (USD M)"] = round(float(ni) / 1e6, 2)

        # ---- EPS growth --------------------------------------------------
        eg = info.get("earningsGrowth")
        if eg is not None and not math.isinf(float(eg)):
            row["1-Year EPS Growth %"] = round(float(eg) * 100, 2)

        row["3-Year EPS Growth %"] = _calculate_eps_cagr(stock, 3, info)
        row["5-Year EPS Growth %"] = _calculate_eps_cagr(stock, 5, info)

        # Forward EPS: yfinance → yahooquery fallback
        fwd = (
            info.get("earningsGrowthForecast")
            or info.get("longTermPotentialGrowthRate")
        )
        if fwd is not None:
            val = float(fwd)
            # yfinance sometimes returns as decimal, sometimes as percent
            row["Future EPS Growth Est. %"] = round(val * 100 if abs(val) < 2 else val, 2)
        else:
            row["Future EPS Growth Est. %"] = _get_forward_eps_growth(ticker)

        # ---- Profitability -----------------------------------------------
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")
        if roa is not None and not math.isinf(float(roa)):
            row["ROA %"] = round(float(roa) * 100, 2)
        if roe is not None and not math.isinf(float(roe)):
            row["ROE %"] = round(float(roe) * 100, 2)

        row["ROIC %"] = _calculate_roic(stock, info)

        # ---- Dividends ---------------------------------------------------
        dy = info.get("dividendYield")
        pr = info.get("payoutRatio")
        if dy is not None:
            row["Dividend Yield %"] = round(float(dy) * 100, 2)
        if pr is not None and not math.isinf(float(pr)):
            row["Payout Ratio %"] = round(float(pr) * 100, 2)

        # ---- Liquidity ---------------------------------------------------
        cr = info.get("currentRatio")
        if cr is not None:
            row["Current Ratio"] = round(float(cr), 2)

        # ---- Advanced scores ---------------------------------------------
        row["Altman Z-Score"]    = _calculate_altman_z(stock, info)
        row["Piotroski F-Score"] = _calculate_piotroski(stock, info)

        # ---- Missing-data gate -------------------------------------------
        missing = count_missing(row)
        if missing > MAX_MISSING_METRICS:
            logger.debug(
                f"Skipping {ticker}: {missing} missing metrics (>{MAX_MISSING_METRICS})"
            )
            return None

        # Replace remaining None with "N/A" for display
        for col in COLUMNS:
            if row[col] is None:
                row[col] = "N/A"

        return row

    except Exception as exc:
        logger.error(f"Error fetching {ticker}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Parallel batch fetcher
# ---------------------------------------------------------------------------
def fetch_all_stocks(tickers: list[str]) -> list[dict]:
    """
    Fetch data for all tickers in parallel using ThreadPoolExecutor.

    Returns
    -------
    list[dict]
        List of stock data dicts (only successfully fetched stocks).
    """
    results: list[dict] = []
    skipped = 0

    # Initialise the shared Yahoo Finance session once before spawning threads
    _init_yahoo_session()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_stock_data, t): t for t in tickers
        }
        with tqdm(
            total=len(tickers),
            desc="Fetching financial data",
            unit="stock",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                    else:
                        skipped += 1
                except Exception as exc:
                    logger.error(f"Unhandled error for {ticker}: {exc}")
                    skipped += 1
                finally:
                    pbar.update(1)

    logger.info(f"Fetched {len(results):,} stocks; skipped {skipped:,}")
    return results
