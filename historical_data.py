"""
historical_data.py — 10-year historical fundamental and price data bootstrap.

Data sourcing strategy (2-layer):
  Layer 1 — SEC EDGAR API  (US stocks, free, point-in-time quarterly filings)
  Layer 2 — yfinance       (international tickers + US tickers not in EDGAR)

FMP (Financial Modeling Prep) was removed from the pipeline. The free tier
no longer supports historical financial-statement endpoints for any symbol
(v3 endpoints blocked as "legacy" after Aug 2025; stable endpoints return
HTTP 402 for non-US symbols and hang/timeout for US symbols on free plans).

All data is cached to disk under cache/historical/ to avoid re-fetching.
Prices are stored as Parquet files; fundamentals as JSON.

CLI usage:
    python historical_data.py --fetch         # bootstrap / refresh all data
    python historical_data.py --coverage      # print coverage report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
CACHE_DIR          = "cache/historical"
PRICES_DIR         = os.path.join(CACHE_DIR, "prices")
FUNDAMENTALS_DIR   = os.path.join(CACHE_DIR, "fundamentals")
COVERAGE_FILE      = os.path.join(CACHE_DIR, "coverage.json")
CIK_MAP_FILE       = os.path.join(CACHE_DIR, "cik_map.json")

LOOKBACK_YEARS     = 10
MIN_DATE           = date.today().replace(year=date.today().year - LOOKBACK_YEARS)
MAX_WORKERS        = 8
REQUEST_DELAY      = 0.12   # seconds between EDGAR requests (rate-limit friendly)

EDGAR_BASE         = "https://data.sec.gov"

EDGAR_HEADERS      = {
    "User-Agent": "HedgeFundScreener research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

os.makedirs(PRICES_DIR, exist_ok=True)
os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fundamentals schema:
#   {
#     "ticker": "AAPL",
#     "source":  "edgar" | "yfinance",  # "fmp" may appear in legacy cached files
#     "records": [
#       {
#         "period_end":  "2023-09-30",   # YYYY-MM-DD
#         "filed_date":  "2023-11-02",   # point-in-time filing date (EDGAR only)
#         "period_type": "annual"|"quarterly",
#         "revenue":          float | null,
#         "net_income":       float | null,
#         "total_assets":     float | null,
#         "total_equity":     float | null,
#         "total_debt":       float | null,
#         "current_assets":   float | null,
#         "current_liab":     float | null,
#         "retained_earnings":float | null,
#         "ebit":             float | null,
#         "eps_basic":        float | null,
#         "dividends_paid":   float | null,
#         "operating_cf":     float | null,
#         "roa":              float | null,  # net_income / total_assets
#         "roe":              float | null,  # net_income / total_equity
#         "current_ratio":    float | null,
#       }, ...
#     ]
#   }
# ---------------------------------------------------------------------------


# ===========================================================================
# CIK map (ticker → SEC CIK for EDGAR lookups)
# ===========================================================================

def _load_cik_map() -> dict[str, str]:
    """Load or fetch the SEC ticker → CIK mapping."""
    if os.path.exists(CIK_MAP_FILE):
        with open(CIK_MAP_FILE, "r") as f:
            return json.load(f)
    return _refresh_cik_map()


def _refresh_cik_map() -> dict[str, str]:
    """Download company_tickers.json from SEC and cache it."""
    logger.info("Downloading SEC CIK map...")
    # Primary URL moved to www.sec.gov; data.sec.gov kept as fallback
    candidate_urls = [
        "https://www.sec.gov/files/company_tickers.json",
        f"{EDGAR_BASE}/files/company_tickers.json",
        "https://www.sec.gov/files/company_tickers_exchange.json",
    ]
    raw = None
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            r.raise_for_status()
            raw = r.json()
            logger.info(f"CIK map fetched from {url}")
            break
        except Exception as exc:
            logger.warning(f"CIK map attempt failed ({url}): {exc}")

    if raw is None:
        logger.error("CIK map download failed: all candidate URLs exhausted.")
        return {}

    cik_map: dict[str, str] = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper()
        cik    = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            cik_map[ticker] = cik
    with open(CIK_MAP_FILE, "w") as f:
        json.dump(cik_map, f)
    logger.info(f"CIK map: {len(cik_map):,} companies cached.")
    return cik_map


# ===========================================================================
# Layer 1 — SEC EDGAR
# ===========================================================================

_EDGAR_CONCEPT_MAP = {
    # income statement
    "revenue":          ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                         "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "net_income":       ["NetIncomeLoss", "NetIncome", "ProfitLoss"],
    "ebit":             ["OperatingIncomeLoss"],
    "eps_basic":        ["EarningsPerShareBasic"],
    # balance sheet
    "total_assets":     ["Assets"],
    "total_equity":     ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_debt":       ["LongTermDebt", "LongTermDebtCurrent", "DebtCurrent"],
    "current_assets":   ["AssetsCurrent"],
    "current_liab":     ["LiabilitiesCurrent"],
    "retained_earnings":["RetainedEarningsAccumulatedDeficit"],
    # cash flow
    "operating_cf":     ["NetCashProvidedByUsedInOperatingActivities"],
    "dividends_paid":   ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}


def _edgar_fetch_concept(cik: str, concept: str, session: requests.Session) -> list[dict]:
    """Fetch one XBRL concept from EDGAR for the given CIK. Returns list of facts."""
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = session.get(url, headers=EDGAR_HEADERS, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data  = r.json()
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        for tag in _EDGAR_CONCEPT_MAP.get(concept, []):
            if tag in us_gaap:
                units = us_gaap[tag].get("units", {})
                # prefer USD for monetary, shares/USD for EPS
                for unit_key in ("USD", "USD/shares", "shares"):
                    if unit_key in units:
                        return us_gaap[tag]["units"][unit_key]
        return []
    except Exception:
        return []


def _edgar_build_records(cik: str, session: requests.Session) -> list[dict]:
    """
    Fetch all EDGAR concepts for one company and pivot into period records.

    Returns list of period-level dicts, each representing one filing.
    """
    # Fetch company facts once (single HTTP call)
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = session.get(url, headers=EDGAR_HEADERS, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.debug(f"EDGAR fetch failed for CIK {cik}: {exc}")
        return []

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    # Build {concept: {period_end: {value, filed}}}
    concept_data: dict[str, dict[str, dict]] = {}
    for concept, tags in _EDGAR_CONCEPT_MAP.items():
        for tag in tags:
            if tag not in us_gaap:
                continue
            units = us_gaap[tag].get("units", {})
            facts = units.get("USD") or units.get("USD/shares") or units.get("shares") or []
            for fact in facts:
                form = fact.get("form", "")
                # Only annual (10-K) and quarterly (10-Q) filings
                if form not in ("10-K", "10-Q"):
                    continue
                end   = fact.get("end", "")
                filed = fact.get("filed", "")
                val   = fact.get("val")
                if not end or val is None:
                    continue
                try:
                    end_date = date.fromisoformat(end)
                except ValueError:
                    continue
                if end_date < MIN_DATE:
                    continue
                if concept not in concept_data:
                    concept_data[concept] = {}
                # Keep most recent filing for each period end
                existing = concept_data[concept].get(end, {})
                if not existing or filed > existing.get("filed", ""):
                    concept_data[concept][end] = {"value": val, "filed": filed, "form": form}
            break  # found the tag we need

    # Collect all period ends
    all_periods: set[str] = set()
    for cd in concept_data.values():
        all_periods.update(cd.keys())

    records = []
    for period_end in sorted(all_periods):
        rec: dict[str, Any] = {
            "period_end":  period_end,
            "filed_date":  None,
            "period_type": "unknown",
        }
        max_filed = ""
        for concept, periods in concept_data.items():
            if period_end in periods:
                entry = periods[period_end]
                rec[concept] = entry["value"]
                if entry["filed"] > max_filed:
                    max_filed = entry["filed"]
                    rec["filed_date"]  = entry["filed"]
                    rec["period_type"] = "annual" if entry["form"] == "10-K" else "quarterly"
            else:
                rec[concept] = None

        # Derived ratios
        ni  = rec.get("net_income")
        ta  = rec.get("total_assets")
        te  = rec.get("total_equity")
        ca  = rec.get("current_assets")
        cl  = rec.get("current_liab")
        rec["roa"]           = round(ni / ta * 100, 2) if ni and ta else None
        rec["roe"]           = round(ni / te * 100, 2) if ni and te else None
        rec["current_ratio"] = round(ca / cl, 4)       if ca and cl and cl != 0 else None
        records.append(rec)

    return records


def fetch_edgar(ticker: str, cik: str, session: requests.Session) -> dict | None:
    """Fetch EDGAR fundamentals for one US ticker. Returns fundamentals dict or None."""
    if not cik:
        return None
    time.sleep(REQUEST_DELAY)
    records = _edgar_build_records(cik, session)
    if not records:
        return None
    return {"ticker": ticker, "source": "edgar", "records": records}


# ===========================================================================
# Layer 2 — yfinance (international + US tickers not covered by EDGAR)
# ===========================================================================

def fetch_yfinance_fundamentals(ticker: str) -> dict | None:
    """
    Pull annual financials from yfinance as last-resort fallback.
    Returns a lightweight record set (no point-in-time filing dates).
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        inc = tk.financials          # columns = dates, rows = line items
        bal = tk.balance_sheet
        cf  = tk.cashflow

        if inc is None or inc.empty:
            return None

        records = []
        for col in inc.columns:
            try:
                end_date = col.date() if hasattr(col, "date") else date.fromisoformat(str(col)[:10])
            except Exception:
                continue
            if end_date < MIN_DATE:
                continue

            def _row(df: pd.DataFrame, *labels) -> float | None:
                if df is None or col not in df.columns:
                    return None
                for lbl in labels:
                    if lbl in df.index:
                        v = df.at[lbl, col]
                        try:
                            f = float(v)
                            return f if not pd.isna(f) else None
                        except Exception:
                            pass
                return None

            ni = _row(inc, "Net Income", "Net Income Common Stockholders")
            ta = _row(bal, "Total Assets")
            te = _row(bal, "Stockholders Equity", "Total Stockholders Equity")
            ca = _row(bal, "Current Assets", "Total Current Assets")
            cl = _row(bal, "Current Liabilities", "Total Current Liabilities")

            rec: dict[str, Any] = {
                "period_end":         end_date.isoformat(),
                "filed_date":         None,
                "period_type":        "annual",
                "revenue":            _row(inc, "Total Revenue"),
                "net_income":         ni,
                "ebit":               _row(inc, "EBIT", "Operating Income"),
                "eps_basic":          _row(inc, "Basic EPS"),
                "total_assets":       ta,
                "total_equity":       te,
                "total_debt":         _row(bal, "Total Debt", "Long Term Debt"),
                "current_assets":     ca,
                "current_liab":       cl,
                "retained_earnings":  _row(bal, "Retained Earnings"),
                "operating_cf":       _row(cf, "Operating Cash Flow"),
                "dividends_paid":     _row(cf, "Cash Dividends Paid"),
                "roa":                round(ni / ta * 100, 2) if ni and ta else None,
                "roe":                round(ni / te * 100, 2) if ni and te and te != 0 else None,
                "current_ratio":      round(ca / cl, 4)       if ca and cl and cl != 0 else None,
            }
            records.append(rec)

        if not records:
            return None
        return {"ticker": ticker, "source": "yfinance", "records": records}
    except Exception as exc:
        logger.debug(f"yfinance fundamentals failed for {ticker}: {exc}")
        return None


# ===========================================================================
# Price data
# ===========================================================================

def fetch_price_history(tickers: list[str]) -> dict[str, pd.Series]:
    """
    Download 10+ years of adjusted closing prices for all tickers.
    Uses yf.download() in batches; returns {ticker: pd.Series(date->price)}.
    """
    import yfinance as yf

    start = MIN_DATE.isoformat()
    results: dict[str, pd.Series] = {}
    batch_size = 100

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(f"Downloading prices {i + 1}–{min(i + batch_size, len(tickers))} / {len(tickers)}...")
        try:
            df = yf.download(
                batch,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df.empty:
                continue
            # Multi-ticker: Close is a DataFrame; single-ticker: Series
            close = df["Close"] if "Close" in df else df
            if isinstance(close, pd.Series):
                # single ticker returned
                ticker = batch[0]
                s = close.dropna()
                if not s.empty:
                    s.index = pd.to_datetime(s.index).date
                    results[ticker] = s
            else:
                for ticker in close.columns:
                    s = close[ticker].dropna()
                    if not s.empty:
                        s.index = pd.to_datetime(s.index).date
                        results[ticker] = s
        except Exception as exc:
            logger.warning(f"Batch price download failed: {exc}")
        time.sleep(1)

    return results


def save_prices(prices: dict[str, pd.Series]) -> None:
    """Persist price series to parquet files (one per ticker)."""
    for ticker, series in prices.items():
        path = os.path.join(PRICES_DIR, f"{ticker.replace('/', '_')}.parquet")
        df   = series.reset_index()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(path, index=False)


def load_prices(ticker: str) -> pd.Series | None:
    """Load cached price series for one ticker. Returns pd.Series(date->close) or None."""
    path = os.path.join(PRICES_DIR, f"{ticker.replace('/', '_')}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        s  = df.set_index("date")["close"]
        s.index = pd.to_datetime(s.index).date
        return s
    except Exception:
        return None


def load_all_prices() -> dict[str, pd.Series]:
    """Load all cached price series."""
    result = {}
    for fname in os.listdir(PRICES_DIR):
        if not fname.endswith(".parquet"):
            continue
        ticker = fname[:-8].replace("_", "/")  # crude reverse of save
        s = load_prices(ticker)
        if s is not None:
            result[ticker] = s
    return result


# ===========================================================================
# Fundamental caching helpers
# ===========================================================================

def _fund_path(ticker: str) -> str:
    return os.path.join(FUNDAMENTALS_DIR, f"{ticker.replace('/', '_')}.json")


def save_fundamentals(data: dict) -> None:
    path = _fund_path(data["ticker"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_fundamentals(ticker: str) -> dict | None:
    path = _fund_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_all_fundamentals() -> dict[str, dict]:
    """Load all cached fundamental dicts keyed by ticker."""
    result = {}
    for fname in os.listdir(FUNDAMENTALS_DIR):
        if not fname.endswith(".json"):
            continue
        ticker = fname[:-5].replace("_", "/")
        data   = load_fundamentals(ticker)
        if data:
            result[ticker] = data
    return result


# ===========================================================================
# Orchestration: fetch one ticker (3-layer)
# ===========================================================================

def _is_us_ticker(ticker: str) -> bool:
    return "." not in ticker


def fetch_ticker(ticker: str, cik_map: dict[str, str], session: requests.Session) -> dict | None:
    """
    Fetch fundamentals for one ticker using a 2-layer strategy:
      Layer 1 — SEC EDGAR (US tickers with a known CIK)
      Layer 2 — yfinance  (international tickers + US tickers not in EDGAR)
    """
    # Layer 1: EDGAR (US only)
    if _is_us_ticker(ticker):
        cik = cik_map.get(ticker.upper(), "")
        if cik:
            data = fetch_edgar(ticker, cik, session)
            if data and len(data.get("records", [])) >= 2:
                return data

    # Layer 2: yfinance fallback (international + any US ticker that missed EDGAR)
    data = fetch_yfinance_fundamentals(ticker)
    if data and len(data.get("records", [])) >= 2:
        return data

    return None


# ===========================================================================
# Main bootstrap function
# ===========================================================================

def bootstrap_historical_data(tickers: list[str], force_refresh: bool = False) -> None:
    """
    Fetch and cache 10 years of fundamentals + prices for all tickers.

    Parameters
    ----------
    tickers : list[str]
        Full stock universe.
    force_refresh : bool
        Re-fetch even if cached data exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cik_map = _load_cik_map()
    session  = requests.Session()
    session.headers.update(EDGAR_HEADERS)

    # Determine which tickers need fetching
    pending = []
    for t in tickers:
        if force_refresh or not os.path.exists(_fund_path(t)):
            pending.append(t)

    logger.info(f"Fetching fundamentals for {len(pending):,} / {len(tickers):,} tickers...")

    fetched = failed = 0

    def _work(ticker: str) -> tuple[str, dict | None]:
        return ticker, fetch_ticker(ticker, cik_map, session)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_work, t): t for t in pending}
        for fut in as_completed(futures):
            ticker, data = fut.result()
            if data:
                save_fundamentals(data)
                fetched += 1
            else:
                failed += 1
            done = fetched + failed
            if done % 50 == 0:
                logger.info(f"  Progress: {done}/{len(pending)} ({fetched} ok, {failed} failed)")

    # Fetch / update prices
    logger.info("Fetching price history...")
    price_data = fetch_price_history(tickers)
    save_prices(price_data)
    logger.info(f"Price data saved for {len(price_data):,} tickers.")

    # Rebuild coverage.json from ALL files actually on disk (not just this batch)
    _rebuild_coverage_file(tickers)

    logger.info(
        f"Bootstrap complete: {fetched} fundamentals fetched, "
        f"{failed} failed, prices for {len(price_data):,} tickers."
    )


def _rebuild_coverage_file(universe: list[str]) -> None:
    """
    Rebuild coverage.json by scanning the actual fundamentals/ directory.

    For every ticker in `universe`:
      - If a JSON file exists on disk → read its 'source' field
      - Otherwise → mark as 'none'

    This correctly reflects the full cumulative cache state regardless of
    how many incremental runs have been done.
    """
    coverage: dict[str, str] = {}

    # Index cached files: filename stem → source
    cached: dict[str, str] = {}
    if os.path.isdir(FUNDAMENTALS_DIR):
        for fname in os.listdir(FUNDAMENTALS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(FUNDAMENTALS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                # Prefer the stored ticker name; fall back to filename stem
                stored_ticker = d.get("ticker") or fname[:-5]
                source = d.get("source", "unknown")
                cached[stored_ticker] = source
            except Exception:
                pass

    for ticker in universe:
        coverage[ticker] = cached.get(ticker, "none")

    with open(COVERAGE_FILE, "w") as f:
        json.dump(coverage, f, indent=2)
    logger.info(f"Coverage file rebuilt: {len(coverage)} tickers tracked.")


def print_coverage_report() -> None:
    """
    Print a summary of cached fundamental and price data coverage.

    Coverage is computed directly from files on disk so the report is always
    accurate regardless of whether coverage.json is stale or incomplete.
    The full ticker universe is loaded fresh and every ticker is classified.
    """
    # ── Step 1: scan actual files on disk ──────────────────────────────────
    cached: dict[str, str] = {}   # stored_ticker → source
    if os.path.isdir(FUNDAMENTALS_DIR):
        for fname in os.listdir(FUNDAMENTALS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(FUNDAMENTALS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                stored_ticker = d.get("ticker") or fname[:-5]
                source        = d.get("source", "unknown")
                n_records     = len(d.get("records", []))
                # Only count files that actually have usable data
                cached[stored_ticker] = source if n_records >= 1 else "empty"
            except Exception:
                pass

    # ── Step 2: load full universe ──────────────────────────────────────────
    try:
        from data_fetcher import get_stock_universe
        universe = get_stock_universe()
    except Exception:
        # Fall back to whatever is in coverage.json if data_fetcher unavailable
        universe = list(cached.keys())

    # ── Step 3: rebuild coverage.json with the correct picture ─────────────
    cov: dict[str, str] = {}
    for ticker in universe:
        cov[ticker] = cached.get(ticker, "none")
    with open(COVERAGE_FILE, "w") as f:
        json.dump(cov, f, indent=2)

    # ── Step 4: compute stats ───────────────────────────────────────────────
    by_source: dict[str, int] = {}
    for src in cov.values():
        by_source[src] = by_source.get(src, 0) + 1

    total   = len(cov)
    covered = sum(cnt for src, cnt in by_source.items() if src not in ("none", "empty"))

    source_labels = {
        "edgar":    "EDGAR (US, point-in-time)",
        "yfinance": "yfinance (intl + US gap-fill)",
        "fmp":      "FMP",
        "unknown":  "Cached (source unknown)",
        "empty":    "Cached but empty",
        "none":     "No data fetched yet",
    }

    # Tickers with no data — show a sample
    missing = [t for t, s in cov.items() if s == "none"]

    price_files = (
        len([f for f in os.listdir(PRICES_DIR) if f.endswith(".parquet")])
        if os.path.isdir(PRICES_DIR) else 0
    )

    # ── Step 5: print ───────────────────────────────────────────────────────
    print(f"\n{'=' * 56}")
    print(f"  Historical Data Coverage Report")
    print(f"{'=' * 56}")
    print(f"  Fundamental files on disk : {len(cached):,}")
    print(f"  Full universe size        : {total:,}")
    print(f"  With usable fundamentals  : {covered:,}  ({covered/total*100:.1f}% of universe)")
    print(f"  Price series cached       : {price_files:,}")
    print(f"{'-' * 56}")
    print(f"  {'Source':<40} {'Count':>6}")
    print(f"  {'-'*40} {'-'*6}")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        label = source_labels.get(src, src)
        pct   = cnt / total * 100 if total else 0
        print(f"  {label:<40} {cnt:>6,}  ({pct:.1f}%)")
    if missing:
        sample = missing[:10]
        more   = len(missing) - len(sample)
        print(f"\n  Missing ({len(missing):,} tickers): {', '.join(sample)}"
              + (f" … +{more} more" if more else ""))
        print(f"  Run: python main.py --fetch-history  to fetch missing data")
    print(f"{'=' * 56}\n")


# ===========================================================================
# Point-in-time snapshot builder (used by backtest.py)
# ===========================================================================

def _compute_altman_z(snap: dict) -> float | None:
    """
    Altman Z' score (modified, Altman-Hartzell-Peck 1995).

    Uses book equity instead of market cap so it can be computed entirely
    from cached balance-sheet / income-statement data with no live prices.

      Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets
      X4 = book equity / total liabilities   (proxy for market-cap / liabilities)
      X5 = revenue / total assets

    Interpretation: Z' > 2.9 → safe, 1.23–2.9 → grey zone, < 1.23 → distress.
    """
    ta = snap.get("total_assets")
    if not ta or ta == 0:
        return None

    ca   = snap.get("current_assets")    or 0.0
    cl   = snap.get("current_liab")      or 0.0
    re_  = snap.get("retained_earnings") or 0.0
    ebit = snap.get("ebit")              or 0.0
    te   = snap.get("total_equity")      or 0.0
    rev  = snap.get("revenue")           or 0.0

    tl = ta - te  # total liabilities via accounting identity

    x1 = (ca - cl) / ta
    x2 = re_ / ta
    x3 = ebit / ta
    x4 = te / tl if tl != 0 else 0.0
    x5 = rev / ta

    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
    return round(z, 2)


def _compute_piotroski_f(snap_curr: dict, snap_prev: dict | None) -> int:
    """
    Piotroski F-Score from current and (optionally) prior-year snapshots.

    7 of the 9 canonical criteria can be computed from cached data:
      Profitability (4): F1 ROA>0, F2 OCF>0, F3 ΔROA, F4 accruals
      Leverage / liquidity (3): F5 ΔLeverage (uses total_debt), F6 ΔCurrentRatio, F9 ΔAssetTurnover
    Skipped (missing cache fields):
      F7 — no new shares (requires shares_outstanding, not in schema)
      F8 — gross margin increase (requires gross_profit, not in schema)

    Returns an integer 0–7.  When snap_prev is None only F1, F2, F4 are scored (0–3).
    """
    score = 0

    ta0  = snap_curr.get("total_assets")  or 0.0
    ni0  = snap_curr.get("net_income")
    ocf0 = snap_curr.get("operating_cf")

    roa0: float | None = None
    if ta0 and ni0 is not None:
        roa0 = ni0 / ta0
        if roa0 > 0:
            score += 1   # F1: positive ROA

    if ocf0 is not None and ocf0 > 0:
        score += 1       # F2: positive operating cash flow

    if roa0 is not None and ta0 and ocf0 is not None:
        if (ocf0 / ta0) > roa0:
            score += 1   # F4: cash flow ROA > accrual ROA (quality of earnings)

    if snap_prev is not None:
        ta1  = snap_prev.get("total_assets") or 0.0
        ni1  = snap_prev.get("net_income")
        td0  = snap_curr.get("total_debt") or 0.0
        td1  = snap_prev.get("total_debt") or 0.0
        cr0  = snap_curr.get("current_ratio")
        cr1  = snap_prev.get("current_ratio")
        rev0 = snap_curr.get("revenue") or 0.0
        rev1 = snap_prev.get("revenue") or 0.0

        # F3: ROA improving
        if roa0 is not None and ta1 and ni1 is not None:
            roa1 = ni1 / ta1
            if roa0 > roa1:
                score += 1

        # F5: leverage (total_debt / total_assets) decreasing
        if ta0 and ta1:
            lev0 = td0 / ta0
            lev1 = td1 / ta1
            if lev0 < lev1:
                score += 1

        # F6: current ratio improving
        if cr0 is not None and cr1 is not None and cr0 > cr1:
            score += 1

        # F9: asset turnover improving
        if ta0 and ta1 and rev0 and rev1:
            if (rev0 / ta0) > (rev1 / ta1):
                score += 1

    return score


def _effective_filed_date(rec: dict) -> date | None:
    """Return the point-in-time availability date for a record."""
    filed_raw = rec.get("filed_date") or rec.get("period_end")
    if not filed_raw:
        return None
    try:
        d = date.fromisoformat(filed_raw[:10])
    except ValueError:
        return None
    # If no explicit filed_date, assume 45-day reporting lag
    if not rec.get("filed_date") and rec.get("period_end"):
        try:
            d = date.fromisoformat(rec["period_end"][:10]) + timedelta(days=45)
        except ValueError:
            return None
    return d


def get_fundamental_snapshot(
    ticker: str,
    as_of: date,
    use_quarterly: bool = True,
) -> dict | None:
    """
    Return the most recent fundamental record for `ticker` that was
    publicly available on `as_of` (i.e., filed_date <= as_of).

    The returned dict is augmented with two computed fields:
      - ``altman_z``     : Altman Z' (modified, book-equity version)
      - ``piotroski_f``  : Piotroski F-Score (0–7, 7-criteria version)

    Parameters
    ----------
    ticker : str
        Stock ticker.
    as_of : date
        The date we're pretending it is (for point-in-time integrity).
    use_quarterly : bool
        If True, prefer quarterly records (more timely); else annual only.

    Returns
    -------
    dict or None
        The best available fundamental record, or None if no data.
    """
    data = load_fundamentals(ticker)
    if not data:
        return None

    records = data.get("records", [])
    if not records:
        return None

    eligible: list[dict] = []
    for rec in records:
        # Filter by period type
        ptype = rec.get("period_type", "unknown")
        if not use_quarterly and ptype == "quarterly":
            continue

        filed = _effective_filed_date(rec)
        if filed is None or filed > as_of:
            continue
        eligible.append(rec)

    if not eligible:
        return None

    # Best = most recent period_end among eligible records
    best = max(eligible, key=lambda r: r["period_end"])

    # Prior-year record: most recent period_end at least 9 months before best
    best_end = date.fromisoformat(best["period_end"][:10])
    cutoff = best_end - timedelta(days=274)  # ~9 months back
    prior_candidates = [
        r for r in eligible
        if r is not best and date.fromisoformat(r["period_end"][:10]) <= cutoff
    ]
    prior = max(prior_candidates, key=lambda r: r["period_end"]) if prior_candidates else None

    # Augment with computed scores so callers don't need to re-implement
    result = dict(best)
    result["altman_z"]    = _compute_altman_z(best)
    result["piotroski_f"] = _compute_piotroski_f(best, prior)
    return result


def build_universe_snapshot(
    tickers: list[str],
    as_of: date,
    prices: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Build a cross-sectional DataFrame of fundamental data for all tickers
    as they would have been known on `as_of`.

    Returns a DataFrame with columns matching scorer.py expectations plus
    'price_as_of' for the closing price on or before `as_of`.
    """
    rows = []
    for ticker in tickers:
        snap = get_fundamental_snapshot(ticker, as_of)
        if not snap:
            continue

        # Get price as of the date
        price_series = prices.get(ticker)
        price_on_date: float | None = None
        if price_series is not None:
            # Find the last available price <= as_of
            available = [d for d in price_series.index if d <= as_of]
            if available:
                price_on_date = float(price_series[max(available)])

        rows.append({
            "Ticker":              ticker,
            "period_end":          snap.get("period_end"),
            "source":              snap.get("source", "unknown"),
            "net_income":          snap.get("net_income"),
            "revenue":             snap.get("revenue"),
            "total_assets":        snap.get("total_assets"),
            "total_equity":        snap.get("total_equity"),
            "total_debt":          snap.get("total_debt"),
            "current_assets":      snap.get("current_assets"),
            "current_liab":        snap.get("current_liab"),
            "retained_earnings":   snap.get("retained_earnings"),
            "ebit":                snap.get("ebit"),
            "eps_basic":           snap.get("eps_basic"),
            "operating_cf":        snap.get("operating_cf"),
            "dividends_paid":      snap.get("dividends_paid"),
            "roa":                 snap.get("roa"),
            "roe":                 snap.get("roe"),
            "current_ratio":       snap.get("current_ratio"),
            "price_as_of":         price_on_date,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# EPS growth computation from historical records
# ===========================================================================

def compute_eps_growth(ticker: str, as_of: date, years: int) -> float | None:
    """
    Compute annualised EPS growth over `years` years, using only data
    available on `as_of` (point-in-time).

    Returns percentage growth (e.g. 15.0 = 15%) or None.
    """
    data = load_fundamentals(ticker)
    if not data:
        return None

    annual_records = [
        r for r in data.get("records", [])
        if r.get("period_type") == "annual" and r.get("eps_basic") is not None
    ]
    annual_records = [
        r for r in annual_records
        if date.fromisoformat(r["period_end"]) <= as_of
    ]
    annual_records.sort(key=lambda r: r["period_end"])

    if len(annual_records) < years + 1:
        return None

    recent = annual_records[-1]
    past   = annual_records[-(years + 1)]
    eps_recent = recent["eps_basic"]
    eps_past   = past["eps_basic"]

    if eps_past is None or eps_past == 0:
        return None
    try:
        growth = ((eps_recent / eps_past) ** (1 / years) - 1) * 100
        return round(growth, 2)
    except Exception:
        return None


# ===========================================================================
def print_zf_diagnostic(sample_size: int = 300) -> None:
    """
    Print a diagnostic report for Altman Z-Score and Piotroski F-Score coverage
    across the historical fundamental cache.

    Scans up to `sample_size` cached ticker files, calls get_fundamental_snapshot()
    on a recent reference date, and reports:
      - Coverage % (non-null values)
      - Distribution of values
      - Sample records
      - Which raw inputs are missing most often
    """
    import glob
    import math

    ref_date = date.today()

    files = sorted(
        glob.glob(os.path.join(FUNDAMENTALS_DIR, "**", "*.json"), recursive=True)
        + glob.glob(os.path.join(FUNDAMENTALS_DIR, "*.json"))
    )
    if not files:
        print("[DIAG] No cached fundamentals found. Run --fetch first.")
        return

    sample_files = files[:sample_size]
    total = len(sample_files)

    z_vals: list[float] = []
    f_vals: list[int]   = []
    z_null = z_zero = 0
    f_null = f_zero = 0

    missing_inputs: dict[str, int] = {
        "total_assets": 0, "total_equity": 0, "retained_earnings": 0,
        "ebit": 0, "current_assets": 0, "current_liab": 0, "revenue": 0,
        "net_income": 0, "operating_cf": 0,
    }

    sample_z: list[tuple] = []
    sample_f: list[tuple] = []

    for fpath in sample_files:
        try:
            ticker = os.path.splitext(os.path.basename(fpath))[0].replace("_", "/")
            snap = get_fundamental_snapshot(ticker, ref_date)
            if not snap:
                continue

            for field in missing_inputs:
                if snap.get(field) is None:
                    missing_inputs[field] += 1

            z = snap.get("altman_z")
            f = snap.get("piotroski_f")

            if z is None or (isinstance(z, float) and math.isnan(z)):
                z_null += 1
            elif z == 0.0:
                z_zero += 1
            else:
                z_vals.append(z)
                if len(sample_z) < 8:
                    sample_z.append((ticker, z))

            if f is None:
                f_null += 1
            elif f == 0:
                f_zero += 1
            else:
                f_vals.append(f)
                if len(sample_f) < 8:
                    sample_f.append((ticker, f))

        except Exception:
            pass

    scored = total - z_null
    print()
    print("=" * 65)
    print("  Z-SCORE / F-SCORE DIAGNOSTIC REPORT")
    print(f"  Reference date : {ref_date}   Files sampled: {total}")
    print("=" * 65)

    def _pct(n: int) -> str:
        return f"{100 * n / max(total, 1):.1f}%"

    print()
    print("ALTMAN Z'-SCORE  (modified: book equity, no market cap needed)")
    print(f"  Non-null, non-zero : {len(z_vals):>5}  ({_pct(len(z_vals))})")
    print(f"  Zero               : {z_zero:>5}  ({_pct(z_zero)})")
    print(f"  Null / NaN         : {z_null:>5}  ({_pct(z_null)})")
    if z_vals:
        arr = sorted(z_vals)
        print(f"  Min / Median / Max : {min(arr):.2f} / {arr[len(arr)//2]:.2f} / {max(arr):.2f}")
        print(f"  Sample values      : {sample_z}")

    print()
    print("PIOTROSKI F-SCORE  (7-criteria: F7 shares + F8 gross margin skipped)")
    print(f"  Non-null, non-zero : {len(f_vals):>5}  ({_pct(len(f_vals))})")
    print(f"  Zero               : {f_zero:>5}  ({_pct(f_zero)})")
    print(f"  Null / NaN         : {f_null:>5}  ({_pct(f_null)})")
    if f_vals:
        from collections import Counter
        dist = Counter(f_vals)
        print(f"  Distribution       : {dict(sorted(dist.items()))}")
        print(f"  Sample values      : {sample_f}")

    print()
    print("RAW INPUT FIELD NULL RATES  (out of all sampled snapshots)")
    print(f"  {'Field':<22}  Null count  Null %")
    print(f"  {'-'*22}  ----------  ------")
    for field, cnt in sorted(missing_inputs.items(), key=lambda x: -x[1]):
        print(f"  {field:<22}  {cnt:>10}  {_pct(cnt)}")

    print()
    print("NOTES")
    print("  - Z' uses total_liabilities = total_assets - total_equity (accounting identity)")
    print("  - F7 (no new shares) skipped: shares_outstanding not in cache schema")
    print("  - F8 (gross margin) skipped: gross_profit not in cache schema")
    print("  - F-score range is 0-7 (not 0-9)")
    print("=" * 65)


# CLI
# ===========================================================================

if __name__ == "__main__":
    from data_fetcher import get_stock_universe

    parser = argparse.ArgumentParser(description="Historical data bootstrap")
    parser.add_argument("--fetch",       action="store_true", help="Fetch/refresh all historical data")
    parser.add_argument("--coverage",    action="store_true", help="Print coverage report")
    parser.add_argument("--diagnose-zf", action="store_true", help="Print Z-score / F-score diagnostic")
    parser.add_argument("--force",       action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.coverage:
        print_coverage_report()

    if getattr(args, "diagnose_zf", False):
        print_zf_diagnostic()

    if args.fetch:
        tickers = get_stock_universe()
        bootstrap_historical_data(tickers, force_refresh=args.force)

    if not args.fetch and not args.coverage and not getattr(args, "diagnose_zf", False):
        parser.print_help()
