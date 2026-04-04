# Hedge Fund Global Stock Screener

A quantitative equity screening and ranking system that evaluates 500–2,000+ global stocks across 11 major indices, scores them via a proprietary composite formula, and delivers ranked output to Google Sheets with automated scheduling, Slack alerting, and run validation.

Built independently as a personal project to understand how systematic investment processes work in practice.

---

## System Architecture
```
Universe Definition (config.py)
         ↓
Data Fetching — parallel (10 workers), retry + exponential back-off (data_fetcher.py)
         ↓
Score Calculation — 12-factor composite formula (scorer.py)
         ↓
        ├── Google Sheets output — gradient formatting, summary tab (sheets_writer.py)
        ├── CSV backup — local ranked data
        └── Delta tracking — top-25 entry/exit detection (delta_tracker.py)
         ↓
Scheduler — twice-daily on weekdays, dynamic UTC from market hours (scheduler.py)
         ↓
        ├── Slack alerts — startup, run summary, validation warnings (alerts.py)
        ├── Price tracking — live prices for top holdings (price_tracker.py)
        └── Efficacy analysis — ranking predictive accuracy over time (efficacy_analyzer.py)
```

---

## Design Decisions

**Why a composite score rather than a single metric?**
No single financial ratio reliably predicts outperformance across market cycles. The composite combines quality (ROIC, ROE, ROA), growth (EPS CAGR, forward estimates), safety (Altman Z-Score, Current Ratio), value (P/E, P/B, dividend yield), and financial health (Piotroski F-Score) to produce a multidimensional ranking.

**Why weight Altman Z-Score (3.25×) and Piotroski F-Score (2.65×) so heavily?**
Both are proven multi-factor models with strong academic backing for predicting financial distress and earnings quality respectively. They act as quality gates — a stock can score well on growth and value metrics but still rank poorly if its balance sheet signals distress.

**Why exclude stocks with more than 8 of 15 metrics missing?**
Stocks with sparse data produce unreliable composite scores. The 8/15 threshold was calibrated to exclude stocks where missing data would meaningfully distort the ranking while retaining the majority of international equities that have partial coverage.

**Why parallel fetching with 10 workers?**
Sequential fetching of 1,200+ stocks from the Yahoo Finance API takes ~40 minutes. Parallel fetching with rate limiting and exponential back-off reduces this to ~4 minutes while avoiding API bans.

**Why a dynamic scheduler rather than fixed cron times?**
Market open and close times vary by exchange and shift with Daylight Saving Time. The scheduler calculates run times dynamically in UTC each day from real market hours, so the morning run always fires shortly after the earliest market opens (HKEX ~01:30 UTC) and the afternoon run after the earliest close (TSE ~06:30 UTC).

**Why are P/E and P/B percentile-ranked within the universe rather than used as absolute values?**
Absolute P/E and P/B values created a structural bias toward Japanese stocks, which trade at very low P/B ratios (often below 1.0) due to historical deflation and TSE reform pressure rather than superior business quality. A Japanese industrial with P/B = 0.8 received a valuation-term score 400+ points higher than a high-quality US company with P/B = 8, purely because of this regional structural difference — before any quality metrics were considered. Percentile normalization preserves the value signal (cheap-relative-to-peers is rewarded) while eliminating the cross-regional structural bias. The formula structure is unchanged; only the Pe and Pb inputs are relativised.

---

## Composite Score Formula
```
Score = 0.3(E₁) + 0.8(E₃) + 1.4(E₅) + 2.5(Ef)
      + Ra + Re + 2(Rc)
      + 1.4(C) + 3.25(Z) + 2.65(F) + 0.2(A)
      + 3.25 × (Y×(2 − Pr/100) − (5×Pe + 3×Pb))
```

> **Note:** Pe and Pb are cross-sectionally normalized to percentile ranks (0–100) within the screened universe on each run before being used in the formula. A stock at the 20th percentile for P/E (cheap relative to peers) receives Pe = 20; a stock at the 80th percentile (expensive relative to peers) receives Pe = 80, regardless of the absolute ratio. Stocks with missing P/E or P/B data receive the neutral default of 50. All other factors remain on their absolute scales.

| Symbol | Metric | Weight Rationale |
|--------|--------|-----------------|
| E₁ | 1-Year EPS Growth | Recent momentum, low weight to avoid noise |
| E₃ | 3-Year EPS CAGR | Medium-term growth quality |
| E₅ | 5-Year EPS CAGR | Sustained compounding ability |
| Ef | Forward EPS Growth | Market consensus on future trajectory |
| Ra | Return on Assets | Capital efficiency |
| Re | Return on Equity | Shareholder return generation |
| Rc | Return on Invested Capital | Best single measure of business quality (2× weight) |
| C | Current Ratio | Short-term liquidity |
| Z | Altman Z-Score | Bankruptcy risk (highest single weight) |
| F | Piotroski F-Score | 9-point financial health composite |
| A | Annual Net Income | Absolute profitability anchor |
| Y | Dividend Yield | Income, adjusted by payout sustainability |
| Pr | Payout Ratio | Dividend safety check |
| Pe | P/E Ratio | Valuation (penalised) — **percentile-ranked within universe** |
| Pb | P/B Ratio | Asset valuation (penalised) — **percentile-ranked within universe** |

Missing values default to **0** (or **50** for Pe/Pb after normalization). Stocks missing more than 8 of 15 metrics are excluded entirely.

---

## Stock Universe

| Index | Region |
|-------|--------|
| S&P 500, NASDAQ-100, Russell 1000 | United States |
| FTSE 100 | United Kingdom |
| DAX 40 | Germany |
| CAC 40 | France |
| Nikkei 225 | Japan |
| TSX | Canada |
| ASX 200 | Australia |
| Hang Seng | Hong Kong |
| Eurostoxx | Europe |

---

## File Structure
```
├── main.py               # Entry point and CLI
├── config.py             # Weights, universe lists, constants
├── data_fetcher.py       # yfinance/yahooquery fetching, Altman Z, Piotroski F, ROIC
├── scorer.py             # Composite formula and ranking logic
├── sheets_writer.py      # Google Sheets output and formatting
├── scheduler.py          # APScheduler daemon, dynamic market-hour timing
├── alerts.py             # Slack notifications (startup, runs, validation warnings)
├── delta_tracker.py      # Top-25 entry/exit change detection between runs
├── price_tracker.py      # Live price tracking for top holdings
├── efficacy_analyzer.py  # Ranking predictive accuracy analysis
├── backtest.py           # Historical backtesting against scored rankings
├── weight_optimizer.py   # Score weight tuning via historical performance
├── factor_analysis.py    # Per-factor contribution and correlation analysis
├── stress_test.py        # Portfolio stress testing under historical scenarios
├── historical_data.py    # Historical price/metric data retrieval utilities
├── requirements.txt
├── .env.example          # Environment variable template
├── .env                  # Local config (git-ignored)
├── credentials.json      # Google service account key (git-ignored)
└── scripts/
    ├── setup_windows_task.bat   # Register scheduler as a Windows startup task (run once)
    ├── register_task.ps1        # PowerShell Task Scheduler XML registration
    ├── start_scheduler.bat      # Start scheduler as a detached background process
    ├── stop_scheduler.bat       # Stop the running scheduler via PID file
    └── scheduler_status.bat     # Show next scheduled run times
```

---

## Run Validation

After each live run the system automatically checks:
1. At least 400 stocks scored (flags partial run if fewer)
2. Top-25 average rank shift ≤ 15 positions vs previous run (detects data errors)
3. No stock score is > 5 standard deviations from the mean (outlier detection)

Failures are logged to `errors.log` and trigger a Slack alert.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Google Sheets API credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Sheets API** and **Google Drive API**
3. Create a **Service Account** and download the JSON key
4. Save the key as `credentials.json` in the project root
5. Share your Google Sheet with the service account email (Editor access)

### 3. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_SHEET_ID=your_sheet_id_here   # leave blank to auto-create
CREDENTIALS_FILE=credentials.json
```

### 4. Run
```bash
python main.py              # single run
python main.py --backtest   # backtest mode
```

### 5. Windows Deployment (optional)

Scripts in `scripts/` automate running the scheduler as a persistent background process on Windows.

**One-time setup — register as a startup task:**
```bat
scripts\setup_windows_task.bat
```
Registers `HedgeFundScheduler` in Windows Task Scheduler to launch automatically at logon using `pythonw.exe` (no console window). Restarts up to 3 times on failure.

**Manual start/stop (without Task Scheduler):**
```bat
scripts\start_scheduler.bat      :: Start scheduler as a detached background process
scripts\stop_scheduler.bat       :: Stop it (reads PID from logs\scheduler.pid)
scripts\scheduler_status.bat     :: Show next scheduled run times
```

---

## Data Sources

| Source | Used For |
|--------|----------|
| `yfinance` | P/E, P/B, ROA, ROE, net income, dividend yield, payout ratio, current ratio, earnings history |
| `yahooquery` | Forward EPS growth estimates (fallback) |
| In-house calculation | ROIC, Altman Z-Score, Piotroski F-Score, 3yr/5yr EPS CAGR |
| Wikipedia (pandas HTML) | S&P 500, NASDAQ-100, Russell 1000 ticker lists |

---

## Security

Never commit `credentials.json` or `.env`. Both are git-ignored by default.
```
credentials.json
.env
*.log
*.csv
```
