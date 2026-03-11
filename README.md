# Hedge Fund Global Stock Screener

A Python program that automatically screens **500–2,000+ global stocks**, collects
21 financial metrics per stock, calculates a proprietary composite score, ranks all
stocks, and logs everything to Google Sheets with rich formatting.

---

## Features

| Feature | Details |
|---------|---------|
| **Universe** | S&P 500, NASDAQ-100, Russell 1000 + FTSE 100, DAX 40, CAC 40, Nikkei 225, TSX, ASX 200, Hang Seng, Eurostoxx |
| **Metrics** | P/E, P/B, Net Income, EPS growth (1/3/5yr + forward), ROA, ROE, ROIC, Dividend Yield, Payout Ratio, Current Ratio, Altman Z-Score, Piotroski F-Score |
| **Scoring** | Custom composite formula with 12 weighted factors |
| **Google Sheets** | Dated tab, frozen header, alternating rows, gradient score colours, gold top-10 highlight, auto-sized columns, Summary tab |
| **Performance** | Parallel fetching (10 workers), retry with exponential back-off, tqdm progress bar, rate limiting |
| **Resilience** | CSV backup before Sheets write; graceful handling of missing data |

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Google Sheets API credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the **Google Sheets API** and **Google Drive API**
   - APIs & Services → Enable APIs → search "Google Sheets API" → Enable
   - Repeat for "Google Drive API"
4. Create a **Service Account**
   - APIs & Services → Credentials → Create Credentials → Service Account
   - Give it a name (e.g. `stock-screener`)
   - Click Done
5. Generate a JSON key for the service account
   - Click the service account → Keys → Add Key → Create new key → JSON
   - Save the downloaded file as **`credentials.json`** in the project root
6. Share your target Google Sheet with the service account email
   - Open your Google Sheet
   - Share → paste the service account email (e.g. `stock-screener@my-project.iam.gserviceaccount.com`)
   - Give **Editor** access

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Paste your Google Sheet ID from the URL:
# https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit
GOOGLE_SHEET_ID=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms

# Path to your service account credentials file
CREDENTIALS_FILE=credentials.json
```

> **Tip:** Leave `GOOGLE_SHEET_ID` empty on first run — the script will
> automatically create a new spreadsheet and print its URL.

### 4. Run the screener

```bash
python main.py
```

**Expected console output:**
```
[INFO] Loading stock universe... 1,247 stocks found
[INFO] Fetching financial data... ████████░░ 78% (974/1247)
[INFO] Calculating composite scores...
[INFO] Writing to Google Sheets...
[INFO] ✅ Done! 1,103 stocks ranked. Top stock: BRK-B (Score: 487.3)
[INFO] Sheet URL: https://docs.google.com/spreadsheets/d/...
```

---

## Project Structure

```
Hedge Fund Project/
├── main.py            # Entry point — orchestration
├── data_fetcher.py    # Fetching logic (yfinance, yahooquery, calculated scores)
├── scorer.py          # Composite score formula and ranking
├── sheets_writer.py   # Google Sheets writing and formatting
├── config.py          # Constants, weights, ticker universe lists
├── requirements.txt
├── .env.example       # Template for environment variables
├── .env               # Your local config (git-ignored)
├── credentials.json   # Service account key (git-ignored — keep secret!)
└── errors.log         # Auto-generated error log
```

---

## Composite Score Formula

```
Score = 0.3(E₁) + 0.8(E₃) + 1.4(E₅) + 2.5(Ef)
      + Ra + Re + 2(Rc)
      + 1.4(C) + 3.25(Z) + 2.65(F) + 0.2(A)
      + 3.25 × (Y×(2 − Pr/100) − (5×Pe + 3×Pb))
```

| Symbol | Metric | Unit |
|--------|--------|------|
| E₁ | 1-Year EPS Growth | % (e.g. 15) |
| E₃ | 3-Year EPS CAGR | % |
| E₅ | 5-Year EPS CAGR | % |
| Ef | Forward EPS Growth Estimate | % |
| Ra | Return on Assets | % |
| Re | Return on Equity | % |
| Rc | Return on Invested Capital | % |
| C  | Current Ratio | raw ratio |
| Z  | Altman Z-Score | raw |
| F  | Piotroski F-Score | 0–9 |
| A  | Annual Net Income | USD millions |
| Y  | Dividend Yield | % |
| Pr | Payout Ratio | % |
| Pe | P/E Ratio | raw |
| Pb | P/B Ratio | raw |

Missing values default to **0** in the formula (never crash).
Stocks with **more than 8 of 15 metrics missing** are excluded entirely.

---

## Data Sources

| Source | Used For |
|--------|----------|
| `yfinance` | P/E, P/B, ROA, ROE, net income, dividend yield, payout ratio, current ratio, earnings history, balance sheet, income statement, cash flow |
| `yahooquery` | Forward EPS growth estimates (fallback) |
| In-house calculation | ROIC, Altman Z-Score, Piotroski F-Score, 3yr/5yr EPS CAGR |
| Wikipedia (pandas HTML) | S&P 500, NASDAQ-100, Russell 1000 ticker lists |

---

## Google Sheet Layout

**Main tab** (`Stock Screener YYYY-MM-DD`):
- Row 1: Bold white text on navy (`#1a237e`) — frozen
- Alternating white / light-grey rows
- Composite Score: red→white→green gradient
- Ranks 1–10: gold background + bold
- All columns auto-sized

**Summary tab**:
- Total stocks screened / skipped / ranked
- Date & time of last run
- Top 25 stocks (ticker, name, score, rank)
- Average composite score
- Sector breakdown of top 100

---

## Output Files

| File | Description |
|------|-------------|
| `stock_screener_YYYY-MM-DD.csv` | Local backup of all ranked data |
| `errors.log` | Per-stock fetch errors and warnings |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: credentials.json` | Download your service account JSON key and rename it `credentials.json` |
| `SpreadsheetNotFound` | Share the sheet with your service account email, or leave `GOOGLE_SHEET_ID` blank to auto-create |
| Very few stocks returned | yfinance rate limits — re-run after a few minutes; the script already retries 3× with back-off |
| `No module named 'yahooquery'` | Run `pip install yahooquery` |
| International stocks missing data | Expected — many non-US stocks have limited free data; they score with 0 on missing fields |

---

## Security Notes

- **Never commit `credentials.json` or `.env`** to version control.
- Add them to `.gitignore`:
  ```
  credentials.json
  .env
  *.log
  *.csv
  ```
