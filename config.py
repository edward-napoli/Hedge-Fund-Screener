"""
config.py — Constants, weights, column definitions, and stock universe tickers.
"""

import os

# ---------------------------------------------------------------------------
# Composite Score formula weights
# Score = 0.3(E1) + 0.8(E3) + 1.4(E5) + 2.5(Ef) + Ra + Re + 2(Rc)
#       + 1.4(C) + 3.25(Z) + 2.65(F) + 0.2(A)
#       + 3.25 * (Y*(2 - Pr) - (5*Pe + 3*Pb))
# All percentage metrics kept as raw % values (e.g. 15% = 15, not 0.15)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "E1":       0.3,
    "E3":       0.8,
    "E5":       1.4,
    "Ef":       2.5,
    "Ra":       1.0,
    "Re":       1.0,
    "Rc":       2.0,
    "C":        1.4,
    "Z":        3.25,
    "F":        2.65,
    "A":        0.2,
    "Y_outer":  3.25,   # outer coefficient for dividend/valuation term
    "Pe_coef":  5.0,
    "Pb_coef":  3.0,
}

# ---------------------------------------------------------------------------
# Ordered column list (matches Google Sheet layout)
# ---------------------------------------------------------------------------
COLUMNS = [
    "Ticker",
    "Company Name",
    "Exchange",
    "Country",
    "Sector",
    "Industry",
    "P/E Ratio",
    "P/B Ratio",
    "Annual Net Income (USD M)",
    "1-Year EPS Growth %",
    "3-Year EPS Growth %",
    "5-Year EPS Growth %",
    "Future EPS Growth Est. %",
    "ROA %",
    "ROE %",
    "ROIC %",
    "Dividend Yield %",
    "Payout Ratio %",
    "Current Ratio",
    "Altman Z-Score",
    "Piotroski F-Score",
    "Score Delta",
    "Rank Delta",
    "Composite Score",
    "Rank",
]

# Metrics used in the formula (used to count missing fields)
FORMULA_METRICS = [
    "P/E Ratio",
    "P/B Ratio",
    "Annual Net Income (USD M)",
    "1-Year EPS Growth %",
    "3-Year EPS Growth %",
    "5-Year EPS Growth %",
    "Future EPS Growth Est. %",
    "ROA %",
    "ROE %",
    "ROIC %",
    "Dividend Yield %",
    "Payout Ratio %",
    "Current Ratio",
    "Altman Z-Score",
    "Piotroski F-Score",
]

# ---------------------------------------------------------------------------
# Performance / reliability settings
# ---------------------------------------------------------------------------
MAX_WORKERS           = 10
RATE_LIMIT_DELAY      = 0.5      # seconds between requests per thread
MAX_RETRIES           = 3
RETRY_BACKOFF_BASE    = 2        # exponential back-off multiplier
MIN_MARKET_CAP        = 500_000_000   # $500 M USD filter
MAX_MISSING_METRICS   = 8        # skip stock if > this many metrics are missing

# ---------------------------------------------------------------------------
# File / path settings (overridden via .env)
# ---------------------------------------------------------------------------
CREDENTIALS_FILE      = os.getenv("CREDENTIALS_FILE", "credentials.json")
LOG_FILE              = "logs/errors.log"
SLACK_WEBHOOK_URL     = os.getenv("SLACK_WEBHOOK_URL", "")
ENABLE_SLACK_ALERTS   = os.getenv("ENABLE_SLACK_ALERTS", "false").lower() == "true"
PARTIAL_RUN_THRESHOLD = 400   # flag run as "partial" if fewer stocks fetched

# ---------------------------------------------------------------------------
# Google Sheets colours  (normalised 0–1 RGB)
# ---------------------------------------------------------------------------
HEADER_BG_COLOR    = {"red": 0.102, "green": 0.137, "blue": 0.494}  # #1a237e
HEADER_TEXT_COLOR  = {"red": 1.0,   "green": 1.0,   "blue": 1.0}
ROW_COLOR_ODD      = {"red": 1.0,   "green": 1.0,   "blue": 1.0}    # white
ROW_COLOR_EVEN     = {"red": 0.961, "green": 0.961, "blue": 0.961}  # #f5f5f5
GOLD_COLOR         = {"red": 1.0,   "green": 0.843, "blue": 0.0}    # gold
GREEN_COLOR        = {"red": 0.204, "green": 0.659, "blue": 0.325}
RED_COLOR          = {"red": 0.957, "green": 0.263, "blue": 0.212}
WHITE_COLOR        = {"red": 1.0,   "green": 1.0,   "blue": 1.0}

# ---------------------------------------------------------------------------
# Exchange suffix → (exchange label, country) mapping
# ---------------------------------------------------------------------------
EXCHANGE_MAP = {
    ".L":  ("LSE",               "United Kingdom"),
    ".DE": ("XETRA",             "Germany"),
    ".PA": ("Euronext Paris",    "France"),
    ".T":  ("TSE",               "Japan"),
    ".TO": ("TSX",               "Canada"),
    ".AX": ("ASX",               "Australia"),
    ".HK": ("SEHK",              "Hong Kong"),
    ".AS": ("Euronext Amsterdam","Netherlands"),
    ".BR": ("Euronext Brussels", "Belgium"),
    ".HE": ("Nasdaq Helsinki",   "Finland"),
    ".MI": ("Borsa Italiana",    "Italy"),
    ".MC": ("BME",               "Spain"),
    ".SW": ("SIX",               "Switzerland"),
    ".ST": ("Nasdaq Stockholm",  "Sweden"),
    ".OL": ("Oslo Bors",         "Norway"),
    ".CO": ("Nasdaq Copenhagen", "Denmark"),
}

# ---------------------------------------------------------------------------
# International ticker lists (hardcoded — US tickers fetched dynamically)
# ---------------------------------------------------------------------------

FTSE_100 = [
    "AAF.L","AAL.L","ABF.L","AHT.L","ANTO.L","AV.L","AZN.L","BA.L","BARC.L",
    "BATS.L","BKG.L","BP.L","BRBY.L","BT-A.L","CCH.L","CNA.L","CPG.L","CRDA.L",
    "DCC.L","DGE.L","ENT.L","EXPN.L","FRES.L","GLEN.L","GSK.L","HIK.L","HL.L",
    "HLMA.L","HSBA.L","IAG.L","ICG.L","IHG.L","III.L","IMB.L","ITRK.L","ITV.L",
    "JD.L","KGF.L","LAND.L","LGEN.L","LLOY.L","LSEG.L","MKS.L","MNDI.L","MNG.L",
    "MRO.L","NG.L","NWG.L","NXT.L","OCDO.L","PHNX.L","PRU.L","PSN.L","PSON.L",
    "REL.L","RIO.L","RKT.L","RMV.L","RR.L","RS1.L","SBRY.L","SDR.L","SGE.L",
    "SHEL.L","SKG.L","SN.L","SPX.L","SSE.L","STAN.L","STJ.L","SVT.L","TSCO.L",
    "ULVR.L","UU.L","VOD.L","WEIR.L","WPP.L","WTB.L","DPH.L","RTO.L","AUTO.L",
    "BNZL.L","CLLN.L","CRH.L","EMG.L","FLTR.L","PCT.L","SMDS.L","SMWH.L",
    "WDS.L","WG.L",
]

DAX_40 = [
    "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE","BMW.DE","BNR.DE",
    "CON.DE","1COV.DE","DB1.DE","DHER.DE","DTE.DE","DWNI.DE","ENR.DE","EOAN.DE",
    "FRE.DE","HEI.DE","HEN3.DE","IFX.DE","LIN.DE","MBG.DE","MRK.DE","MTX.DE",
    "MUV2.DE","P911.DE","PAH3.DE","QGEN.DE","RHM.DE","RWE.DE","SAP.DE","SHL.DE",
    "SIE.DE","SRT3.DE","SY1.DE","VOW3.DE","VNA.DE","WCH.DE","ZAL.DE","HNR1.DE",
]

CAC_40 = [
    "AC.PA","ACA.PA","AI.PA","AIR.PA","ALO.PA","BN.PA","BNP.PA","CA.PA",
    "CAP.PA","CS.PA","DSY.PA","EN.PA","ENGI.PA","EL.PA","ERF.PA","GLE.PA",
    "HO.PA","KER.PA","LR.PA","MC.PA","ML.PA","OR.PA","ORA.PA","PUB.PA",
    "RI.PA","RMS.PA","RNO.PA","SAF.PA","SAN.PA","SGO.PA","SU.PA","SW.PA",
    "TEC.PA","TTE.PA","VIE.PA","VIV.PA","WLN.PA","MT.PA","STM.PA","EDF.PA",
]

NIKKEI_SUBSET = [
    "6758.T","7203.T","9984.T","6902.T","9433.T","7974.T","8306.T","6501.T",
    "9432.T","4519.T","7267.T","8035.T","6367.T","6954.T","9983.T","4063.T",
    "8411.T","7751.T","3382.T","4452.T","4568.T","6702.T","8766.T","7270.T",
    "2802.T","2914.T","7201.T","6594.T","9022.T","8801.T","8802.T","9020.T",
    "5401.T","6301.T","6326.T","6971.T","6503.T","4503.T","4901.T","8031.T",
    "8058.T","8053.T","8001.T","6752.T","6762.T","7741.T","8309.T","3407.T",
    "6724.T","5711.T","4543.T","4507.T","8750.T","7733.T","6645.T","9613.T",
    "4661.T","8604.T","8601.T","9064.T","2503.T","7832.T","9531.T","4005.T",
]

TSX_SUBSET = [
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","MFC.TO","SLF.TO","GWO.TO",
    "FFH.TO","POW.TO","CNQ.TO","SU.TO","CVE.TO","IMO.TO","TRP.TO","ENB.TO",
    "PPL.TO","CNR.TO","CP.TO","WCN.TO","ATD.TO","MRU.TO","L.TO","DOL.TO",
    "SHOP.TO","CSU.TO","GIB-A.TO","OTEX.TO","CAE.TO","CCO.TO","ABX.TO",
    "AEM.TO","K.TO","WPM.TO","FM.TO","TECK-B.TO","NTR.TO","BCE.TO","T.TO",
    "RCI-B.TO","BAM.TO","BIP-UN.TO","BEP-UN.TO","H.TO","EMP-A.TO","WN.TO",
    "QSR.TO","MG.TO","TRI.TO","FSV.TO",
]

ASX_SUBSET = [
    "BHP.AX","CBA.AX","CSL.AX","ANZ.AX","WBC.AX","NAB.AX","WES.AX","MQG.AX",
    "WOW.AX","RIO.AX","TLS.AX","AMC.AX","ALL.AX","FMG.AX","TCL.AX","GMG.AX",
    "STO.AX","WDS.AX","XRO.AX","BXB.AX","COL.AX","QAN.AX","REA.AX","SHL.AX",
    "IAG.AX","APA.AX","ORI.AX","SUN.AX","RMD.AX","CPU.AX","JHX.AX","QBE.AX",
    "AMP.AX","ASX.AX","AZJ.AX","MPL.AX","TWE.AX","MIN.AX","ALQ.AX","CGF.AX",
    "LLC.AX","NHF.AX","NST.AX","S32.AX","WPR.AX","DXS.AX","NCM.AX","EVN.AX",
    "APT.AX","ZIP.AX",
]

HANG_SENG_SUBSET = [
    "0700.HK","0005.HK","0941.HK","0388.HK","1299.HK","2318.HK","1398.HK",
    "3988.HK","0939.HK","2628.HK","0883.HK","1928.HK","0011.HK","0016.HK",
    "0688.HK","0002.HK","0003.HK","0006.HK","0012.HK","0017.HK","0019.HK",
    "0027.HK","0066.HK","0101.HK","0151.HK","0175.HK","0267.HK","0291.HK",
    "0316.HK","0384.HK","0386.HK","0669.HK","0762.HK","0823.HK","0857.HK",
    "0868.HK","0960.HK","1038.HK","1044.HK","1093.HK","1109.HK","1177.HK",
    "1211.HK","1288.HK","1336.HK","1378.HK","1810.HK","1876.HK","1997.HK",
    "2020.HK","9988.HK","3690.HK",
]

EUROSTOXX_EXTRA = [
    # Netherlands
    "ADYEN.AS","ASML.AS","HEIA.AS","INGA.AS","NN.AS","PHIA.AS","RAND.AS",
    "UNA.AS","WKL.AS","ABN.AS","AKZA.AS","AGN.AS","AD.AS",
    # Belgium
    "ABI.BR","UCB.BR","KBC.BR","SOLB.BR",
    # Finland
    "NOKIA.HE","SAMPO.HE","FORTUM.HE","NESTE.HE",
    # Italy
    "ENEL.MI","ISP.MI","UCG.MI","STM.MI","ENI.MI","G.MI","PRY.MI","TIT.MI",
    # Spain
    "AMS.MC","BBVA.MC","BKT.MC","CABK.MC","ELE.MC","FER.MC","IBE.MC",
    "ITX.MC","REP.MC","SAN.MC",
    # Switzerland
    "NOVN.SW","ROG.SW","ABBN.SW","NESN.SW","UBSG.SW","CSGN.SW","ZURN.SW",
    # Sweden
    "VOLV-B.ST","ERIC-B.ST","ATCO-A.ST","SEB-A.ST","SHB-A.ST","SWED-A.ST",
    # Norway
    "EQNR.OL","DNB.OL","TEL.OL",
]

# All international tickers combined
ALL_INTERNATIONAL = (
    FTSE_100 + DAX_40 + CAC_40 + NIKKEI_SUBSET +
    TSX_SUBSET + ASX_SUBSET + HANG_SENG_SUBSET + EUROSTOXX_EXTRA
)
