# Data Sources Specification: Fund Manager Skill Detector

## 1. Executive Summary
This document defines the exact data sources, tickers, identifiers, sampling frequencies, raw units, required transformations, reliability limits, and active-management verification protocols for the Fund Manager Skill Detector project.

---

## 2. Acquired Primary Data Sources (Task 3 Execution Summary)

### 2.1 Target Actively Managed U.S. Equity Mutual Fund Universe
- **Acquired Universe Size**: 12 Actively Managed U.S. Equity Mutual Funds.
- **Requested Evaluation Window**: `2019-01-01` through `2023-12-31`.
- **Actual Acquired Window**: `2019-01-03` through `2023-12-29` (1,257 trading days per fund).
- **Data Provider**: Yahoo Finance (`yfinance` API).
- **Raw Storage Location**: `data/raw/raw_fund_prices.csv` (1.84 MB).

#### Primary U.S. Equity Active Fund Universe & Acquired Observation Counts:

| # | Ticker | Fund Name | Category / Style | Acquired Date Range | Raw Rows | Screening Status |
|---|---|---|---|---|---|---|
| 1 | `AGTHX` | American Funds Growth Fund of America | Large Growth Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 2 | `FCNTX` | Fidelity Contrafund | Large Growth Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 3 | `DODGX` | Dodge & Cox Stock Fund | Large Value Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 4 | `FDGRX` | Fidelity Growth Company Fund | Large Growth Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 5 | `TRBCX` | T. Rowe Price Blue Chip Growth | Large Growth Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 6 | `AIVSX` | Investment Company of America | Large Value Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 7 | `FLPSX` | Fidelity Low-Priced Stock Fund | Mid-Cap Value Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 8 | `AMECX` | Income Fund of America | Equity Income Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 9 | `PRHSX` | T. Rowe Price Health Sciences | Sector Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 10 | `PAGRX` | Permanent Portfolio Aggressive Growth Fund | Multi-asset / exploratory, not broad-equity comparable | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 11 | `FMAGX` | Fidelity Magellan Fund | Large Blend Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |
| 12 | `MIGFX` | MFS Massachusetts Investors Growth Stock | Large Growth Active | 2019-01-03 to 2023-12-29 | 1,257 | `CANDIDATE` |

---

### 2.2 Market Benchmark
- **Target Benchmark**: S&P 500 Index.
- **Ticker / Provider**: `^GSPC` (Yahoo Finance).
- **Raw Storage Location**: `data/raw/raw_benchmark_sp500.csv` (149 KB).
- **Actual Acquired Window**: `2019-01-03` to `2023-12-29` (1,257 trading days).

---

### 2.3 Primary Risk-Free Rate & Multi-Factor Data
- **Authoritative Provider**: Kenneth R. French Data Library (Tuck School of Business, Dartmouth).
- **Downloaded Files**:
  1. `data/raw/raw_fama_french_daily.csv` (41.1 KB): Contains `Mkt-RF`, `SMB`, `HML`, `RF`.
  2. `data/raw/raw_momentum_daily.csv` (21.8 KB): Contains `Mom` / `UMD`.
- **Actual Acquired Window**: `2019-01-03` to `2023-12-29` (1,257 trading days).
- **Transformation**: Raw factor percentages divided by 100 ($0.15 \rightarrow 0.0015$).

---

## 3. Active-Management Screening Status (Task 3 Record)

All 12 funds are recorded in `data/raw/screening_metadata.json` with status `"CANDIDATE"` and verification status `"UNVERIFIED_CANDIDATE"`. No fund has been falsely marked verified active prior to prospectus/active share evidence audit in Task 4.


**Identity correction:** PAGRX is Permanent Portfolio Aggressive Growth Fund, not T. Rowe Price Growth Stock Fund. The latter uses PRGFX and is not included because a PRGFX price history has not been acquired in this saved dataset. AMECX (income-oriented) and PRHSX (healthcare-sector) are also retained only as exploratory, non-peer observations.
