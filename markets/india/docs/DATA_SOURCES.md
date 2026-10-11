# Data Sources Specification: Indian Fund Manager Skill Detector

## 1. Summary
This document specifies the data sources, identifiers, frequencies, raw units, transformations and verification
status for the Indian Fund Manager Skill Detector. Everything below describes the data **as shipped in `data/raw/`**.

---

## 2. Primary Data Sources

### 2.1 Fund universe (12 actively managed Indian equity schemes, Direct Plan - Growth)
- **Provider**: AMFI scheme master / NAV history via the open `mfapi.in` REST API (`https://api.mfapi.in/mf/<scheme code>`).
- **Requested window**: `2019-01-01` to `2023-12-31`. **Acquired**: `2019-01-01` to `2023-12-29` (1,231-1,232 NAVs per fund).
- **Raw file**: `data/raw/raw_fund_prices.csv` (`Date`, `Adj Close` = NAV, `symbol` = AMFI scheme code).
- **Plan type**: Direct Plan - Growth, which has no distributor commission and reinvests dividends, so the NAV series is a total-return series
  net of the scheme's expenses. (Direct plans exist for index funds as well; this is a design choice of the project, not a SEBI mandate for active schemes.)
- **Identity check (performed in review)**: every scheme code was queried on `mfapi.in`; each returns the intended scheme, category and plan
  (e.g. 120586 is "ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund) - Direct Plan - Growth", 118989 is "HDFC Mid Cap Fund" (formerly Mid-Cap Opportunities)).

#### Registry (`data/raw/screening_metadata.json`, generated from `FUND_METADATA` in `src/data_loader.py`)

| Scheme Code | Fund | SEBI Category | Named Manager | AMC | Manager Since | Direct TER (current) |
|---|---|---|---|---|---|---|
| `122639` | Parag Parikh Flexi Cap Fund | Flexi Cap | Rajeev Thakkar | PPFAS Mutual Fund | 2013-05-28 | 0.63% |
| `118955` | HDFC Flexi Cap Fund | Flexi Cap | Roshi Jain | HDFC Mutual Fund | 2022-07-29 | 0.81% |
| `120586` | ICICI Prudential Large Cap Fund | Large Cap | Anish Tawakley | ICICI Prudential AMC | 2018-09-05 | 0.92% |
| `118834` | Mirae Asset Large & Midcap Fund | Large & Mid Cap | Neelesh Surana | Mirae Asset Mutual Fund | 2010-07-09 | 0.61% |
| `119598` | SBI Large Cap Fund | Large Cap | Saurabh Pant | SBI Mutual Fund | 2020-03-30 | 0.88% |
| `120166` | Kotak Flexi Cap Fund | Flexi Cap | Harsha Upadhyaya | Kotak Mahindra AMC | 2012-08-04 | 0.64% |
| `118989` | HDFC Mid-Cap Opportunities Fund | Mid Cap | Chirag Setalvad | HDFC Mutual Fund | 2007-06-25 | 0.74% |
| `120505` | Axis Midcap Fund | Mid Cap | Shreyash Devalkar | Axis Mutual Fund | 2016-11-23 | 0.58% |
| `118668` | Nippon India Growth Mid Cap Fund | Mid Cap | Rupesh Patel | Nippon India Mutual Fund | 2021-04-18 | 0.95% |
| `118778` | Nippon India Small Cap Fund | Small Cap | Samir Rachh | Nippon India Mutual Fund | 2017-01-01 | 0.68% |
| `120828` | Quant Small Cap Fund | Small Cap | Sandeep Tandon | Quant Mutual Fund | 2019-11-01 | 0.77% |
| `119071` | DSP Midcap Fund | Mid Cap | Vinit Sambre | DSP Mutual Fund | 2012-07-01 | 0.79% |

**Verification status.** Scheme codes are verified against `mfapi.in`. Manager names, tenure dates and TERs are curated registry values that are
**not independently machine-verified**: schemes often have co-managers, managers change, and TERs are point-in-time. One correction was made in
review: HDFC Flexi Cap was managed by Prashant Jain until July 2022; Roshi Jain was assigned the scheme on 2022-07-29 (the earlier registry value
2022-01-14 was wrong) and has since left HDFC AMC. For four schemes (Nippon India Growth Mid Cap, HDFC Flexi Cap, SBI Large Cap, Quant Small Cap)
the named manager was in charge for only part of the sample window; `final_evaluation.csv` quantifies this in `tenure_overlap_pct` / `attribution_note`.

### 2.2 Market benchmark series
- **Raw file**: `data/raw/raw_benchmark_nifty500.csv`; **ticker** `^CRSLDX` via Yahoo Finance (`yfinance`); fallback `^NSEI` (NIFTY 50), recorded in
  `acquisition_metadata.json` under `ticker` if it is ever used.
- **What it is**: the NIFTY 500 **price** index, not a Total Return Index: the closes are 9,197.9 (2019-01-01) and 19,429.15 (2023-12-29) - price levels
  (a TRI would be substantially higher by 2023 because dividends are reinvested) - and the `Dividends` column is empty. It excludes dividends, so used as a
  CAPM market it biases every alpha **upward** by roughly the dividend yield. It is therefore kept only as a labelled sensitivity series (`nifty500_price_return`).
  Compounded over the sample it returns 16.6% per year.
- **Optional genuine TRI**: place the historical NIFTY 500 TRI series (from niftyindices.com) at `data/raw/raw_benchmark_nifty500_tri.csv` with a `Date`
  column and a TRI-level column. The pipeline then uses it as the CAPM market (`market_proxy = nifty500_tri`). NSE blocks scripted downloads, so this
  is a manual step.

### 2.3 IIMA risk-free rate and factor library
- **Provider**: Indian Institute of Management Ahmedabad, Indian Fama-French-Momentum data library (Agarwalla, Jacob, Varma).
- **URL**: `https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA/2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv`
- **Raw file**: `data/raw/raw_iima_four_factors_daily.csv`: columns `Date, SMB, HML, WML, MF, RF`, **percent** units, 1993-10-01 to 2025-12-31 (the first
  row has NA for MF/RF). Release notes: returns are holding-period returns; this release uses the total-returns field of the underlying
  database (so they should include dividends - the notes do not say so explicitly for the factor files); the "survivorship-bias adjusted" file is the
  recommended one and corrects for companies that vanished in the mid-1990s (immaterial for 2019-2023).

| Raw column | Internal column | Description |
|---|---|---|
| `MF` | `Mkt-RF` | market excess return (market premium) |
| `SMB` | `SMB` | small-minus-big size factor |
| `HML` | `HML` | high-minus-low value factor |
| `WML` | `Mom` | winners-minus-losers momentum factor |
| `RF` | `RF` | risk-free rate; accrues per **calendar** day (Monday value ~3x a weekday value) |

- **Transformation**: all columns divided by 100 (percent -> decimal). `iima_market_return` = `Mkt-RF` + `RF`.
- **No synthetic fallback**: if the file is missing and cannot be downloaded the pipeline stops with instructions. Earlier versions silently wrote random
  numbers in that case; a constant-RF placeholder file is now detected and rejected.

---

## 3. Data Storage Map

### 3.1 Raw data (`data/raw/`)
| File | Source | Content |
|---|---|---|
| `raw_fund_prices.csv` | AMFI via mfapi.in | daily NAVs, 12 schemes |
| `raw_benchmark_nifty500.csv` | Yahoo Finance (`^CRSLDX`) | NIFTY 500 price index (OHLC, volume) |
| `raw_iima_four_factors_daily.csv` | IIMA | SMB, HML, WML, MF, RF (percent) |
| `raw_benchmark_nifty500_tri.csv` | NSE (manual, optional) | NIFTY 500 TRI level |
| `acquisition_metadata.json` | generated | retrieval timestamps, date ranges, series type |
| `screening_metadata.json` | generated | scheme registry (Section 2.1) |

### 3.2 Processed data (`data/processed/`)
| File | Generating module | Content |
|---|---|---|
| `aligned_analysis_dataset.csv` | `data_loader` | synchronous daily returns, 12 funds (14,586 rows) + factors + market proxies |
| `data_validation_report.json` | `data_loader` | structural validation of every raw input |
| `capm_results.csv/json` | `capm` | CAPM (primary market proxy), OLS and HAC inference |
| `capm_sensitivity_nifty500_price.csv` | `run_pipeline` | CAPM against the dividend-free price index (labelled sensitivity) |
| `bootstrap_results.csv/json` | `bootstrap` | CBB (B=1,000, b=30, seed 42) p-values, CIs, BH q-values |
| `bootstrap_block_sensitivity.csv` | `bootstrap` | p-values for b in {1,5,10,20,30,60} |
| `factor_model_results.csv/json` | `factor_model` | Carhart 4-factor alphas, loadings, HAC alpha p-values |
| `persistence_period_results.csv/json`, `persistence_summary.csv/json` | `persistence` | four-subperiod alphas and persistence metrics |
| `descriptive_statistics.csv` | `descriptive` | buy-and-hold CAGR, volatility, Sharpe, drawdown (full raw series) |
| `benchmark_sensitivity.csv` | `sensitivity` | classification under each market proxy |
| `final_evaluation.csv/json` | `evaluator` | evidence classification, flags, scores, attribution context |
| `run_manifest.json` | `run_pipeline` | how the outputs were produced (proxy, sample, settings, timestamp) |

---

## 4. Common-Date, Common-Interval Alignment

Algorithm (`create_aligned_dataset`):
1. Compute 1-step simple returns per series on its own calendar and record the date each return is measured **from** (`prev_date`).
2. Inner-join fund returns with the benchmark and the IIMA factors on `Date` (no forward- or back-fill, ever).
3. Keep a row only if the fund, benchmark and factor `prev_date` agree (and, if a TRI series is supplied, its `prev_date` too).
4. Drop rows with missing return, risk-free or factor values; compute excess returns (`R_fund - RF`, `R_market - RF`).

**Result**: 14,721 candidate fund-day rows -> 14,586 retained (135, i.e. 0.9%, dropped as asynchronous); 1,216 distinct dates from `2019-01-02`
to `2023-12-29`, with 1,214-1,216 observations per fund (three schemes lack one NAV date that the other nine have).

Calendar mismatches that the rule handles: special Diwali/Saturday sessions present in the IIMA file but not in Yahoo or the AMFI NAVs (2019-10-27, 2020-02-01,
2020-11-14, 2023-11-12); regular trading days missing in Yahoo (2019-02-13, 2019-03-29, 2021-01-01, 2022-12-26); dates present in the Yahoo series for which the schemes
publish no NAV (the 2021-11-04 and 2022-10-24 Diwali Muhurat sessions, 2023-09-28); and single NAV dates missing for individual schemes (2019-07-29, 2023-12-20).
Ignoring them (as version 1.0.0 did) paired two-day fund returns with one-day market returns on the following day and moved individual alphas by up to +/-0.5 points per year.

Other quality observations: a handful of zero-return (stale NAV) days per fund (at most 8 of ~1,215; Axis Midcap 8, ICICI Large Cap 7, HDFC Mid-Cap 3, PPFAS 1);
extreme returns (|r| > 8%) occur only on 2020-03-12, 2020-03-23 and 2020-04-07 and are genuine market moves; no non-positive NAVs, duplicate dates or unparseable dates.

---

## 5. Source Reliability and Failure Behaviour

| Source | Risk | Behaviour |
|---|---|---|
| `mfapi.in` (NAVs) | rate limits, downtime | existing `raw_fund_prices.csv` is reused; a failed scheme download is reported and, if no scheme can be fetched, the pipeline raises |
| Yahoo Finance (`^CRSLDX`) | gaps, renamed tickers | falls back to `^NSEI` and **records the ticker actually used**; the series is only a sensitivity series unless a TRI file is absent *and* the user selects it |
| IIMA | server or URL change | the pipeline **stops** with the download URL and the expected save path; no substitute data is generated |

Raw files are written once and never overwritten by later runs (idempotent acquisition).

---

## 6. Registry Fields (`screening_metadata.json`)

| Field | Description |
|---|---|
| `ticker` | AMFI scheme code (string) |
| `fund_name`, `short_name` | scheme name; dashboard label |
| `category` | SEBI equity category |
| `lead_manager`, `tenure_start` | named manager and the date that manager took charge (curated; see Section 2.1) |
| `amc`, `expense_ratio` | asset manager; Direct Plan TER in percent (point-in-time) |
| `notes` | scheme-specific caveats (e.g. predecessor manager, overseas holdings) |
| `status` | `CANDIDATE` |
| `verification_status` | `SCHEME_CODE_VERIFIED_AGAINST_MFAPI; MANAGER_TENURE_NOT_INDEPENDENTLY_VERIFIED` |
| `metadata_version` | registry version; a stale file is regenerated automatically |
