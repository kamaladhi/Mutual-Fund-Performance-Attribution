# Functional and System Requirements: Indian Fund Manager Skill Detector

*Updated for rules version 2.0.0. Requirement identifiers are unchanged where the intent is unchanged; changed requirements are marked **(revised)**, new ones **(new)**.*

## 1. Overview
Functional, analytical, dataset, dashboard, testing and limitation-disclosure requirements for evaluating Indian active equity mutual funds against a total-return market proxy and
the IIM Ahmedabad (IIMA) Indian multi-factor series.

---

## 2. Dataset Requirements

### FR-DAT-01: Indian Mutual Fund NAV Data
Ingest daily NAV series for 12 Indian active equity funds (Direct Plan - Growth) from AMFI via the `mfapi.in` REST API. AMFI scheme codes are the fund identifiers (strings, never integers).

### FR-DAT-02: Benchmark Series **(revised)**
Ingest the NIFTY 500 index (`^CRSLDX`, Yahoo Finance) and **label it for what it is: a price index that excludes dividends**. It must not be described as a Total Return Index. If a genuine
NIFTY 500 TRI file is supplied (`data/raw/raw_benchmark_nifty500_tri.csv`) it must be used as the CAPM market proxy. Fallback ticker `^NSEI` must be recorded when used.

### FR-DAT-03: IIMA Risk-Free Rate and Multi-Factor Series
Ingest the IIMA daily factor file (`MF`, `SMB`, `HML`, `WML`, `RF`; percent units; survivorship-bias-adjusted release).

### FR-DAT-04: Factor Scaling
Scale all IIMA values from percent to decimal (divide by 100). Rename `MF` -> `Mkt-RF`, `WML` -> `Mom`. Derive `iima_market_return` = `Mkt-RF` + `RF`.

### FR-DAT-05: Date Format Handling
Parse the mfapi.in `DD-MM-YYYY` dates and convert to ISO 8601 before processing.

### FR-DAT-06: Data Validation Pipeline **(revised)**
Validation must run inside the pipeline and fail loudly: required columns, duplicate (Date, symbol) pairs, chronological order per fund, non-positive NAVs, missing-value counts. The report is written to
`data/processed/data_validation_report.json`.

### FR-DAT-07: Strict Common-Date, Common-Interval Alignment **(revised)**
Strict inner join across funds, benchmark and factors; **no forward-fill, backward-fill or imputation**. Every retained return must cover the same one-step interval in the fund, the benchmark and the
factors (previous-date match); rows that do not are dropped and counted. The number of retained observations is a reported result, not a fixed requirement.

### FR-DAT-08: Idempotent Raw Data Acquisition **(revised)**
Raw files are written once and never overwritten by later runs. **No synthetic or substitute data may ever be generated**; a failed required download stops the pipeline with instructions, and a
placeholder factor file (constant risk-free rate) is rejected.

### FR-DAT-09: Screening Metadata Registry **(revised)**
Generate `data/raw/screening_metadata.json` with scheme code, full name, short name, SEBI category, named manager, **date that manager took charge**, AMC, expense ratio, scheme notes,
a truthful `verification_status` (scheme code verified; manager/tenure not independently verified) and a registry version (stale files are regenerated).

---

## 3. Descriptive Statistics Requirements

### FR-STA-01: Return and Risk Metrics **(revised)**
Annualised return (CAGR), annualised volatility, Sharpe ratio and maximum drawdown for all funds and the market series, computed on the full buy-and-hold record (cumulative return must equal
last NAV / first NAV - 1).

### FR-STA-02: Higher Distribution Moments
Residual skewness and excess kurtosis for the CAPM regressions.

### FR-STA-03: OLS Diagnostics
Durbin-Watson and Jarque-Bera for all CAPM and 4-factor regressions.

---

## 4. CAPM Analysis Requirements

### FR-CAP-01: Single-Factor OLS Regression **(revised)**
    R_{i,t} - R_{f,t} = alpha + beta * (R_{m,t} - R_{f,t}) + epsilon
where R_f is the IIMA daily RF and R_m is the primary market proxy (genuine NIFTY 500 TRI if supplied, otherwise the IIMA total-return market).

### FR-CAP-02: Parameter Outputs
Jensen's alpha (daily and annualised), beta, R-squared, adjusted R-squared, alpha standard error, t-statistic, p-value, 95% confidence intervals.

### FR-CAP-03: Hypothesis Test **(revised)**
Two-sided t-test of H0: alpha = 0 under classical OLS (t_{n-2}) **and** with Newey-West HAC standard errors; the HAC p-value is the one used for decisions.

### FR-CAP-04: Alpha Annualization
alpha_annual = ((1 + alpha_daily)^252 - 1) x 100.

### FR-CAP-05: Market-proxy Sensitivity **(new)**
The complete analysis must be re-run under alternative market proxies (IIMA market, NIFTY 500 price index, price index plus an assumed dividend yield, genuine TRI if supplied) and the resulting
classifications reported (`benchmark_sensitivity.csv`).

---

## 5. Circular Block Bootstrap Requirements

### FR-BOO-01: Circular Block Bootstrap Implementation
Circular block resampling of paired (fund excess return, market excess return) observations (Politis & Romano, 1992) to preserve temporal dependence.

### FR-BOO-02: Block Length **(revised)**
Default block length 30 trading days, parameterisable. The rationale must not claim equality with N^{1/3} (about 10.7 for this sample); a block-length sensitivity analysis is required.

### FR-BOO-03: Bootstrap Iterations
At least B = 1,000 replications per fund, fixed seed (42) for reproducibility.

### FR-BOO-04: Null-Centred Hypothesis Test **(revised)**
alpha_null = alpha_boot - alpha_observed (imposes H0: alpha = 0). Report the smoothed two-sided p-value (#{|alpha_null| >= |alpha_obs|} + 1)/(B + 1) and the one-sided (H1: alpha > 0) p-value.
A fund is flagged significantly **positive** only if the two-sided p < 0.05 **and** alpha_obs > 0.

### FR-BOO-05: Confidence Intervals
95% percentile intervals (2.5th / 97.5th percentiles) of the annualised bootstrap alpha.

### FR-BOO-06: Multiple Testing **(new)**
Report Benjamini-Hochberg FDR-adjusted p-values across the fund universe.

---

## 6. Carhart 4-Factor (IIMA) Requirements

### FR-APT-01: 4-Factor OLS Regression
R_{i,t} - R_{f,t} = alpha + b_mkt Mkt-RF + b_smb SMB + b_hml HML + b_mom Mom (design matrix [1, Mkt-RF, SMB, HML, Mom], df = n - 5).

### FR-APT-02: Factor Loading Outputs
Per factor: loading, standard error, t-statistic, p-value, 95% CI. Also 4-factor annualised alpha with HAC inference, R-squared, adjusted R-squared.

### FR-APT-03: Alpha Attribution **(revised)**
alpha_reduction_pct = (alpha_CAPM - alpha_4F) / |alpha_CAPM| x 100 **and** alpha_reduction_pts_annual = annual alpha_CAPM - annual alpha_4F. CAPM and 4-factor alphas must be measured
against the same market series.

### FR-APT-04: IIMA Momentum Factor
Use the IIMA `WML` column, renamed `Mom`, as the momentum factor.

---

## 7. Performance Persistence Requirements

### FR-PER-01: Chronological Subperiod Partitioning
Four non-overlapping chronological subperiods of (nearly) equal length; no shuffling.

### FR-PER-02: Subperiod CAPM Regressions
CAPM within each subperiod; at least 3 observations per subperiod (ValueError otherwise); significance by HAC p-value.

### FR-PER-03: Persistence Metrics
positive_alpha_persistence_rate, significant_positive_alpha_persistence_rate, alpha mean / median / std / min / max (annualised).

### FR-PER-04: Consistency Indicators **(revised)**
alpha_positive_all_periods (4 of 4), alpha_positive_majority (**strictly more than half**, i.e. at least 3 of 4), significant_positive_majority (strict majority, HAC).

---

## 8. Final Skill Evidence Evaluation Requirements

### FR-EVA-01: Deterministic Rule-Based Classification
Transparent, deterministic, versioned (**v2.0.0**) evaluation; no machine learning; identical inputs always give identical outputs.

### FR-EVA-02: Four Classification Categories **(revised)**
Mutually exclusive, in precedence order: (1) Likely Style/Luck Driven (positive, at-least-marginally-significant CAPM alpha with >= 75% reduction and no surviving factor alpha), (2) Strong Evidence of Skill (all five
core criteria), (3) Moderate Evidence, (4) Weak/Inconclusive. Full rules: `docs/METHODOLOGY.md` section 7.

### FR-EVA-03: 0-100 Evidence Score **(revised)**
Performance (0-20) + Significance (0-25; zero for non-positive alpha) + Factor Model (0-25) + Persistence (0-20) + Style/Luck Adjustment (-10 to +10).

### FR-EVA-04: Input Validation
Validate all four input CSVs: existence, required columns, unique symbols, numeric range checks (p-values and rates in [0, 1]).

### FR-EVA-05: Output Files **(revised)**
`data/processed/final_evaluation.csv` and `.json`; every row tagged with the evaluation version; manager-attribution columns (`tenure_overlap_pct`, `attribution_note`).

### FR-EVA-06: Direction Awareness **(new)**
A significantly negative alpha must earn no significance, performance or skill credit and must never be labelled style/luck-driven outperformance.

---

## 9. Streamlit Dashboard Requirements

### FR-DSH-01: Interactive Scheme Selector
Sidebar selector `"[scheme_code] | [short_name] [[named_manager]]"`.

### FR-DSH-02: Institutional Metadata Display **(revised)**
Per-fund report: full name, named manager, AMC, SEBI category, date the manager took charge, Direct TER, and an **attribution caveat** where the manager covers less than 90% of the sample.

### FR-DSH-03: Executive Summary KPIs
Universe size and the count of funds in each of the four categories, plus a one-line robustness statement across market proxies.

### FR-DSH-04: Econometric Analysis Tabs **(revised)**
CAPM (OLS and HAC columns, return/risk profile), block bootstrap (two-sided, one-sided, BH q-value, CI, block-length sensitivity), Carhart 4-factor (loadings with confidence intervals, alpha bridge),
subperiod persistence, universe leaderboard (with the market-proxy robustness table). **Every data table must render** (no CSS that blanks `st.dataframe`).

### FR-DSH-05: Sidebar Benchmark Information **(revised)**
Sample period, observation counts, market proxy, factor source, risk-free source and inference settings - all derived from the data / run manifest, none hard-coded; no claim that the benchmark is a TRI unless it is.

### FR-DSH-06: Performance and Caching
`st.cache_data` on `load_dashboard_data()`; symbol columns cast to strings before any join or lookup.

---

## 10. Testing Requirements

### FR-TST-01 ... FR-TST-07
Unit tests for each module (data loader, CAPM, bootstrap, factor model, persistence, evaluator, dashboard) as in version 1.0.0, **plus**: HAC against brute force; no-synthetic-fallback; synchronous alignment;
direction-aware bootstrap flags; strict majority; sign-aware scoring; style/luck eligibility; monotone Moderate rule; dashboard CSS regression; headless render of every fund.

### FR-TST-08: Test Execution **(revised)**
The entire suite must pass with `python -m pytest` from the project root (no fixed test count is mandated).

### FR-TST-09: Independent Verification **(new)**
Integrity tests must re-derive the shipped results independently of the pipeline code: every aligned return covers one common interval (recomputed from the raw files), stored CAPM / 4-factor results equal
`numpy.linalg.lstsq` re-estimates, stored HAC p-values equal a brute-force Newey-West implementation, `final_evaluation.csv` equals a fresh re-evaluation of its inputs, and the README result table equals `final_evaluation.csv`.

### FR-TST-10: End-to-End Reproducibility **(new)**
Running `run_pipeline.py` twice from the cached raw data must produce identical `final_evaluation.csv`.

---

## 11. Documented Limitation Disclosure Requirements

### FR-LIM-01: Sample Size Disclosure
Communicate that only 12 hand-picked Indian funds over ~5 years are evaluated; results must not be generalised.

### FR-LIM-02: Statistical Significance Caveat
Classifications must carry the caveat that significance does not imply causality or proof of manager skill.

### FR-LIM-03: Factor Model Completeness Caveat
The four-factor model may omit Indian-market risk factors (liquidity, quality, macro) and cannot capture overseas holdings or large cash/arbitrage books.

### FR-LIM-04: Verification Status Transparency
State what is and is not verified (scheme codes verified; manager names and tenure dates not independently verified).

### FR-LIM-05: Benchmark Dependence **(new)**
State that conclusions depend on the market proxy and show them under each proxy.

### FR-LIM-06: Selection and Multiple Testing **(new)**
State that the universe is hand-picked (survivorship / selection bias) and that 12 tests at 5% carry no multiplicity correction in the rules.
