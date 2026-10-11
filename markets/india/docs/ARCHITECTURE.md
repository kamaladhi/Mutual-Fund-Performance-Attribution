# System Architecture and Design: Indian Fund Manager Skill Detector

## 1. System Overview
A modular, decoupled Python library plus a Streamlit presentation layer. Data ingestion, the analytical engines, the evidence-evaluation rules
and the visualisation are separate concerns: each engine reads files produced by the stage before it and writes its own results, so any stage can
be re-run, audited or replaced independently. `run_pipeline.py` runs every stage in order.

---

## 2. High-Level Architecture

```
+-----------------------------------------------------------------------+
|                            USER INTERFACE                             |
|     app.py (Streamlit)  |  results/tables + results/figures + README  |
+-----------------------------------------------------------------------+
                                   ^
+-----------------------------------------------------------------------+
|                  REPORTING & ROBUSTNESS (read-only)                   |
|   src/reporting.py  |  src/sensitivity.py  |  src/descriptive.py      |
+-----------------------------------------------------------------------+
                                   ^
+-----------------------------------------------------------------------+
|                          INTEGRATION ENGINE                           |
|          src/evaluator.py (rule-based evidence synthesis v2.0.0)      |
+-----------------------------------------------------------------------+
        ^                 ^                  ^                 ^
+-------------+  +----------------+  +----------------+  +----------------+
| CAPM engine |  | Bootstrap eng. |  | 4-factor eng.  |  | Persistence    |
| src/capm.py |  | src/bootstrap  |  | src/factor_    |  | src/persistence|
| (OLS + HAC) |  | (CBB, BH, b-   |  | model.py       |  | (4 subperiods) |
|             |  |  sensitivity)  |  | (OLS + HAC)    |  |                |
+-------------+  +----------------+  +----------------+  +----------------+
        ^                 ^                  ^                 ^
        +-----------------+---------+--------+-----------------+
                                    |   shared: src/stats_utils.py (Newey-West HAC, BH FDR)
+-----------------------------------------------------------------------+
|                    DATA PREPROCESSING AND VALIDATION                  |
|   src/data_loader.py (ingestion, validation, synchronous alignment)   |
+-----------------------------------------------------------------------+
                                   ^
+-----------------------------------------------------------------------+
|                             DATA SOURCES                              |
|  AMFI via mfapi.in (NAVs) | Yahoo ^CRSLDX (NIFTY 500 PRICE index)     |
|  IIM Ahmedabad factor library | optional NSE NIFTY 500 TRI file       |
+-----------------------------------------------------------------------+
```

---

## 3. Module Breakdown

### 3.1 `src/data_loader.py`
Acquires NAVs (mfapi.in), the NIFTY 500 price index (Yahoo) and the IIMA factor file; **never fabricates data** (a failed factor download raises; a constant-RF
placeholder file is rejected). `validate_dataframe()` is wired into the pipeline and its report is written to `data_validation_report.json`.
`preprocess_price_series()` computes returns and records each return's start date (`prev_date`); `preprocess_iima_factors()` scales percent to decimals and adds
the IIMA total market return; `create_aligned_dataset()` performs the strict inner join **with the synchronous-interval rule** and resolves the CAPM market proxy
(`auto` = genuine TRI if supplied, else the IIMA market). `initialize_screening_metadata()` regenerates the scheme registry whenever its version changes.

### 3.2 `src/stats_utils.py`
`newey_west_cov()` (Bartlett-kernel HAC covariance with the Newey-West plug-in bandwidth), `default_hac_lags()`, `two_sided_t_pvalue()` (survival-function based, no cancellation),
`benjamini_hochberg()` (FDR q-values). Verified against `statsmodels` and brute-force references in the tests.

### 3.3 `src/capm.py`
`fit_capm_ols()` (normal-equation OLS; classical and HAC standard errors, t-statistics, p-values, parametric CIs, compounded annual alpha, Durbin-Watson, Jarque-Bera),
`run_capm_analysis()`, `save_capm_results()`. Output dataclass `CAPMResult`.

### 3.4 `src/bootstrap.py`
`generate_circular_block_indices()`, `run_circular_block_bootstrap()` (null-centred, paired; two-sided and one-sided p-values; direction-aware significance flags),
`run_bootstrap_analysis()` (adds Benjamini-Hochberg q-values), `run_block_length_sensitivity()`. Output dataclass `BootstrapResult`.

### 3.5 `src/factor_model.py`
`fit_four_factor_ols()` (Carhart model on IIMA factors; HAC alpha inference; alpha reduction in percent and in annual points), `run_four_factor_analysis()`.

### 3.6 `src/persistence.py`
`get_chronological_period_splits()` (four non-overlapping chronological periods), `run_period_capm_analysis()` (CAPM with HAC inside each period),
`calculate_persistence_metrics()` (strict-majority flags).

### 3.7 `src/evaluator.py` (version 2.0.0)
`load_and_validate_inputs()`, `compute_evidence_flags()`, `compute_scores()`, `classify_fund()`, `generate_explanation()`, `evaluate_fund()`, `run_evaluation()`, plus
`load_attribution_context()` (named-manager tenure overlap). Deterministic, versioned, no ML; direction-aware (only positive alpha earns skill credit).

### 3.8 `src/descriptive.py`, `src/sensitivity.py`, `src/reporting.py`
`descriptive`: buy-and-hold CAGR / volatility / Sharpe / drawdown from the full raw series. `sensitivity`: re-runs CAPM -> bootstrap -> 4-factor -> persistence -> evaluator
under each market proxy in a temporary directory. `reporting`: `results/tables`, `results/figures` and the auto-generated result blocks of `README.md`.

### 3.9 `app.py`
Streamlit dashboard. Reads only files from `data/` (never recomputes statistics). Sidebar and header facts (sample, observations, market proxy, bootstrap settings) are derived
from the data and `run_manifest.json`, not hard-coded. Five tabs: CAPM, block bootstrap (with block-length sensitivity), Carhart 4-factor (with confidence intervals),
subperiod persistence, universe leaderboard (with the market-proxy robustness table). Scheme-level attribution caveats are shown above the classification.
`load_dashboard_data()` is cached (`st.cache_data`, 5-minute TTL).
CSS note: no blanket background rule may be applied to `div`/`span` - it blanks every `st.dataframe` canvas (guarded by a regression test).

### 3.10 `run_pipeline.py`
Executes data -> CAPM (+ price-index sensitivity) -> bootstrap (+ block sensitivity) -> 4-factor -> persistence -> descriptive -> evaluation -> benchmark sensitivity ->
tables/figures/README -> run manifest. About one minute offline.

---

## 4. Data Flow

```
data/raw/raw_fund_prices.csv ........ AMFI NAVs
data/raw/raw_benchmark_nifty500.csv . NIFTY 500 price index (sensitivity series)
data/raw/raw_iima_four_factors_daily.csv  IIMA factors, RF, market
[data/raw/raw_benchmark_nifty500_tri.csv] optional genuine TRI
data/raw/screening_metadata.json .... scheme registry
          |  src/data_loader.py  (validate, align synchronously, choose market proxy)
          v
data/processed/aligned_analysis_dataset.csv
     |            |             |                |
     v            v             v                v
 capm.py     bootstrap.py  factor_model.py  persistence.py        descriptive.py
     |            |             |                |                (raw series)
capm_results  bootstrap_*   factor_model_*   persistence_*        descriptive_statistics
     \______________\____________|________________/
                                  v   src/evaluator.py (+ screening_metadata, period dates)
                         final_evaluation.csv / .json
                                  |
        sensitivity.py (whole stack, other proxies)  ->  benchmark_sensitivity.csv
                                  |
          reporting.py -> results/tables, results/figures, README blocks   |   app.py -> dashboard
```

---

## 5. Directory Structure
See `README.md` ("Project structure") for the full annotated tree; it is the single maintained copy.

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| CAPM and 4-factor share the IIMA market | alpha reduction then isolates size/value/momentum tilts; IIMA market is a total-return series; no price-vs-TRI error |
| NIFTY 500 price index kept as a *labelled sensitivity* | Yahoo has no TRI series; using the price index as the market would bias every alpha upward by the dividend yield |
| Optional genuine-TRI file | lets the user restore the originally intended benchmark without code changes |
| IIMA factors vs another source | the IIMA library is the standard academic Indian Fama-French-Carhart source |
| AMFI/mfapi.in for NAVs | Yahoo Finance does not carry Indian mutual-fund NAV series |
| Direct Plan - Growth only | no distributor commission; dividends reinvested, so NAV is a total-return series (a design choice, not a SEBI mandate) |
| Synchronous-interval alignment | prevents regressing multi-day fund returns on one-day market returns; no imputation |
| Newey-West HAC for decisions | residuals are autocorrelated and heteroskedastic; classical p-values are over-confident |
| Bootstrap block length 30 + sensitivity table | block length is a judgement; sensitivity shows which conclusions depend on it |
| Direction-aware evidence | a significantly negative alpha is evidence of under-performance, never of skill |
| Strict-majority persistence | 2 of 4 is a coin-flip tie |
| Style/luck label gated on a noteworthy CAPM alpha | the percentage alpha reduction is ill-conditioned when |CAPM alpha| is near zero |
| No synthetic fallbacks | fabricated factor data silently invalidates every result |
| `EVALUATION_VERSION` tag on every output row | traceability across rule changes (1.0.0 -> 2.0.0) |
| README result tables generated from the outputs | documentation cannot drift from results (guarded by a test) |
| Idempotent raw acquisition | no unnecessary API calls; network failures cannot corrupt cached data |
