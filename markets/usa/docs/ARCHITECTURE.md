# System Architecture & Design: Fund Manager Skill Detector

> **Note.** The module layout below is current. Where this document mentions `EVALUATION_VERSION = "1.0.0"` or the original four categories, read
> `METHODOLOGY.md`: the evaluator is at v2.3.0 with six classes and shares its rule code with the Indian market.

## 1. System Overview
The **Fund Manager Skill Detector** system is engineered as a modular, decoupled Python library paired with an interactive presentation layer (Streamlit). High architectural priority is placed on separation of concerns: data ingestion, analytical engines, evaluation rules, and visualization logic are strictly decoupled.

---

## 2. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------+
|                            USER INTERFACE                             |
|                    Streamlit Application (app.py)                     |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                         PRESENTATION / VISUALS                        |
|                    src/visualization.py (Plotly/Matplotlib)           |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                          INTEGRATION ENGINE                           |
|                    src/evaluator.py (Skill Scoring)                   |
+-----------------------------------------------------------------------+
       |                           |                           |
       v                           v                           v
+---------------+          +---------------+          +---------------+
|  CAPM Engine  |          | Bootstrap Engine|        |  APT Engine   |
| (src/capm.py) |          |(src/bootstrap.py)|       |(src/factor_model)|
+---------------+          +---------------+          +---------------+
       |                           |                           |
       +---------------------------+---------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                     PERSISTENCE ANALYZER ENGINE                       |
|                 src/persistence.py (Sub-period rank/Chi2)             |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                     DATA PREPROCESSING & VALIDATION                   |
|              src/data_loader.py (Ingestion, Validation, Align)        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                             DATA STORAGE                              |
|          data/raw/  |  data/processed/  |  results/tables/ & figures/ |
+-----------------------------------------------------------------------+
```

---

## 3. Module Breakdown & Responsibilities

### 3.1 `src/data_loader.py` (Implemented in Task 3)
- Ingest raw CSV/API data for mutual funds, benchmark indices, Treasury Bills, and Fama-French factors.

### 3.2 `src/capm.py` (Implemented in Task 4)
- Fits OLS single-factor CAPM regression and outputs `CAPMResult` objects.

### 3.3 `src/bootstrap.py` (Implemented in Task 5)
- Circular Block Bootstrap resampling (default $b=30, B=1000, seed=42$) and null-centered empirical p-value calculation.

### 3.4 `src/factor_model.py` (Implemented in Task 6)
- Fits Carhart 4-Factor OLS linear regressions and calculates alpha reduction metrics.

### 3.5 `src/persistence.py` (Implemented in Task 7)
- **Responsibilities**:
  - Implements `get_chronological_period_splits()`, `run_period_capm_analysis()`, `calculate_persistence_metrics()`, and `save_persistence_results()`.
  - Partitions trading history into 4 non-overlapping chronological subperiods (`P1`, `P2`, `P3`, `P4`).
  - Fits OLS CAPM regressions within each subperiod (`PeriodCAPMResult`).
  - Computes `positive_alpha_persistence_rate`, `significant_positive_alpha_persistence_rate`, subperiod mean/median/std annual alpha, and sign consistency indicators (`PersistenceSummaryResult`).
  - Exports outputs to `persistence_period_results.csv / json` and `persistence_summary.csv / json`.

### 3.6 `src/evaluator.py` (Implemented in Task 8)
- **Responsibilities**:
  - Loads four processed result CSVs (CAPM, Bootstrap, 4-Factor, Persistence) read-only.
  - Validates file existence, required columns, unique symbols, join integrity, numeric ranges, p-value / rate bounds.
  - Computes 11 Boolean evidence flags per fund (documented in `METHODOLOGY.md §6.4`).
  - Computes a 5-component transparent evidence score (0–100) per fund (documented in `METHODOLOGY.md §6.5`).
  - Classifies each fund into one of five categories using a documented precedence rule system.
  - Generates deterministic rule-based `final_explanation` strings.
  - Exports `data/processed/final_evaluation.csv` and `data/processed/final_evaluation.json`.
  - Tags every output row with `EVALUATION_VERSION = "1.0.0"` for traceability.
- **Design constraints**: No ML; no modification of source files; identical inputs produce identical outputs.

### 3.7 `src/visualization.py`
- Reusable plotting utilities for Plotly and Matplotlib charts.

### 3.8 `app.py` (Implemented in Task 9)
- Interactive Streamlit dashboard presenting all analytical and evaluation results from Tasks 1–8.
- Features: Executive Summary KPIs, Active-Fund Verification Warnings, Fund Selector, CAPM Single-Factor Analysis, Circular Block Bootstrap CI Visualizations, Carhart 4-Factor Risk Adjustment Charts, Subperiod Performance Persistence Charts, Transparent 0–100 Evidence Score Breakdown, Cross-Fund Comparison Tables & Charts, and Documented Methodology & Risk Limitations.

---

## 4. Data Flow (Full Pipeline)

```
raw/raw_fund_prices.csv
raw/raw_benchmark_sp500.csv
raw/raw_fama_french_daily.csv
raw/raw_momentum_daily.csv
          |
          v  [data_loader.py]
processed/aligned_analysis_dataset.csv
processed/cleaned_fund_returns.csv
processed/cleaned_factors.csv
          |
    +-----+------+
    |            |
    v            v
[capm.py]   [bootstrap.py]
    |            |
    v            v
capm_results.csv  bootstrap_results.csv
    |                      |
    +------+--------+------+
           |        |
           v        v
   [factor_model.py] [persistence.py]
           |                |
           v                v
factor_model_results.csv  persistence_summary.csv
           |                |
           +-------+--------+
                   |
                   v  [evaluator.py]
           final_evaluation.csv
           final_evaluation.json
                   |
                   v  [app.py]
           Streamlit Interactive Dashboard (Task 9 Complete)
```

