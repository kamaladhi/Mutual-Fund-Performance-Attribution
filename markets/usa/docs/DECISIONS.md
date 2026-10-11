# Architectural & Methodological Decision Log (ADR)

## Overview
This document maintains a chronological record of key technical, architectural, and methodological decisions made during the lifecycle of the **Fund Manager Skill Detector** project.

---

## ADR-001: Modular Package Architecture & Strict Decoupling
## ADR-002: Circular Block Bootstrap (CBB) for Non-Parametric Significance Testing
## ADR-003: Inclusion of Multi-Factor APT (Fama-French 3 & Carhart 4-Factor Models)
## ADR-004: Multi-Criterion Manager Skill Classification Rules
## ADR-005: Target Fund Universe Selection & Primary Data Frequency
## ADR-006: Active Management Screening & Evidence Requirement
## ADR-007: Strict Raw Data Immutability Policy
## ADR-008: Primary Risk-Free Rate Selection
## ADR-009: Mutual Fund Total Return via Adjusted Close
## ADR-010: Replacement of Allocation Funds for Benchmark Equity Comparability
## ADR-011: Active Share Fallback Rule & Candidate vs Verified Status
## ADR-012: Deterministic Data Acquisition & Preprocessing Pipeline
## ADR-013: Single-Factor CAPM Regression & Dataclass Result Serialization
## ADR-014: Compounded Annualized Alpha Metric
## ADR-015: Circular Block Bootstrap (CBB) & Null-Centered Hypothesis Testing
## ADR-016: Carhart 4-Factor OLS Engine & Safe Alpha Reduction Metrics

---

## ADR-017 (Task 7 Implementation): Chronological Subperiod Partitioning & Alpha Persistence Metrics

### Decision
Implement performance persistence analysis in [`src/persistence.py`](../src/persistence.py) by partitioning trading history into 4 non-overlapping chronological subperiods (`P1`, `P2`, `P3`, `P4`) of ~314 trading days each, running period-level OLS CAPM regressions, and calculating alpha persistence rates and volatility stability metrics ($s_{\alpha}$).

### Rationale
- Chronological partitioning preserves real-world time-series order and avoids look-ahead or data leakage.
- Subperiod CAPM regressions isolate whether a fund's outperformance is consistently sustained across independent time windows or driven by a single fortunate subperiod.

---

## ADR-018 (Task 8 Implementation): Integrated Evaluator Architecture

### Decision
Implement the evidence-synthesis layer in [`src/evaluator.py`](../src/evaluator.py) as a deterministic, rule-based, no-ML engine that reads four processed CSV files, computes eleven Boolean evidence flags, a five-component score (0–100), and assigns one of four ordered categories using a documented precedence system.

### Rationale

**Why deterministic, rule-based classification?**
A machine-learning classifier would require a labelled dataset of funds with known skill outcomes, which does not exist in a scientifically validated form. Deterministic Boolean rules are:
- Fully auditable by a financial examiner.
- Guaranteed reproducible across platforms.
- Explainable in regulatory or academic contexts.
- Not subject to overfitting to the 12-fund training set.

**Why a 0–100 score in addition to four categories?**
The evidence categories provide the primary classification. The continuous score provides a within-category ranking and allows future thresholds to be recalibrated without changing the category structure. The score does NOT determine the category — the Boolean rule system does.

**Why the specific precedence order?**
The `Likely Style/Luck Driven` category is evaluated first because it is a *disqualifying* condition: if the factor model entirely explains the apparent outperformance, no amount of raw alpha or bootstrap significance should produce a Strong classification. This prevents the scoring system from rewarding high-alpha funds that do not demonstrate independent alpha after factor adjustment.

**Why 75% reduction threshold for Style/Luck?**
75% was chosen as a conservative threshold: a fund retaining 25% of its CAPM alpha after full four-factor adjustment is given the benefit of the doubt for moderate or stronger classification. Below 25% retained alpha, the residual is too small to distinguish from noise in a 5-year window.

**Why column-mapping dictionaries?**
Upstream modules (Tasks 4–7) may rename columns in future versions. Isolating the column mapping in module-level dictionaries (`CAPM_COL_MAP`, `BOOTSTRAP_COL_MAP`, `FACTOR_COL_MAP`, `PERSISTENCE_COL_MAP`) means future column renames require only a one-line change in the mapping dictionary, not a refactor of evaluation logic.

**Why `EVALUATION_VERSION`?**
Version tagging allows downstream users (including the Streamlit dashboard) to distinguish results computed under different scoring logic. If the scoring formula changes, the version is bumped and old result files remain identifiable.

### Limitations
- Category rules encode normative judgements about which combinations of evidence constitute "strong" vs "moderate" evidence. Different analysts might choose different threshold combinations.
- The 5-component scoring weights are not empirically calibrated; they reflect expert judgement about the relative importance of each evidence dimension.
- The evaluator does not perform statistical meta-analysis (e.g., combining p-values via Fisher's method). This is intentional: meta-p-values would create a false impression of analytical rigour beyond what the underlying data supports.

---

## ADR-019 (Task 9 Implementation): Streamlit Diagnostic Dashboard Architecture

### Decision
Implement the interactive presentation layer in [`app.py`](../app.py) using Streamlit and Plotly. The application is strictly designed as a read-only diagnostic and evidence-synthesis presentation tool.

### Rationale

**1. Strict Consumption of Pre-calculated Artifacts**:
- The dashboard does **not** run OLS regressions, block bootstrap resampling, subperiod partitioning, or skill score recalculations at runtime.
- It consumes `data/processed/final_evaluation.csv / json`, `capm_results.csv`, `bootstrap_results.csv`, `factor_model_results.csv`, `persistence_summary.csv`, and `persistence_period_results.csv` directly.
- This ensures zero latency, zero non-deterministic recomputations, and strict immutability of underlying analytical engines.

**2. Prominent Active-Management Verification Warning**:
- To enforce research integrity and prevent misinterpretation, a prominent alert box displays the `UNVERIFIED_CANDIDATE` screening status across candidate funds.
- Clearly states that output represents statistical evidence within the analyzed sample and not investment advice or proof of skill.

**3. Multi-Tab Deep-Dive Structure**:
- Organizes complex quantitative findings into 5 focused tabs per fund (CAPM, Circular Bootstrap CI, Carhart 4-Factor, Subperiod Persistence, Evidence Score Breakdown).
- Uses interactive Plotly visualizations (horizontal confidence intervals, CAPM vs 4-Factor comparisons, subperiod alpha bars, score component bars) for enhanced user engagement while maintaining technical rigor.

**4. Unit Testing Framework (`tests/test_app.py`)**:
- Created a headless unit test suite validating importability, file detection, schema requirements, dynamic fund loading, category aggregation, score bounds, read-only file immutability, and missing file/column error handling without launching a browser.



**Why add a Strong Potential Candidate category in evaluation v1.3.0?**

The original Strong Evidence category intentionally requires agreement across CAPM significance, bootstrap significance, significant positive four-factor alpha, and persistence. A fund can show a strong positive residual alpha and majority-period persistence while failing one or more corroboration tests. Rather than either overstating this as confirmed skill or hiding the useful signal under Moderate Evidence, the new candidate category makes the distinction explicit. Its rule is applied uniformly to all funds, and the UI explains that it is preliminary rather than proof.

---

## ADR-020: One rule set for both markets (evaluation v2.3.0)

### Decision
The US evaluator (`src/evaluator.py`) and persistence module (`src/persistence.py`) adopt, verbatim, the rule code of the Indian project, and a
repository-level test (`tests/test_rule_parity.py`) compares the two so that they cannot drift apart again.

### Rationale
- The combined dashboard compares the markets' classes. That is only meaningful if the same rules produce them. The US rules (v1.5.0) still counted a
  2-of-4 persistence tie as a majority, gave "Moderate Evidence" for a positive CAPM alpha with no statistical support (p up to 0.94), made a
  four-factor alpha alone a "candidate", and scored a significantly negative alpha as evidence; the Indian review had already corrected all four.
- The Indian rules were the ones that had been independently reviewed (statsmodels cross-check, brute-force HAC, rule re-implementation).

### Consequences
- Raw statistics are untouched (CAPM, bootstrap, four-factor and persistence estimates are bit-for-bit the same); only classes and scores moved:
  FDGRX Candidate -> Moderate; FCNTX and MIGFX Moderate -> Positive but Inconclusive; FLPSX and PAGRX Moderate -> Weak.
- The class distribution was **not** a criterion for this decision and is not targeted; it is whatever the fixed rules produce.
- The US bootstrap and factor summaries pre-date a few columns the shared rules read; the evaluator derives them from the saved results exactly as
  the Indian engine defines them (`_upgrade_bootstrap_schema`, `_upgrade_factor_schema`). Nothing is re-estimated.
