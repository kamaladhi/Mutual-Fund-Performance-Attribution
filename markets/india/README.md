# Indian Fund Manager Skill Detector: CAPM & Carhart 4-Factor Evaluation with HAC and Circular Block Bootstrap Inference

## Overview
A quantitative framework that asks whether the outperformance of 12 flagship Indian active equity mutual
funds (all **Direct Plan - Growth** schemes, 2019-2023) can be distinguished from luck, market exposure
($\beta$) and systematic factor tilts (size, value, momentum). Evidence is combined from

1. **CAPM** Jensen's alpha (OLS point estimate, Newey-West **HAC** inference),
2. a **circular block bootstrap** of the alpha (null-centred, paired resampling),
3. the **Carhart 4-factor** model on the IIM Ahmedabad (IIMA) Indian factor library (Agarwalla, Jacob, Varma),
4. **subperiod persistence** (four chronological windows),

into a transparent, deterministic, rule-based **evidence classification and 0-100 evidence index**
(`src/evaluator.py`, rules version **2.3.0**, shared with the US market).

> **Read this first.** The classification measures the *strength of statistical evidence* under one specification
> - it is not proof of manager skill and not investment advice. Twelve hand-picked, well-known schemes over a
> ~5-year window cannot support general conclusions; conclusions **depend on the market proxy** (see the
> robustness table below); the named lead manager did not run every scheme for the whole window; and
> multiple testing is handled by Benjamini-Hochberg q-values, which are reported for every fund and gate only
> the strict "Strong Evidence of Skill" tier.

---

## Quick start

```bash
pip install -r requirements.txt

python run_pipeline.py          # reproduces EVERY output in ~1 minute (raw data is cached, no network needed)
streamlit run app.py            # interactive dashboard
python -m pytest                # full test suite
```

The pipeline steps can also be run one at a time (each needs the previous outputs):

```bash
python -m src.data_loader       # 1. align NAVs, benchmark and IIMA factors  -> aligned_analysis_dataset.csv
python -m src.capm              # 2. single-factor CAPM (OLS + HAC)
python -m src.bootstrap         # 3. circular block bootstrap (+ block-length sensitivity)
python -m src.factor_model      # 4. Carhart 4-factor model
python -m src.persistence       # 5. four-subperiod persistence
python -m src.descriptive       # 6. return / risk statistics
python -m src.evaluator .       # 7. integrated evidence classification
python -m src.sensitivity       # 8. whole stack re-run under alternative market proxies
```

`run_pipeline.py` additionally writes the price-index CAPM sensitivity, the run manifest, `results/tables`,
`results/figures` and refreshes the result tables below.

Raw data under `data/raw/` is never re-downloaded when it already exists. There is **no synthetic fallback**: if a
required download fails the pipeline stops with instructions (an earlier version silently substituted random
numbers for the factor file).

---

## Project structure
```
indian_fund_skill_detector/
├── run_pipeline.py                  # one-command, end-to-end reproduction of every output
├── app.py                           # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── raw_fund_prices.csv                  # AMFI NAVs via mfapi.in (12 Direct-Growth schemes)
│   │   ├── raw_benchmark_nifty500.csv           # NIFTY 500 PRICE index (^CRSLDX; excludes dividends)
│   │   ├── raw_iima_four_factors_daily.csv      # IIMA SMB/HML/WML/MF/RF, 1993-2025, percent units
│   │   ├── raw_benchmark_nifty500_tri.csv       # OPTIONAL: genuine NIFTY 500 TRI (not shipped; see below)
│   │   ├── acquisition_metadata.json
│   │   └── screening_metadata.json              # scheme registry: category, named manager, tenure, notes
│   └── processed/
│       ├── aligned_analysis_dataset.csv         # synchronous daily returns, 12 funds
│       ├── data_validation_report.json, run_manifest.json
│       ├── capm_results.csv/json                # CAPM (primary market proxy)
│       ├── capm_sensitivity_nifty500_price.csv  # CAPM vs the dividend-free price index (labelled sensitivity)
│       ├── bootstrap_results.csv/json, bootstrap_block_sensitivity.csv
│       ├── factor_model_results.csv/json
│       ├── persistence_period_results.csv/json, persistence_summary.csv/json
│       ├── descriptive_statistics.csv
│       ├── benchmark_sensitivity.csv            # classification under alternative market proxies
│       └── final_evaluation.csv/json            # rule-based evidence classification
├── results/
│   ├── tables/                      # skill_leaderboard.csv, alpha_comparison.csv
│   └── figures/                     # evidence_index, alpha_comparison, subperiod_alpha, block_length_sensitivity (PNG)
├── docs/
│   ├── METHODOLOGY.md  DATA_SOURCES.md  ARCHITECTURE.md  REQUIREMENTS.md
│   └── REVIEW_REPORT.md             # independent audit: what was verified, what was wrong, what changed
├── src/
│   ├── data_loader.py               # acquisition, validation, synchronous-interval alignment
│   ├── stats_utils.py               # Newey-West HAC covariance, Benjamini-Hochberg FDR
│   ├── capm.py  bootstrap.py  factor_model.py  persistence.py   # the four analytical engines
│   ├── descriptive.py               # annualised return / volatility / Sharpe / drawdown
│   ├── evaluator.py                 # rule-based evidence synthesis (v2.3.0)
│   ├── sensitivity.py               # market-proxy robustness analysis
│   └── reporting.py                 # tables, figures, README result blocks
└── tests/                           # pytest suite (unit, integrity, end-to-end, dashboard)
```

---

## Results

Generated from `data/processed/final_evaluation.csv` by `run_pipeline.py` (do not edit by hand). Primary
specification: CAPM and 4-factor alphas are measured against the **same** IIMA total-return market series.

The v2.2.0 evaluator distinguishes **Strong Skill Candidate (Preliminary)** from confirmed Strong Evidence of Skill, and separately reports **Positive but Inconclusive** when both alpha estimates are positive and CAPM alpha is positive in at least 3/4 subperiods without meeting a higher tier. The potential-candidate rule requires positive marginal CAPM evidence, a positive significant four-factor alpha, positive alpha in at least 3/4 subperiods, and a bootstrap p-value below 0.10. Neither category claims confirmed skill. The executive summary reports the combined number of confirmed and preliminary skill candidates separately from the stricter confirmed-evidence count.

<!-- RESULTS_TABLE_START -->
| Scheme Code | Fund | Named Manager | Classification | Score (0-100) | CAPM alpha (HAC p) | 4-factor alpha (HAC p) |
|---|---|---|---|---|---|---|
| `122639` | PPFAS Flexi Cap | Rajeev Thakkar | **Strong Skill Candidate (Preliminary)** | **67.0** | +7.01% (p=0.074) | +7.88% (p=0.045) |
| `118778` | Nippon Small Cap | Samir Rachh | **Positive but Inconclusive** | **45.0** | +7.59% (p=0.119) | +4.96% (p=0.135) |
| `118668` | Nippon Growth Mid Cap | Rupesh Patel † | **Positive but Inconclusive** | **42.0** | +4.07% (p=0.294) | +3.13% (p=0.390) |
| `118989` | HDFC Mid-Cap | Chirag Setalvad | **Positive but Inconclusive** | **37.0** | +2.44% (p=0.537) | +1.51% (p=0.665) |
| `120505` | Axis Midcap | Shreyash Devalkar | **Positive but Inconclusive** | **37.0** | +2.41% (p=0.474) | +2.61% (p=0.426) |
| `120828` | Quant Small Cap | Sandeep Tandon † | **Weak/Inconclusive** | **36.0** | +11.08% (p=0.096) | +5.04% (p=0.306) |
| `118834` | Mirae Large & Midcap | Neelesh Surana | **Positive but Inconclusive** | **32.0** | +0.45% (p=0.882) | +1.25% (p=0.683) |
| `119071` | DSP Midcap | Vinit Sambre | **Weak/Inconclusive** | **12.0** | -0.11% (p=0.974) | -0.23% (p=0.941) |
| `118955` | HDFC Flexi Cap | Roshi Jain † | **Weak/Inconclusive** | **6.0** | -0.39% (p=0.922) | -0.62% (p=0.867) |
| `119598` | SBI Large Cap | Saurabh Pant † | **Weak/Inconclusive** | **6.0** | -2.63% (p=0.435) | -1.27% (p=0.709) |
| `120166` | Kotak Flexi Cap | Harsha Upadhyaya | **Weak/Inconclusive** | **6.0** | -3.53% (p=0.248) | -2.44% (p=0.426) |
| `120586` | ICICI Pru Large Cap | Anish Tawakley | **Weak/Inconclusive** | **6.0** | -1.52% (p=0.639) | -0.25% (p=0.937) |

† the named manager was in charge of the scheme for less than 90% of the sample window (see `attribution_note` in `final_evaluation.csv`).
<!-- RESULTS_TABLE_END -->

### Robustness: classification counts under alternative market proxies

The same rules, applied with a different market portfolio in the CAPM step (the 4-factor model always uses the
IIMA factors):

**Sensitivity-output status:** `benchmark_sensitivity.csv` was regenerated under the current v2.3.0 rules
(`python -m src.sensitivity`); the table below is generated from it. Under the price-index proxies PPFAS Flexi Cap
is a *Strong Skill Candidate (Preliminary)*, not "Strong Evidence of Skill" (an earlier version of this file showed
the latter because it pre-dated the Benjamini-Hochberg gate: PPFAS's four-factor q-value is 0.54).

<!-- SENSITIVITY_TABLE_START -->
| Market proxy | Strong Evidence of Skill | Strong Skill Candidate (Preliminary) | Moderate Evidence | Positive but Inconclusive | Weak/Inconclusive | Likely Style/Luck Driven |
|---|---|---|---|---|---|---|
| `iima_market` | 0 | 1 | 0 | 5 | 6 | 0 |
| `nifty500_price` | 0 | 1 | 5 | 0 | 5 | 1 |
| `nifty500_price_plus_assumed_dividend_yield` | 0 | 1 | 4 | 2 | 5 | 0 |
<!-- SENSITIVITY_TABLE_END -->

* `iima_market` - IIMA value-weighted market portfolio, total return (**primary**; same market as the 4-factor model).
* `nifty500_price` - NIFTY 500 *price* index. It excludes dividends, so every fund's alpha is biased **upward** by
  roughly the index dividend yield. (The original version of this project labelled this series "TRI".)
* `nifty500_price_plus_assumed_dividend_yield` - price index plus an **assumed** 1.3% p.a. dividend yield, a rough
  stand-in for the NIFTY 500 TRI. The yield is an assumption, not a measurement.

The NIFTY 500 **TRI** itself is not freely scriptable (NSE blocks automated downloads). To use it, download the
historical TRI series from niftyindices.com, save it as `data/raw/raw_benchmark_nifty500_tri.csv` (columns `Date`
and the TRI level) and re-run `python run_pipeline.py`: the pipeline then uses it automatically as the CAPM
market and adds it to the sensitivity table.

---

## Interpreting the output

* **Score (0-100)** ranks evidence strength: performance (0-20) + significance (0-25) + factor survival (0-25) +
  persistence (0-20) + style/luck adjustment (-10..+10). The **category**, not the score, drives the classification.
* A significantly **negative** alpha earns no significance or skill credit (rules v1.0.0 awarded points for it).
* "Persistent" means alpha positive in a **strict majority (at least 3 of 4)** of the subperiods.
* The **attribution caveat** in the dashboard / `attribution_note` column warns where the named manager ran the
  scheme for only part of the sample (e.g. HDFC Flexi Cap was run by Prashant Jain until July 2022).

## Known limitations (full list: `docs/METHODOLOGY.md`, section 7)

1. 12 hand-picked, well-known funds - selection and survivorship bias; no generalisation to the ~1,400 active schemes.
2. Five-year window (2019-2023) dominated by the 2020 crash/recovery; four persistence windows have little power.
3. Results depend on the market proxy; a genuine NIFTY 500 TRI is needed for the originally intended benchmark.
4. 12 hand-picked fund tests; BH q-values are now reported and required for the confirmed-strong tier. Several funds can change significance with the bootstrap block length (see sensitivity outputs).
5. India-only factors do not capture overseas holdings or large cash/arbitrage books (e.g. Parag Parikh Flexi Cap).
6. Manager names and tenure dates are curated registry data, not machine-verified.
7. Statistical significance is not causal evidence of skill.

See `docs/REVIEW_REPORT.md` for the independent verification of the numerical results and the full list of
corrections made in version 2.0.0.


Evaluation version 2.3.0 reports market-level Benjamini-Hochberg q-values for CAPM, bootstrap, and four-factor tests. The confirmed-skill tier also requires q < 0.10 across these evidence families. This is deliberately conservative; a zero confirmed count is a valid result, not a project failure.


## Current primary results (evaluation v2.3.0)

| Category | Funds |
|---|---:|
| Weak/Inconclusive | 6 |
| Positive but Inconclusive | 5 |
| Strong Skill Candidate (Preliminary) | 1 |

PPFAS Flexi Cap (scheme code 122639) is the sole **Strong Skill Candidate (Preliminary)**. Its annualized CAPM alpha is +7.01% (HAC p=0.074), four-factor alpha is +7.88% (HAC p=0.045), and bootstrap p=0.061. However, BH-adjusted q-values are much larger than 0.10, so the evidence does not survive cross-sectional multiple-testing adjustment. This is a candidate for further investigation, not confirmed manager skill.
