# Fund Manager Skill Detector (US market): CAPM & Carhart 4-Factor Evaluation with HAC and Circular Block Bootstrap Inference

## Overview
A quantitative framework that asks whether the outperformance of 12 US mutual funds (2019-2023) can be distinguished from luck, market
exposure and systematic factor tilts (size, value, momentum). Evidence from four layers is combined into a transparent, deterministic,
rule-based **evidence classification and 0-100 evidence index** (`src/evaluator.py`, rules version **2.3.0**):

1. **CAPM** Jensen's alpha (OLS estimate, Newey-West HAC inference), against the Fama-French `Mkt-RF` market excess return;
2. a **circular block bootstrap** of the CAPM alpha (paired, null-centred, two-sided);
3. the **Carhart four-factor** model (Mkt-RF, SMB, HML, momentum);
4. **subperiod persistence** (four chronological windows; "majority" = at least 3 of 4).

> **Same rules as the India market.** The rule layer is identical to `markets/india/src/evaluator.py` (a repository-level test enforces
> this), so US and Indian classes are directly comparable. The classification measures the *strength of statistical evidence*; it is not
> proof of manager skill and not investment advice.

## Project structure
```
usa/
├── app.py                    # Streamlit dashboard (this market only; the combined dashboard is in the repository root)
├── requirements.txt
├── data/
│   ├── raw/                  # Yahoo Finance fund and S&P 500 prices, Ken French factors, screening_metadata.json
│   └── processed/            # aligned dataset and every analytical output (csv/json), final_evaluation.csv
├── docs/                     # METHODOLOGY, ARCHITECTURE, DATA_SOURCES, DATA_DICTIONARY, DECISIONS, ... (see note below)
├── src/
│   ├── data_loader.py        # acquisition, validation, alignment (fund returns, Fama-French factors, S&P 500 for description)
│   ├── stats_utils.py        # Newey-West HAC covariance, Benjamini-Hochberg q-values
│   ├── capm.py  bootstrap.py  factor_model.py  persistence.py   # the four analytical engines
│   └── evaluator.py          # rule-based evidence classification (rules v2.3.0, shared with India)
└── tests/                    # pytest suite (111 tests; evaluator tests use temporary copies, never the shipped outputs)
```
`docs/PROGRESS.md`, `DECISIONS.md` and `ARCHITECTURE.md` are the historical build log (written under rules v1.0.0); the current rules are in
`docs/METHODOLOGY.md`.

## Running (from this folder)
```bash
streamlit run app.py                   # dashboard for this market
python src/data_loader.py              # data acquisition + alignment (raw files are never overwritten)
python src/capm.py                     # CAPM (OLS + HAC)
python -m src.bootstrap                # circular block bootstrap
python -m src.factor_model             # Carhart four-factor model
python -m src.persistence              # four-subperiod persistence
python -m src.evaluator .              # rule-based classification -> data/processed/final_evaluation.csv
python -m pytest -q tests              # 111 tests
```

## Current results (evaluation v2.3.0)

Generated from `data/processed/final_evaluation.csv` (CAPM and four-factor p-values are Newey-West HAC p-values):

| Fund | Ticker | Class | Score | CAPM alpha (HAC p) | 4-factor alpha (HAC p) |
|---|---|---|---:|---|---|
| Fidelity Growth Company Fund | FDGRX | Moderate Evidence | 62 | +5.41% (p=0.223) | +5.51% (p=0.025) |
| MFS Massachusetts Investors Growth Stock Fund | MIGFX | Positive but Inconclusive | 37 | +1.95% (p=0.324) | +1.54% (p=0.361) |
| Fidelity Contrafund | FCNTX | Positive but Inconclusive | 37 | +1.32% (p=0.632) | +0.84% (p=0.657) |
| Fidelity Low-Priced Stock Fund | FLPSX | Weak/Inconclusive | 26 | +0.31% (p=0.936) | +0.79% (p=0.731) |
| Permanent Portfolio Aggressive Growth Fund | PAGRX | Weak/Inconclusive | 26 | +0.40% (p=0.922) | +1.02% (p=0.791) |
| Investment Company of America | AIVSX | Weak/Inconclusive | 20 | +0.36% (p=0.803) | +0.14% (p=0.917) |
| American Funds Growth Fund of America | AGTHX | Weak/Inconclusive | 12 | -0.27% (p=0.911) | -0.20% (p=0.905) |
| Dodge & Cox Stock Fund | DODGX | Weak/Inconclusive | 12 | -0.81% (p=0.853) | -0.48% (p=0.822) |
| Fidelity Magellan Fund | FMAGX | Weak/Inconclusive | 6 | -0.06% (p=0.985) | -0.62% (p=0.731) |
| T. Rowe Price Health Sciences Fund | PRHSX | Weak/Inconclusive | 6 | -0.79% (p=0.879) | -0.55% (p=0.904) |
| T. Rowe Price Blue Chip Growth Fund | TRBCX | Weak/Inconclusive | 6 | -2.24% (p=0.583) | -2.63% (p=0.249) |
| Income Fund of America | AMECX | Weak/Inconclusive | 0 | -1.30% (p=0.608) | -1.37% (p=0.474) |

| Class | Funds |
|---|---:|
| Strong Evidence of Skill | 0 |
| Strong Skill Candidate (Preliminary) | 0 |
| Moderate Evidence | 1 |
| Positive but Inconclusive | 2 |
| Weak/Inconclusive | 9 |
| Likely Style/Luck Driven | 0 |

No fund survives the Benjamini-Hochberg multiple-testing correction (smallest four-factor q-value 0.30). FDGRX is the only fund with a
significant four-factor alpha (HAC p = 0.025), but its CAPM and bootstrap tests do not agree (p = 0.223 / 0.245), so it is "Moderate", not a
candidate. The classes come from fixed rules applied identically to every fund; no distribution is targeted.

## Release history (rule layer only; raw statistics have not changed)
- **v2.3.0 (combined v2):** the US evaluator and persistence module were brought onto the India rule set: strict-majority persistence (a 2-of-4
  tie no longer counts), sign-aware significance score and style/luck adjustment, India's gates for "Moderate" and "Candidate", and the
  "Positive but Inconclusive" tier. Effect: FDGRX Candidate -> Moderate; FCNTX and MIGFX Moderate -> Positive but Inconclusive; FLPSX and PAGRX
  Moderate -> Weak.
- v1.5.0: HAC inference, Fama-French Mkt-RF for CAPM, Benjamini-Hochberg q-values, style/luck label requires positive CAPM alpha with p < 0.10.
- v1.3.0 / v1.1.0: candidate tier introduced; style/luck precedence corrected.

## Methodological warnings
> - Only 12 candidate funds are analysed over about 5 years (2019-2023); survivorship and selection bias are likely.
> - AMECX (income), PRHSX (healthcare sector) and PAGRX (multi-asset) are exploratory non-peers for a broad US equity universe. PAGRX is
>   Permanent Portfolio Aggressive Growth, **not** T. Rowe Price Growth Stock (PRGFX, not acquired).
> - Statistical significance does not prove manager skill; factor adjustment is evidence, not causal attribution.
> - A single fund's four-factor alpha is measured only to about ±4 points a year (95%), so only a large true alpha would be detected.
> - The S&P 500 series is a price index (descriptive only); CAPM and the four-factor model use the Fama-French market excess return.
