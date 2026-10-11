# Fund Manager Skill Detector: India + USA

Do flagship mutual funds show investment skill, or only market exposure, style and luck? This repository holds the Indian and
US analyses (12 funds each, 2019-2023) and one Streamlit dashboard that presents them side by side.

**Both markets are classified by one rule set (evaluation v2.3.0).** A class therefore means exactly the same thing in India and
the USA. The rules and saved results are never tuned to a target. For readability the dashboard shows the preliminary tier
("Strong Skill Candidate (Preliminary)") under the short name **Strong**, and shows "Strong Evidence of Skill" and "Likely
Style/Luck Driven" only if a fund falls into them (currently none does, except one fund under a price-index benchmark on the
Benchmark robustness page). This is display only; the CSV and the documents use the full class names.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py            # or double-click run_dashboard.bat
```

The dashboard has seven pages: **Overview** (the classes in use, definitions, counts, alpha chart), **India**, **USA**
(fund-by-fund tables and plain-English explanations), **Cross-market comparison**, **Benchmark robustness** (India under three
market benchmarks), **Fund explorer** (one fund, test by test) and **Methodology & data**. It only reads the saved results; it
never estimates anything.

## The six evidence classes

| Class | Meaning in one line |
|---|---|
| Strong Evidence of Skill | Every strict test passes, including Benjamini-Hochberg q < 0.10 for all three tests |
| Strong Skill Candidate (Preliminary) | Screening tier: marginal CAPM evidence, significant four-factor alpha, 3-of-4 persistence, bootstrap p < 0.10 |
| Moderate Evidence | At least one material positive signal without the convergence above |
| Positive but Inconclusive | Both alphas positive and persistent, nothing statistically significant |
| Weak/Inconclusive | No convincing positive evidence (not proof of no skill) |
| Likely Style/Luck Driven | A positive CAPM alpha that the factor adjustment removes |

The rules are applied top to bottom with fixed thresholds, identically to every fund (see `docs/CLASSIFICATION_GUIDE.md`).
**No class distribution is targeted.** A class is a statement about the strength of statistical evidence, not about the quality of a
manager; with 12 funds, five years of daily data and a multiple-testing correction, most funds are expected to be inconclusive.

## Current results (evaluation v2.3.0)

| Class | India | USA | Total |
|---|---:|---:|---:|
| Strong Evidence of Skill | 0 | 0 | 0 |
| Strong Skill Candidate (Preliminary) | 1 (PPFAS Flexi Cap) | 0 | 1 |
| Moderate Evidence | 0 | 1 (FDGRX) | 1 |
| Positive but Inconclusive | 5 | 2 | 7 |
| Weak/Inconclusive | 6 | 9 | 15 |
| Likely Style/Luck Driven | 0 | 0 | 0 |

These counts are generated from the saved results (`python scripts/build_combined_results.py`), not typed in. Nothing survives the
Benjamini-Hochberg multiple-testing correction (smallest four-factor q-value: 0.54 in India, 0.30 in the USA). Under alternative
Indian benchmarks the classification changes (see the Benchmark robustness page): against a dividend-free price index several funds
look *Moderate* and one looks *Style/Luck driven*, which is why the benchmark choice is reported as a finding.

## Project layout

```text
fund_manager_skill_detector_combined/
├── app.py                         # Unified dashboard (reads saved results only)
├── combined_data.py               # Data layer shared by the dashboard, the export script and the tests
├── requirements.txt
├── data/combined_results.csv      # Combined display table (generated; see scripts/build_combined_results.py)
├── docs/                          # COMBINED_ARCHITECTURE, CLASSIFICATION_GUIDE, AUDIT_REPORT (.md/.json)
├── scripts/
│   ├── build_combined_results.py  # regenerate / --check data/combined_results.csv
│   └── validate_project.py        # numerical audit of the saved outputs
├── markets/
│   ├── india/                     # self-contained project: src/ data/ docs/ tests/ app.py run_pipeline.py
│   └── usa/                       # self-contained project: src/ data/ docs/ tests/ app.py
└── tests/                         # repository-level checks (rule parity, statsmodels cross-check, dashboard render, export)
```

The two market projects keep separate `src` packages (they share module names), so each is run from its own folder. The rule code
exists in two files and is **kept identical by a test** (`tests/test_rule_parity.py`).

## Tests

```bash
run_combined_tests.bat                # Windows: repository tests, then USA, then India
# or, by hand:
python -m pytest -q tests             # from the repository root  (43 tests)
cd markets\usa   && python -m pytest -q tests     # 111 tests
cd markets\india && python -m pytest -q tests     # 181 tests
```

Highlights: the saved OLS coefficients, Newey-West HAC p-values and Benjamini-Hochberg q-values of both markets are re-checked
against **statsmodels** (`tests/test_independent_numerics.py`); the rule functions of both markets are compared syntactically
(`tests/test_rule_parity.py`); every dashboard page is rendered headlessly (`tests/test_combined_app.py`); `data/combined_results.csv`
must equal a fresh build from its sources. The tests assert invariants, not the current classifications.

## Re-running an analysis

The dashboard reads `markets/<market>/data/processed/final_evaluation.csv`. To rebuild a market's results:

```bash
cd markets/india && python run_pipeline.py                 # whole India pipeline (data cached; no download)
cd markets/usa   && python -m src.persistence && python -m src.evaluator .   # USA persistence + classification
python scripts/build_combined_results.py                   # refresh the combined export
```

The dashboard notices changed result files by itself (refresh the browser page).

If one market is ever re-evaluated with a different rule version, the dashboard says so on every page and refuses to present the
classes as comparable.

## Limitations (do not conceal)

- Twelve hand-picked, well-known funds per market over five years: selection and survivorship bias, low statistical power (a single
  fund's four-factor alpha is only pinned down to roughly ±4 to ±7 points a year).
- The Indian primary market is the IIMA total-return portfolio; a genuine NIFTY 500 TRI series has not been added, and the NIFTY 500 series
  in the data is a price index.
- Three US funds (AMECX, PRHSX, PAGRX) are exploratory non-peers; T. Rowe Price Growth Stock (PRGFX) was not acquired.
- Manager names and tenure dates are curated, not machine-verified; where a named manager ran a scheme for under 90% of the window
  the result describes the scheme (the dashboard flags these funds).
- The raw alphas of the two markets use different market portfolios, factor sets and calendars and are not a like-for-like league table.
- This is a statistical evidence summary, not investment advice.
