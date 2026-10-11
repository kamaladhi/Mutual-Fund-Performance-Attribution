# Numerical and Methodology Audit Report

**Release:** combined v2 (one rule set for both markets)
**Scope:** reproduction of the saved numerical outputs, consistency of the classification rules across the two markets, return
arithmetic, the dashboard, the test suites and the documented limitations.

## Executive result

**Local numerical-integrity checks: PASS.** The saved regression coefficients, Newey-West HAC p-values and Benjamini-Hochberg q-values
of both markets reproduce, both from the project's own formulas (`scripts/validate_project.py`) and from an independent implementation
(**statsmodels**, `tests/test_independent_numerics.py`). This does **not** certify the external data sources, the benchmark choice,
manager attribution or the economic interpretation.

**Rule consistency: PASS.** Both markets are classified by one rule set (evaluation v2.3.0). The rule functions are identical in
`markets/india/src/evaluator.py` and `markets/usa/src/evaluator.py`, and `tests/test_rule_parity.py` fails if they ever diverge.

## What was wrong in the previous release, and what changed

The previous release presented India (rules v2.3.0) and the USA (rules v1.5.0) side by side as if their classes were equivalent. They
were not: the USA evaluator still carried defects that the India review had already fixed.

| Defect (USA evaluator, previous release) | Effect | Now |
|---|---|---|
| A 2-of-4 persistence tie counted as a "majority" | Five funds (FLPSX, FMAGX, PAGRX, PRHSX, TRBCX) were labelled "persistent" on a coin flip | Strict majority (at least 3 of 4), as in India |
| "Moderate Evidence" needed only a positive CAPM alpha plus that majority | FLPSX (CAPM p = 0.94) and PAGRX (p = 0.92) were "Moderate" with no statistical support | Sign persistence needs CAPM p < 0.10 (India's gate) |
| "Candidate" tier needed no CAPM or bootstrap support | FDGRX (CAPM p = 0.22, bootstrap p = 0.24) was a candidate | Candidate needs CAPM p < 0.10 and bootstrap p < 0.10 |
| Significance score gave credit to a significantly negative alpha; style/luck score penalty applied to any positive alpha | Latent mis-scoring (e.g. AIVSX penalised at p = 0.80) | Sign-aware scoring; penalty needs CAPM p < 0.10 |
| No "Positive but Inconclusive" tier | US funds with only a positive direction were called "Moderate" | Same six classes as India |

Raw statistics are **unchanged**: the maximum absolute difference is 0 over the 23 CAPM, bootstrap, four-factor and persistence-summary
columns of the USA `final_evaluation.csv`, and the recomputed subperiod estimates agree to 1e-14. Only the rule layer moved. Effect on the
USA classification:

| Fund | Before (v1.5.0) | Now (v2.3.0) |
|---|---|---|
| FDGRX | Strong Potential Candidate | Moderate Evidence |
| FCNTX, MIGFX | Moderate Evidence | Positive but Inconclusive |
| FLPSX, PAGRX | Moderate Evidence | Weak/Inconclusive |
| all others | unchanged class | unchanged class |

India's results are unchanged (identical contents). Other corrections in this release:

- `benchmark_sensitivity.csv` (India) was stale: it pre-dated the Benjamini-Hochberg gate and showed PPFAS Flexi Cap as "Strong Evidence of
  Skill" under the price-index benchmarks. It was regenerated under the current rules (about 8 seconds); PPFAS is now a preliminary candidate
  under every benchmark. A test now fails if this file stops matching the primary results.
- `run_manifest.json` (India) recorded evaluator 2.0.0; it now records 2.3.0 with a provenance note.
- The candidate explanation no longer states "not below the strict 0.05 threshold" when the bootstrap p-value is below 0.05.
- The combined dashboard no longer merges classes (the old "Positive / Moderate Evidence" bucket hid two different tiers); it shows the
  classes in use (the preliminary tier under the display name "Strong"; the two classes with no funds only if a fund falls into them),
  with zeros, plain-English definitions, per-fund caveats (manager tenure, non-peer funds, overseas holdings) and a loud banner if
  the markets ever differ in rule version.
- USA dashboard text was corrected (block length 30, two-sided null, Mkt-RF market, rules version, a boolean printed as "Interpretation: False").
- `data/combined_results.csv` now has a generator and a freshness check.

## Independent numerical checks

| Check | India | USA |
|---|---:|---:|
| Funds | 12 | 12 |
| Aligned observations | 14,586 | 15,084 |
| Date window | 2019-01-02 to 2023-12-29 | 2019-01-03 to 2023-12-29 |
| Max CAPM alpha reproduction error | 8.9e-17 | 7.2e-17 |
| Max four-factor alpha reproduction error | 3.0e-16 | 2.7e-15 |
| Max HAC p-value reproduction error | 1.3e-12 | 3.0e-11 |

`tests/test_independent_numerics.py` additionally matches, for every fund in both markets, the OLS coefficients, classical p-values,
HAC p-values (Bartlett kernel, n/(n-k) correction) and the Benjamini-Hochberg q-values against statsmodels. Machine-readable results:
`AUDIT_REPORT.json`.

## Current classifications (evaluation v2.3.0, generated from the saved results)

| Market | Strong Evidence | Skill Candidate (Prelim.) | Moderate | Positive but Inconclusive | Weak/Inconclusive | Style/Luck |
|---|---:|---:|---:|---:|---:|---:|
| India | 0 | 1 | 0 | 5 | 6 | 0 |
| USA | 0 | 0 | 1 | 2 | 9 | 0 |
| **Combined** | **0** | **1** | **1** | **7** | **15** | **0** |

The classes come from fixed rules applied identically to every fund; no distribution was targeted, selected or adjusted. The empty classes
are empty because no fund meets their rules on this data.

- **India: PPFAS Flexi Cap (122639)** is the only preliminary candidate. CAPM alpha +7.01% (HAC p = 0.074), four-factor alpha +7.88%
  (p = 0.045), bootstrap p = 0.061. BH q-values 0.477 (CAPM), 0.540 (four-factor), 0.723 (bootstrap): the evidence does not survive
  multiple-testing adjustment. It also holds overseas equities and large cash positions that the India-only factors cannot capture.
- **USA: Fidelity Growth Company (FDGRX)** is "Moderate Evidence": four-factor alpha +5.51% (HAC p = 0.025) but CAPM alpha +5.41%
  (p = 0.223) and bootstrap p = 0.245. BH q-values 0.985 / 0.303 / 0.984.

### Classification under alternative Indian benchmarks (same rules; Benchmark robustness page)

| Benchmark | Strong | Candidate | Moderate | Positive but Inconclusive | Weak | Style/Luck |
|---|---:|---:|---:|---:|---:|---:|
| IIMA total-return market (primary) | 0 | 1 | 0 | 5 | 6 | 0 |
| NIFTY 500 price index (excludes dividends) | 0 | 1 | 5 | 0 | 5 | 1 |
| NIFTY 500 price + assumed 1.3% dividend yield | 0 | 1 | 4 | 2 | 5 | 0 |

## Automated test results

| Suite | Result |
|---|---:|
| Repository (root) | 43 passed |
| USA project | 111 passed |
| India project | 181 passed |
| **Total** | **335 passed, 0 skipped** |

Each market suite is run from its own directory (`python -m pytest -q tests`); the repository suite from the root. The India Streamlit
render test, which was skipped in the previous audit (Streamlit not importable at test time), now runs, and the combined dashboard has its
own headless render test of every page. The tests assert invariants, not the current classifications, and the USA and India suites use
temporary copies so a test run never rewrites the shipped outputs.

## Outstanding limitations (do not conceal)

1. **Benchmark completeness:** the US S&P 500 series is a price index (descriptive only); CAPM and the four-factor model use the Ken French
   `Mkt-RF` series. The India primary market is the IIMA total-return portfolio; the saved NIFTY 500 series is a price-index sensitivity proxy,
   not a genuine TRI. Acquire the official TRI before making TRI-based claims.
2. **US fund-universe comparability:** AMECX is income-oriented, PRHSX is healthcare-sector and PAGRX is multi-asset. They remain visible as
   exploratory observations and are flagged wherever they appear. PRGFX (T. Rowe Price Growth Stock) has not been acquired.
3. **Selection and survivorship:** 12 selected funds per market and a 2019-2023 window are not representative of all funds.
4. **Statistical power:** a single fund's four-factor alpha is measured only to about ±4 (USA) to ±7 (India) points a year (95%), so only a large
   true alpha would be detected. Few strong verdicts is the expected outcome, not a defect.
5. **Manager attribution:** manager names and tenure dates are curated, not machine-verified. For four Indian schemes the named manager was
   in charge for under 90% of the window; the result describes the scheme.
6. **Multiple testing:** Benjamini-Hochberg q-values are computed within each market, not pooled across the 24 funds; they gate only the
   strict class.
7. **Bootstrap:** the circular-block bootstrap tests CAPM alpha with a fixed block length (30 days), a raw (not studentised) statistic and 1,000
   draws; results near the 5% line are sensitive to these choices. The USA bootstrap summary does not store one-sided p-values.
8. **Model risk:** four factors omit sector, quality, liquidity, cash and overseas exposure. Alpha is model-relative, not causal proof of skill.
9. **Rule history:** the Candidate and "Positive but Inconclusive" tiers were added to India's rules after its first results were seen. They are
   fixed and applied identically to every fund, but a reader should know the rules were not pre-registered.

## Conclusion

Both markets are now classified by one auditable rule set, the saved numbers reproduce against an independent implementation, the
dashboard shows every class and every caveat, and the outputs are regenerated from their sources. The result is an honest preliminary
analysis: it finds no robust evidence of skill among these 24 funds, one preliminary candidate (PPFAS Flexi Cap) and one moderate signal
(FDGRX), neither of which survives multiple-testing adjustment. It is not externally validated for data sources, benchmarks or manager
attribution.
