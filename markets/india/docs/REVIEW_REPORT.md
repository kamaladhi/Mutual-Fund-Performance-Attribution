# Independent Review Report: Indian Fund Manager Skill Detector

*Review date: 2026-10-11. Subject: `Finance project/indian_fund_skill_detector` (original rules version 1.0.0). Result: rules version 2.0.0.*
*The original project is preserved unchanged in `Finance project - ORIGINAL BACKUP before review (2026-10-11).zip` on the Desktop.*

## 0. Verdict

| | Original (v1.0.0) | After this review (v2.0.0) |
|---|---|---|
| Arithmetic of the estimators | **Correct** | Correct (unchanged) |
| Headline conclusions ("2 Strong, 5 Moderate, 5 Style/Luck") | **Not reliable**: built on a mislabelled, upward-biased benchmark and on classification rules with logic defects | Re-derived: **0 Strong / 1 Moderate / 11 Weak** under the primary specification; **1 / 4-5 / 5-7 / 0-1** under TRI-like proxies |
| Dashboard | Ran, numbers matched the CSVs, **but every data table rendered blank** and several on-screen statements were false | All tables render; statements corrected; robustness shown |
| Reproducibility | Reproducible to ~1e-11, but README commands partly broken, no `requirements.txt` | One command (`python run_pipeline.py`), deterministic |
| Ready for final submission? | **No** | **Yes, with the open items in Section 5** (chiefly: decide on the benchmark, ideally by supplying a genuine NIFTY 500 TRI file; re-write any text that quotes v1 numbers) |

**Scope and honesty about limits.** Every source, test, data and documentation file in the project folder was read; the 76 original tests were run; every regression was re-estimated
independently (`numpy.linalg.lstsq` and `statsmodels`); the bootstrap and the evaluator rules were re-implemented from the documentation; raw data were audited and checked against
external sources (the mfapi.in scheme master, the IIMA release notes, press reports on HDFC fund management); the dashboard was run headless for all 12 funds and inspected in a real browser
(Microsoft Edge). **Not verified** (cannot be from this environment): the IIMA factor values themselves, the AMFI NAV values against AMC sites, daily NIFTY 500 prices against NSE,
fund manager names/tenures other than HDFC Flexi Cap, and expense ratios.

---

## 1. What is correct (verified)

| Area | Evidence |
|---|---|
| OLS mathematics (CAPM and 4-factor) | alpha, beta/loadings, standard errors, t-statistics, p-values, R-squared, Durbin-Watson and Jarque-Bera of all 12 funds equal `statsmodels` to <= 2e-12 |
| Annualisation | ((1+a)^252 - 1) x 100 implemented as documented; CI transformation monotone |
| Bootstrap mechanics | independent vectorised re-implementation (B = 4,000, different seed) reproduces the p-values and CIs within Monte-Carlo noise; null-centring and +1 smoothing correct; paired resampling keeps fund/market observations together |
| Persistence engine | four non-overlapping chronological windows, no shuffling, counts consistent with the period alphas |
| Evaluator vs its documentation | re-implemented from `METHODOLOGY.md`: 0 mismatches in score or category for all 12 funds; JSON equals CSV; score components sum to the total |
| Factor data | the IIMA file is genuine (1993-10 to 2025-12, time-varying RF); percent -> decimal scaling correct; RF accrues per calendar day (Monday ~3x), consistent with weekend-spanning NAV returns |
| Fund identity | all 12 AMFI scheme codes map to the intended scheme, category and Direct-Growth plan (queried on mfapi.in) |
| Data hygiene | no non-positive or duplicate NAVs; extreme returns only on 2020-03-12/23 and 2020-04-07 (genuine) |
| Reproducibility | re-running the original pipeline reproduces every output to <= 4e-11 (seed 42 is deterministic) |
| Original dashboard code path | ran without exceptions for all 12 funds; displayed numbers equal the CSVs |

The original 76 tests all passed and none of the defects below was detected by them: the suite tested that code does what it says, not that what it says is right.

---

## 2. What was wrong

Severity: **C** = changes the headline conclusions, **H** = materially misleading, **M** = methodological or quality defect, **L** = polish.

| # | Sev | Finding | Evidence | Impact | Correction in v2.0.0 |
|---|---|---|---|---|---|
| 1 | C | **The "NIFTY 500 TRI" benchmark is the NIFTY 500 *price* index.** Yahoo `^CRSLDX` closes at 9,197.9 / 19,429.2 (2019-01-01 / 2023-12-29), `Dividends` column empty; Yahoo has no NIFTY TRI series. Docs, dashboard and metadata all called it TRI. | raw file; compounding gives 16.6% p.a. | Dividends omitted from the market but included in fund NAVs: every CAPM alpha is biased **upward** by roughly the dividend yield | CAPM now uses the IIMA total-return market; the price index is kept only as a labelled sensitivity; a genuine TRI file is picked up automatically if supplied; every "TRI" claim corrected |
| 2 | C | **CAPM and 4-factor used different markets** (Nifty price vs IIMA `MF`); `alpha_reduction_pct` therefore mixed "benchmark swap" with "factor tilts". The two series differ by 4.6 pts/yr (IIMA 21.2% vs 16.6%) over the sample | descriptive statistics | the "Style/Luck" label partly measured the benchmark change; on the same sample the price-index CAPM alphas exceed the IIMA-market alphas by 3.1-5.5 pts/yr (mean 4.1) | both models share the IIMA market, so the reduction isolates SMB/HML/WML |
| 3 | C | **Evaluator logic defects** (reproduced with the project's own functions): (a) significantly *negative* alpha earned 15/25 significance points because the thresholds ignore sign; (b) a negative-alpha fund could be labelled "Likely Style/Luck Driven" with text about "observed outperformance"; (c) the Moderate rule was non-monotone (adding persistence demoted a fund from Moderate to Weak); (d) persistence "majority" counted a 2-of-4 tie, so **12 of 12** funds were "persistent", flag with no discriminating power | `tests`-style synthetic rows; real data | inflated scores and mislabelled categories; the five "Style/Luck" funds had CAPM p-values 0.20-0.95, i.e. no outperformance to explain (Kotak: alpha +0.10%, "reduction" 2,628%) | sign-aware scoring; style/luck requires a positive CAPM alpha with p < 0.10; monotone Moderate; strict majority (>= 3 of 4: now 7 of 12) |
| 4 | C | **Classical OLS p-values on autocorrelated, non-normal residuals** (Durbin-Watson as low as 1.67, Jarque-Bera rejects normality for every fund) drove "significant" flags. The "Strong" status of Nippon Small Cap hinged on a 4-factor p-value of 0.048; with Newey-West HAC it is 0.098 on the original sample | independent HAC re-estimate | over-confident inference | HAC p-values (verified against `statsmodels` and brute force) are used for decisions |
| 5 | C | **Silent synthetic data.** If the IIMA download failed, `data_loader` generated random "calibrated" factor returns (constant RF) and continued without error. The shipped file is genuine, so no result was affected, but the hazard was real and undocumented in the dashboard | code | any network hiccup would silently fabricate the factor data | removed: the pipeline raises; a constant-RF placeholder file is rejected |
| 6 | H | **Every `st.dataframe` rendered blank.** A global CSS rule gave `background-color` to every `div`/`span`, hiding the grid canvas (isolated by A/B test: font/colour rules harmless, background rule blanks it). CAPM table, persistence table and leaderboard were empty boxes in the browser | headless Edge screenshots | the dashboard's central tables were invisible | CSS scoped; regression test |
| 7 | H | **Asynchronous returns.** Returns were computed per series *before* alignment. 135 of 14,721 observations paired a 2-day fund (or benchmark) return with a 1-day return from another source (Muhurat/Saturday sessions, days missing in Yahoo, missing NAVs) | recomputed from raw files | alphas moved by up to +/-0.54 pts/yr | synchronous-interval rule; observations counted |
| 8 | H | **Manager attribution invalid for four funds.** Four named managers took charge after the sample opened or only part-way: HDFC Flexi Cap (Roshi Jain **2022-07-29**, not 2022-01-14; Prashant Jain ran it until July 2022; she has since left), Nippon Growth Mid Cap (54% of window), SBI Large Cap (75%), Quant Small Cap (83%). 2019-2023 results were presented as that manager's skill | press reports (Section 6) | the "manager skill" label was wrong for a quarter-to-half of the sample on these funds | registry corrected; `tenure_overlap_pct` and an attribution caveat on the dashboard and in the output |
| 9 | H | **False or stale statements** in docs/UI: "NIFTY 500 TRI (^CRSLDX)"; block length 30 "= T^{1/3}" (T^{1/3} = 10.7; 30 is 2.8x larger); "H0: alpha <= 0" while the test is two-sided H0: alpha = 0; "Direct Plan mandated by SEBI for active strategies / exclusively active" (false: index funds have direct plans); "survivorship adjustment critical for 2019-2023" (it corrects mid-1990s vanishing companies); "1,227 common trading days" (only nine of the twelve funds had all 1,227 dates; three had 1,226); "11-condition decision matrix penalising non-normal p-value inflation" (no such thing); "four core criteria" (code needs five); persistence date ranges that disagree with the code; warning W5 "some funds UNVERIFIED" vs metadata "all VERIFIED"; references to a "US version" that exists only as a documentation scaffold | docs, app, code | misleading a reader or examiner | all corrected; a test fails if the known-false phrases reappear |
| 10 | H | **No multiple-testing or selection control.** Twelve hand-picked, famous funds, 5% test each. Bootstrap p-values: minimum 0.061 (two-sided) and **minimum BH q-value 0.72** under the primary specification | bootstrap results | an apparent "winner" among 12 tests is expected by chance | BH q-values reported; limitation stated; classification rules deliberately not changed |
| 11 | M | **Bootstrap flag was sign-blind** (two-sided p < 0.05 counted a significantly negative alpha as "significant") and had no block-length sensitivity | code | contributed to finding 3(a); several funds crossed 5% as b varied under the original spec (Quant Small Cap: p = 0.014 at b = 1, 0.087 at b = 60) | direction-aware flag, one-sided p, sensitivity table and plot |
| 12 | M | **`alpha_reduction_pct` is ill-conditioned** when |CAPM alpha| ~ 0 | Kotak 2,628%, SBI 442% | meaningless percentages drove classification | annual points reported alongside; classification gated on a noteworthy CAPM alpha |
| 13 | M | **Validation never executed**: `validate_dataframe()` existed and was unit-tested but the pipeline never called it; the dropna in alignment ignored SMB/HML/Mom | code | invalid raw data would pass silently | validation wired in (report written), full-column completeness enforced |
| 14 | M | **Required descriptive statistics not implemented** (FR-STA-01: annualised return, volatility, Sharpe) | grep | stated requirement unmet | `descriptive.py` (buy-and-hold basis, verified against point-to-point NAV arithmetic to 1e-12) |
| 15 | M | **Process/QA gaps**: tests pinned the *answers* (`2 Strong, 5 Moderate, 5 Style/Luck`, min 12 / max 90) so a correct fix would "fail"; tests overwrite deliverables; `python src/bootstrap.py` (README step) fails with `ModuleNotFoundError`; no `requirements.txt`; `results/figures` and `results/tables` empty although documented | run | brittle, unreproducible hand-off | invariant-based tests, independent integrity tests, `run_pipeline.py`, `requirements.txt`, generated tables/figures, README tables generated from outputs |
| 16 | L | Dashboard polish: p-value metric strings truncated ("0.0079 (p < 0..."), bar-count labels clipped, serif chart fonts clashing with the UI font, sidebar facts hard-coded ("1,227 common days"), leaderboard cut off at 10 rows, deprecated `use_container_width` | screenshots | cosmetic | fixed |
| 17 | L | Repository hygiene: doubly nested folder (`indian_fund_skill_detector/indian_fund_skill_detector`), cache directories in the folder, a second sibling folder (`fund_manager_skill_detector`) that contains only a documentation scaffold for a never-built "US version" | listing | confusing for an examiner | caches removed; the nesting and scaffold folder left for you (Section 5) |

---

## 3. Effect of the corrections on the results

Classification per fund (score in brackets). v1 = original; the three right-hand columns are v2 rules under each market proxy.

| Fund | v1 | v2 **primary** (IIMA market) | v2 price index | v2 price + assumed 1.3% dividend yield | v1 CAPM alpha (OLS p) | v2 CAPM alpha (HAC p) | v2 4-factor alpha (HAC p) |
|---|---|---|---|---|---|---|---|
| PPFAS Flexi Cap | Strong (90) | **Moderate (67)** | Strong (95) | Strong (82) | +10.3% (0.011) | +7.0% (0.074) | +7.9% (0.045) |
| Nippon Small Cap | Strong (80) | Weak (45) | Moderate (60) | Moderate (52) | +12.6% (0.004) | +7.6% (0.119) | +5.0% (0.135) |
| Nippon Growth Mid Cap | Moderate (65) | Weak (42) | Moderate (57) | Moderate (42) | +8.9% (0.008) | +4.1% (0.294) | +3.1% (0.390) |
| Mirae Large & Midcap | Moderate (60) | Weak (32) | Moderate (65) | Moderate (45) | +4.4% (0.014) | +0.5% (0.882) | +1.2% (0.683) |
| Axis Midcap | Moderate (50) | Weak (37) | Moderate (50) | Weak (50) | +6.1% (0.054) | +2.4% (0.474) | +2.6% (0.426) |
| HDFC Mid-Cap | Moderate (50) | Weak (37) | Style/Luck (37) | Weak (42) | +7.2% (0.045) | +2.4% (0.537) | +1.5% (0.665) |
| Quant Small Cap | Moderate (47) | Weak (36) | Moderate (47) | Moderate (47) | +16.8% (0.013) | +11.1% (0.096) | +5.0% (0.306) |
| DSP Midcap | Style/Luck (32) | Weak (12) | Weak (32) | Weak (27) | +4.0% (0.202) | -0.1% (0.974) | -0.2% (0.941) |
| HDFC Flexi Cap | Style/Luck (22) | Weak (6) | Weak (26) | Weak (21) | +3.1% (0.275) | -0.4% (0.922) | -0.6% (0.867) |
| ICICI Pru Large Cap | Style/Luck (17) | Weak (6) | Weak (27) | Weak (22) | +1.7% (0.295) | -1.5% (0.639) | -0.3% (0.937) |
| Kotak Flexi Cap | Style/Luck (12) | Weak (6) | Weak (16) | Weak (6) | +0.1% (0.946) | -3.5% (0.248) | -2.4% (0.426) |
| SBI Large Cap | Style/Luck (12) | Weak (6) | Weak (22) | Weak (6) | +0.5% (0.760) | -2.6% (0.435) | -1.3% (0.709) |
| **Counts (Strong/Moderate/Weak/Style)** | 2 / 5 / 0 / 5 | **0 / 1 / 11 / 0** | 1 / 5 / 5 / 1 | 1 / 4 / 7 / 0 | | | |

How to read this:
* **Robust conclusions**: (i) the original "2 Strong / 5 Style-Luck" picture is not supportable; (ii) Parag Parikh Flexi Cap ranks first under every proxy and is the only fund whose 4-factor alpha survives
  (+7.9% p.a., HAC p = 0.045; the 4-factor alpha does not depend on the CAPM proxy) - but note its p = 0.045 is borderline, its bootstrap p is 0.052-0.080 across block lengths, its BH q-value is 0.72, and it holds overseas equities and large cash/arbitrage books that India-only factors cannot capture;
  (iii) no fund has a significantly positive CAPM alpha after multiple-testing adjustment (minimum BH q = 0.72).
* **Benchmark-dependent conclusions**: whether the small/mid-cap funds (Nippon Small Cap, Quant Small Cap, Nippon Growth Mid Cap) reach "Moderate" depends on the market proxy. The IIMA market compounds at
  21.2% p.a. against 16.6% for the NIFTY 500 price index (about 3 points above a TRI-like series), so it is a demanding yardstick, especially for large-cap funds: an index-like fund would show a CAPM alpha of
  roughly -3% against it. This is why a genuine NIFTY 500 TRI file is the most valuable remaining input (Section 5).
* Keeping the original (price-index) benchmark, correcting the rules, the inference and the alignment alone already moves 2 / 5 / 0 / 5 to 1 / 5 / 5 / 1; the remainder of the movement comes from using a total-return market.

---

## 4. Method used to verify (reproducible)

| Check | How |
|---|---|
| Regression arithmetic | `statsmodels` OLS (and HAC) vs stored CSVs: max difference 2e-12; permanently guarded by `tests/test_data_integrity.py` (uses `numpy.linalg.lstsq` and a brute-force Newey-West) |
| HAC / FDR implementations | vs `statsmodels` `cov_type="HAC"` (rel. diff 1.4e-15) and `multipletests(method="fdr_bh")` (1e-16) |
| Bootstrap | independent vectorised implementation with a different seed; block-length grid {1,5,10,20,30,60} |
| Evaluator | re-implemented from the documentation; 0 mismatches (v1) |
| Alignment | recomputed `prev_date` of fund, benchmark and IIMA from the raw files for every aligned row (test) |
| Descriptive statistics | cumulative return equals last/first NAV - 1 to 1.5e-12; NIFTY 500 price CAGR 16.56% |
| Dashboard | `streamlit.testing` headless run for all 12 funds (test) + Microsoft Edge screenshots of every tab (original and corrected) |
| Reproducibility | full pipeline run twice on a temporary copy -> identical `final_evaluation.csv` (test) |
| Documentation | README tables must equal the shipped outputs; known-false phrases must not reappear (tests) |

Final state: the complete suite passes (`python -m pytest`).

---

## 5. Open items for you (not things I could or should decide)

1. **Benchmark decision.** The primary specification uses the IIMA total-return market (verifiable, same market as the 4-factor model, but demanding). If your write-up is meant to use the **NIFTY 500 TRI**, download its daily history from niftyindices.com
   (NSE blocks scripted access, so this is a manual step), save it as `data/raw/raw_benchmark_nifty500_tri.csv` (`Date` + TRI column) and run `python run_pipeline.py`; everything (including the README tables and the dashboard) updates.
2. **Re-write any text that quotes v1 results** (2 Strong / 5 Moderate / 5 Style-Luck, "Parag Parikh 90 / Nippon Small Cap 80", "TRI benchmark", "76 tests"). Suggested honest headline: *"After correcting for dividends, autocorrelation and multiple testing,
   no fund shows robust evidence of skill; Parag Parikh Flexi Cap is the strongest candidate (4-factor alpha +7.9% p.a., HAC p = 0.045), and conclusions for the small/mid-cap funds depend on the benchmark."*
3. **Verify manager names and tenures** against AMC scheme documents (co-managers are common; the registry is curated, and only HDFC Flexi Cap was cross-checked here). The dashboard labels the column "Named Manager" for that reason.
4. **Folder hygiene before submission**: remove or explain the sibling `fund_manager_skill_detector` folder (a Task-1 documentation scaffold for a "US version" with no code or data) and consider flattening the doubly nested `indian_fund_skill_detector/indian_fund_skill_detector`. I did not move or delete either.
5. *Optional extensions* (beyond a correction review): extend the window to 2025 (IIMA and NAV data are available), studentised / cross-sectional bootstrap in the style of Kosowski et al. (2006), persistence on 4-factor alpha, an international-equity factor for Parag Parikh.

---

## 6. External sources used

* IIMA Indian Fama-French-Momentum data library notes: https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/
* AMFI scheme master via mfapi.in (e.g. https://api.mfapi.in/mf/122639); scheme names/categories/plan types checked for all 12 codes.
* HDFC Flexi Cap management history: [Business Standard, 26 Jul 2022](https://www.business-standard.com/amp/article/companies/hdfc-amc-assigns-prashant-jain-s-schemes-to-new-managers-122072501086_1.html),
  [Morningstar India](https://www.morningstar.in/posts/73876/hdfc-flexi-cap-from-prashant-jain-to-roshi-jain.aspx),
  [Business Standard on Roshi Jain's exit](https://www.business-standard.com/amp/companies/people/roshi-jain-quits-hdfc-amc-fund-manager-exit-125112101238_1.html).
* Difficulty of obtaining the NIFTY 500 TRI series programmatically: https://tradingqna.com/t/nifty-500-tri-historical-data/141395 (and the NSE endpoint returned only its web page to scripted requests).


## 7. Follow-up change: evaluator v2.1.0
A separate **Strong Potential Candidate** screening tier was added without changing the strict **Strong Evidence of Skill** criteria. The tier requires positive CAPM alpha with marginal HAC evidence, a positive statistically significant four-factor alpha, positive alpha in at least three of four subperiods, and a circular-block-bootstrap p-value below 0.10. This reflects convergent but not conclusive evidence and is explicitly not a claim of proven skill. Under the current primary IIMA-market specification, PPFAS Flexi Cap is the sole fund meeting this rule. The bootstrap p-value (0.061) does not meet the strict 5% criterion, and its multiple-testing-adjusted evidence remains inconclusive. A separate **Positive but Inconclusive** label now identifies funds with positive CAPM and four-factor alpha plus positive CAPM alpha in at least three of four subperiods, while making clear that these signals do not meet the higher statistical evidence thresholds.


## 8. Follow-up change: evaluator v2.2.0
The preliminary candidate category is now named **Strong Skill Candidate (Preliminary)** and the executive dashboard's primary skill-signal KPI counts confirmed strong-evidence funds plus preliminary strong candidates. A separate KPI continues to show the stricter confirmed-evidence count. For the current Indian universe, PPFAS Flexi Cap is the one preliminary strong skill candidate; confirmed Strong Evidence remains zero because its bootstrap p-value is 0.061 and its CAPM alpha p-value is 0.074. No financial estimates, p-values, or classification thresholds were altered in this display/category-label update.


## Post-review current evaluator state (v2.3.0)

The current evaluator now reports market-level BH q-values for CAPM, bootstrap and four-factor tests and requires q < 0.10 across those families for confirmed Strong Evidence. Current primary categories are 0 confirmed Strong Evidence, 1 Strong Skill Candidate (Preliminary), 5 Positive but Inconclusive, 6 Weak/Inconclusive, and 0 Style/Luck Driven. The sensitivity tables were regenerated under v2.3.0 in the combined v2 release (`python -m src.sensitivity`, about 8 seconds): the primary-proxy block reproduces the main results exactly, and under the two price-index proxies PPFAS Flexi Cap is now a Strong Skill Candidate (Preliminary) instead of the "Strong Evidence of Skill" shown by the earlier file, which pre-dated the Benjamini-Hochberg gate (its four-factor q-value is 0.54). The USA market was brought onto this same rule set in that release.
