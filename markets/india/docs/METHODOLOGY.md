# Theoretical and Methodological Rationale: Indian Fund Manager Skill Detector

*Documents rules version **2.3.0** (the same rule code classifies the US market; see `tests/test_rule_parity.py` at the repository root). Section 8 lists the changes relative to version 1.0.0.*

## 1. Executive Rationale
Measuring fund manager skill is one of the foundational challenges of empirical asset pricing. A positive
return, or raw outperformance of a benchmark, is not evidence of talent. Performance can stem from three origins:
1. **Market exposure (beta)**: a rising market lifts every fund.
2. **Systematic factor tilts**: exposure to known premia (size, value, momentum).
3. **Luck**: sampling variation in heavy-tailed, autocorrelated return series.

The project applies the standard econometric toolkit (CAPM alpha, Carhart four-factor alpha, bootstrap
inference, persistence) to 12 Indian active equity funds, with the IIMA Indian factor library supplying the
market, size, value, momentum and risk-free series.

---

## 2. Data and Return Construction

### 2.1 Returns
* Fund return: R_t = NAV_t / NAV_{t-1} - 1 on the fund's own NAV calendar (Direct Plan - Growth NAVs: dividends are
  reinvested by construction, expenses are already deducted).
* Excess return: R_t - RF_t, where RF is the IIMA risk-free series (percent -> decimal, i.e. divided by 100).
  The IIMA RF accrues per **calendar** day: its Monday value is about three times a weekday value, which matches the
  multi-day NAV return that spans a weekend. The sample mean is 0.019% per trading day = 4.9% per year (x252).

### 2.2 Synchronous-interval alignment
Every analytical return must cover the **same one-step interval** in the fund, the market series and the
factors. Because the three sources have slightly different calendars (special Diwali/Saturday sessions, days
missing in Yahoo Finance, days with no NAV), a row is kept only if the fund's previous NAV date, the benchmark's
previous date and the IIMA previous date coincide. Otherwise a two-day fund return would be regressed on a one-day
market return. This removes 135 of 14,721 fund-day observations (0.9%); the remaining sample is 14,586
observations (1,216 distinct dates, 1,214-1,216 per fund). Missing days are never forward- or back-filled.

### 2.3 Market proxy (the choice that matters most)
| Proxy | What it is | Property |
|---|---|---|
| `iima_market` (**primary**) | IIMA value-weighted market portfolio, return = Mkt-RF + RF | total return (per IIMA's release notes); the *same* market as in the 4-factor model |
| `nifty500_tri` | genuine NIFTY 500 Total Return Index, if `data/raw/raw_benchmark_nifty500_tri.csv` is supplied | the originally intended benchmark; used automatically when present |
| `nifty500_price` | Yahoo `^CRSLDX` | **price index, excludes dividends**: alpha is biased upward by roughly the dividend yield |
| `nifty500_price_plus_assumed_dividend_yield` | price index + assumed 1.3% p.a. | rough TRI stand-in; the yield is an assumption |

Why IIMA is primary: (i) it is a total-return series, (ii) CAPM and 4-factor alphas are then measured against the
same market, so `alpha_reduction` isolates size/value/momentum tilts instead of mixing in a change of market
series, (iii) it needs no unverifiable constant. Caveat: over the sample the IIMA market compounds at about
21.2% per year versus 16.6% for the NIFTY 500 price index (about 3 points above a TRI-like series), because its
universe is broader; it is therefore a **demanding** yardstick for large-cap funds. The classification is shown
under every proxy (`benchmark_sensitivity.csv`, dashboard Leaderboard tab) and should be read as a range.

---

## 3. Single-Factor CAPM Engine (`src/capm.py`)

### 3.1 Model
    R_{i,t} - RF_t = alpha_i + beta_i * (R_{m,t} - RF_t) + epsilon_{i,t}

### 3.2 Estimation and inference
* **Estimator**: OLS via the normal equations (verified against `numpy.linalg.lstsq` and `statsmodels` to ~1e-12).
* **Classical standard errors** sigma^2 (X'X)^{-1} and t_{n-2} p-values are reported (`alpha_pvalue`), but they assume
  i.i.d. errors. The residuals are autocorrelated and non-normal (Durbin-Watson ranges 1.67-2.11 across funds, i.e.
  clearly positive autocorrelation for several; Jarque-Bera rejects normality for every fund).
* **Newey-West HAC** standard errors (Bartlett kernel, lags = floor(4 (n/100)^{2/9}) = 6 for the full sample, small-sample
  factor n/(n-k)) give `alpha_pvalue_hac`, **the p-value used by the classifier**. Implementation checked against
  `statsmodels` (relative difference ~1e-15) and a brute-force loop.
* **Annualisation**: alpha_annual = ((1 + alpha_daily)^252 - 1) x 100.
* The `alpha_*_ci95_*` columns are parametric OLS intervals; the bootstrap provides the robust interval.
* Diagnostics: Durbin-Watson, Jarque-Bera, residual skewness/excess kurtosis.

---

## 4. Circular Block Bootstrap Engine (`src/bootstrap.py`)

### 4.1 Motivation
Classical p-values assume i.i.d. residuals. Bootstrap inference for fund alphas follows Kosowski, Timmermann, Wermers
& White (2006), who bootstrap fund residuals under the null. This project uses the dependence-preserving *circular block*
variant (Politis & Romano, 1992) on paired (fund, market) returns: contiguous blocks preserve short-range dependence
and volatility clustering. The test statistic is the raw alpha (not the studentised t-statistic preferred by Kosowski
et al.), and it is a fund-by-fund test rather than their cross-sectional bootstrap of the best and worst funds.

### 4.2 Algorithm
1. Draw ceil(N/b) block start positions uniformly on 0..N-1 and take b consecutive observations from each (indices wrap
   around), trimming to length N. Fund and market observations stay paired.
2. For each of B = 1,000 replications re-estimate the CAPM intercept (closed form; identical to OLS).
3. **Null-centring**: alpha_null = alpha* - alpha_hat. This imposes H0: alpha = 0.
4. **Two-sided p-value** (the headline `p_value`): (#{|alpha_null| >= |alpha_hat|} + 1) / (B + 1) (smoothed, never exactly 0).
5. **One-sided p-value** (H1: alpha > 0, `p_value_one_sided`): (#{alpha_null >= alpha_hat} + 1) / (B + 1).
6. **95% CI**: 2.5th / 97.5th percentiles of the *uncentred* bootstrap alphas (annualised in `ci_annual_*`).
7. **Direction rule**: `significantly_positive_5pct` = two-sided p < 0.05 **and** alpha_hat > 0. A significantly *negative* alpha
   is evidence of under-performance and earns no credit.
8. **Multiplicity**: `p_value_bh` are Benjamini-Hochberg FDR-adjusted p-values across the 12 funds (informational).

### 4.3 Block length
b = 30 trading days is about 2.8 times the n^{1/3} = 10.7 rule of thumb, chosen to cover multi-week volatility
clustering. It is a judgement, not an optimum: `bootstrap_block_sensitivity.csv` re-runs the test for
b in {1, 5, 10, 20, 30, 60}, and the dashboard plots the p-value against b. Under the original price-index specification
several funds crossed the 5% line as b varied (Quant Small Cap: p = 0.014 at b = 1 but 0.087 at b = 60); under the primary
specification no fund crosses it, but Parag Parikh Flexi Cap sits right at the boundary (p = 0.052-0.080 across b), so conclusions
near the 5% line should not be treated as established. A fixed seed (42) makes every run reproducible.

---

## 5. Carhart 4-Factor Engine (`src/factor_model.py`)

### 5.1 Model
    R_{i,t} - RF_t = alpha_i + b_mkt * MKT_t + b_smb * SMB_t + b_hml * HML_t + b_mom * WML_t + epsilon_{i,t}

All factors come from the IIMA Data Library (survivorship-bias-adjusted daily file; `MF` = market premium, `WML` = momentum),
converted from percent to decimals. OLS with design matrix [1, MKT, SMB, HML, WML], df = n - 5, HAC inference for alpha
(`alpha_pvalue_hac`), 95% OLS intervals for the loadings.

### 5.2 Alpha reduction
    alpha_reduction_pct     = (alpha_CAPM - alpha_4F) / |alpha_CAPM| x 100
    alpha_reduction_pts_annual = alpha_CAPM_annual - alpha_4F_annual      (percentage points per year)

Both are reported. The percentage explodes when |alpha_CAPM| is close to zero (e.g. 2,628% in version 1.0.0 for a fund whose
CAPM alpha was +0.10%), so the classifier only applies it to funds with a positive, at-least-marginally-significant CAPM
alpha (Section 7.3) and the dashboard also shows the stable points-per-year figure.

---

## 6. Persistence Engine (`src/persistence.py`)

* The 1,216 distinct dates are cut into **four non-overlapping chronological subperiods of 304 dates each**:
  P1 2019-01-02 to 2020-04-08, P2 2020-04-09 to 2021-07-02, P3 2021-07-05 to 2022-09-26, P4 2022-09-27 to 2023-12-29.
* CAPM is re-estimated inside each subperiod (HAC lags = 5); metrics per fund: positive-alpha count and rate, significantly
  positive count and rate (HAC p < 0.05), mean/median/std/min/max annualised alpha.
* `alpha_positive_majority` means **strictly more than half** of the subperiods (at least 3 of 4); a 2-of-4 tie is a coin-flip outcome
  and does not count. `alpha_positive_all_periods` means 4 of 4.
* Limitations: four windows give little statistical power, windows are dominated by different market regimes (2020 crash/
  recovery), and persistence is measured on CAPM alpha, so a persistent *style* premium can pass it.

---

## 7. Integrated Evidence Evaluator (`src/evaluator.py`, version 2.3.0)

The evaluator performs **no new estimation**. It reads the four result files, derives Boolean evidence flags, a 0-100 score and a
category, and writes `final_evaluation.csv/json`. Every rule is deterministic; every output row is tagged with the version.

### 7.1 Evidence flags
| Flag | Rule |
|---|---|
| `capm_positive` | CAPM annual alpha > 0 |
| `capm_significant` | `capm_positive` AND HAC p < 0.05 (significantly **positive**) |
| `capm_marginally_significant` | `capm_positive` AND HAC p < 0.10 |
| `capm_significantly_negative` | alpha < 0 AND HAC p < 0.05 (reported as a weakness) |
| `bootstrap_significant` | bootstrap flag (two-sided p < 0.05 AND alpha > 0) OR (p < 0.05 AND `capm_positive`) |
| `bootstrap_ci_positive` | bootstrap CI lower bound > 0 |
| `factor_alpha_positive` / `factor_alpha_significant` | 4-factor alpha > 0 / HAC p < 0.05 |
| `factor_alpha_survives` | both of the above |
| `alpha_materially_reduced` | `alpha_reduction_pct` >= 50 |
| `style_explanation` | `capm_marginally_significant` AND `alpha_materially_reduced` |
| `persistent_positive_majority` / `persistent_all_periods` | at least 3 of 4 / 4 of 4 positive subperiods |

### 7.2 Score (0-100)
* **Performance (0-20)**: CAPM annual alpha <= 0 -> 0; (0,1%) -> 10; [1%,3%) -> 15; >= 3% -> 20.
* **Significance (0-25)**: HAC component 0-15 (p < 0.01 -> 15, < 0.05 -> 10, < 0.10 -> 5; **zero unless alpha > 0**) + bootstrap
  component 0-10 (significant and CI positive -> 10, one of them -> 5).
* **Factor (0-25)**: alpha survives -> 25; positive only -> 10; else 0.
* **Persistence (0-20)**: all 4 periods -> 20; strict majority -> 12; exactly half -> 6; else 0; +3 if the significant-positive rate
  is >= 0.5 (cap 20).
* **Style/luck adjustment (-10..+10)**, only if `capm_marginally_significant`: reduction >= 75% -> -10; >= 50% -> -5; >= 25% -> -2.
  Otherwise +5 if reduction < 25% and the factor alpha survives, else 0.
* Total = clamp(sum, 0, 100). The score ranks evidence strength; the **category** drives the classification.

### 7.3 Categories (first match wins, rules version 2.3.0)
1. **Likely Style/Luck Driven**: `capm_marginally_significant` AND `alpha_reduction_pct` >= 75 AND the factor alpha does not survive. There must be a noteworthy positive CAPM alpha for factor exposures to explain away.
2. **Strong Evidence of Skill**: all five strict criteria: `capm_positive`, `capm_significant`, `bootstrap_significant`, `factor_alpha_survives`, and `persistent_positive_majority`.
3. **Strong Skill Candidate (Preliminary)**: `capm_positive`, `capm_marginally_significant`, `factor_alpha_survives`, `persistent_positive_majority`, and bootstrap p-value < 0.10. This is a screening tier for convergent but not fully robust evidence, not proof of skill and not interchangeable with Strong Evidence.
4. **Moderate Evidence**: any of `capm_significant`; `factor_alpha_survives`; `persistent_positive_majority` AND `capm_marginally_significant`; or `capm_positive` AND `bootstrap_significant`.
5. **Positive but Inconclusive**: CAPM alpha and four-factor alpha are both positive, and CAPM alpha is positive in at least three of four subperiods, but the fund does not meet a higher evidence tier. This is a directional description, not statistical confirmation of skill.
6. **Weak/Inconclusive**: everything else, including cases without enough positive evidence.

### 7.4 Explanations
Explanations are generated by deterministic branching (no free text, no LLM). Language policy: evidence-oriented ("evidence consistent
with ..."), never "proved" or "caused by the manager"; the Strong text carries an explicit no-causality note.

### 7.5 Manager-attribution context
`final_evaluation.csv` adds `lead_manager`, `tenure_start`, `tenure_overlap_pct` (share of the sample window during which the
named manager was in charge) and `attribution_note`. Where the overlap is below 90% the result describes the *scheme*, not that manager.

---

## 8. Changes Relative to Version 1.0.0
| Area | v1.0.0 | v2.0.0 |
|---|---|---|
| CAPM market | NIFTY 500 *price* index labelled "TRI" (dividends omitted; different market from the 4-factor model) | IIMA total-return market (same as 4-factor); price index kept as labelled sensitivity |
| Factor-file fallback | silently generated random "calibrated" factors on download failure | raises an error; synthetic placeholder detected and rejected |
| Return alignment | returns computed per series, then joined (2-day vs 1-day mismatches) | synchronous-interval rule; 135 observations dropped |
| Inference | classical OLS p-values | Newey-West HAC p-values (bootstrap unchanged in spirit) |
| Bootstrap flag | two-sided p only (a significantly negative alpha counted as "significant") | direction-aware flag, one-sided p, BH q-values, block-length sensitivity |
| Persistence "majority" | >= 2 of 4 (a tie) | >= 3 of 4 |
| Significance score | credit for any p < thresholds, whatever the sign | zero unless alpha > 0 |
| Style/Luck label | any positive CAPM alpha with >= 75% reduction (even p = 0.95); also applied to negative alphas | requires positive, p < 0.10 CAPM alpha |
| Moderate rule | non-monotone (adding persistence could demote a fund) | monotone |
| Documentation claims | "TRI" benchmark, "T^{1/3}" block length, "mandated" Direct Plans, an unqualified "common trading days" count, ... | corrected (see `docs/REVIEW_REPORT.md`) |

---

## 9. Methodological Limitations (apply to every result)

1. **Sample and selection**: 12 hand-picked, well-known schemes (survivors with continuous history) - results are biased towards flattering outcomes and do not generalise to the ~1,400 active schemes.
2. **Window**: ~5 years (2019-2023) dominated by the 2020 crash and recovery and by a strong small/mid-cap cycle.
3. **Market proxy**: conclusions depend on it (Section 2.3). A genuine NIFTY 500 TRI is needed for the originally intended benchmark.
4. **Multiple testing**: 12 funds are tested at 5%; the rules do not adjust (BH q-values are shown).
5. **Bootstrap design**: significance near the 5% boundary changes with the block length b; the statistic is the raw (not studentised) alpha and each fund is tested separately.
6. **Factor model completeness**: four factors omit liquidity, quality, sector tilts and India-specific macro exposure; they also cannot capture overseas holdings or large cash/arbitrage positions (e.g. Parag Parikh Flexi Cap).
7. **Manager attribution**: a scheme's record belongs to the AMC process and to every manager who ran it; four of the twelve named managers took charge after the sample started (or only partly overlap it). Manager names and tenure dates are curated, not machine-verified, and some have since changed.
8. **Category history**: SEBI's Flexi Cap category exists only since late 2020; earlier years reflect predecessor mandates.
9. **Expense ratios** are point-in-time disclosures and are not adjusted over the window (NAV returns are already net of fees).
10. **Persistence** uses four windows of ~300 days and CAPM alpha only.
11. **IIMA factor quality**: the series depends on the CMIE Prowess database and can be revised retrospectively.
12. **Statistical significance is not causality**: a significant alpha does not prove that manager decisions caused the outperformance.


### 7.6 Version 2.1.0 candidate tier
The additional candidate tier makes a distinction between the strict standard for confirmed statistical evidence and a fund that merits closer follow-up. It is not a relaxation of the Strong Evidence threshold. The candidate label requires all five pre-defined conditions in section 7.3 and must be accompanied by the bootstrap limitation in the dashboard.


### 7.7 Version 2.2.0 naming and dashboard summary
The former “Strong Potential Candidate” category is renamed **Strong Skill Candidate (Preliminary)** for clearer presentation. The dashboard's “Skill Candidates Identified” KPI counts this preliminary category plus the strict “Strong Evidence of Skill” category; a separate KPI reports confirmed strong evidence only. For the current primary specification, PPFAS Flexi Cap is the one preliminary strong skill candidate. Its CAPM alpha p-value (0.074) and bootstrap p-value (0.061) do not satisfy the strict 5% criteria, so it is not labelled confirmed skill. No financial estimates or decision thresholds changed in v2.2.0.


## Current primary specification (v2.3.0)

- CAPM and four-factor alpha inference use Newey-West HAC standard errors and p-values.
- BH q-values are computed separately for CAPM, bootstrap and four-factor fund-level tests. The confirmed Strong Evidence category requires q < 0.10 across all three families in addition to the unadjusted evidence gates.
- The preliminary PPFAS candidate label is a screening signal and is not a claim of confirmed skill; its raw p-values do not survive BH adjustment.
- Percentage alpha reduction is undefined when CAPM alpha is non-positive or too close to zero. A style/luck label additionally requires positive CAPM alpha with HAC p < 0.10, substantial reduction, and non-surviving factor alpha.
- The primary CAPM market series is the IIMA market portfolio, matching the market factor used by the four-factor model. The NIFTY 500 price index is used only for sensitivity; it is not a total-return index. The genuine NIFTY 500 TRI remains an outstanding data-acquisition improvement.
