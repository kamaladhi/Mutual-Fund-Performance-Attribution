# Theoretical & Methodological Rationale: Fund Manager Skill Detector (US market)

*Documents rules version **2.3.0**, the same rule set that classifies the Indian market (`markets/india/docs/METHODOLOGY.md` section 7 is the
parallel text; `tests/test_rule_parity.py` at the repository root keeps the rule code identical).*

## 1. Executive Rationale
Measuring fund manager skill is one of the foundational challenges of empirical asset pricing. A simple positive return or raw
outperformance over a benchmark is insufficient proof of manager talent. Performance can stem from three distinct origins:
1. **Market exposure ($\beta$)**: rising market tides lifting all assets.
2. **Systematic factor tilts**: exposure to known risk premia (small-cap bias, value bias, momentum bias).
3. **Luck**: random sampling variation in heavy-tailed, autocorrelated return distributions.

---

## 2. Data and return construction
* Fund return: $R_t = P_t / P_{t-1} - 1$ from Yahoo Finance **adjusted close** (distributions reinvested; fees already deducted).
* Excess return: $R_t - RF_t$, with RF the Ken French daily risk-free rate (percent divided by 100).
* Market: the Fama-French `Mkt-RF` market excess return, used for **both** CAPM and the four-factor model so that the two alphas are measured
  against the same market. The S&P 500 price index (`^GSPC`) is kept only as descriptive context: it excludes dividends.
* Factors: Ken French daily SMB, HML and momentum, all converted from percent to decimals. Dates are aligned by strict inner join (no fill).

## 3. Single-factor CAPM engine (`src/capm.py`)
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \epsilon_{i,t}$$
OLS point estimates; classical standard errors are kept as diagnostics. **Inference used by the classifier is Newey-West HAC**: Bartlett kernel,
lags $= \lfloor 4 (n/100)^{2/9} \rfloor$ (7 for the full sample), small-sample factor $n/(n-k)$, $t_{n-2}$ p-values. Alpha is annualised by
compounding: $\alpha_{annual} = (1+\alpha_{daily})^{252} - 1$.

## 4. Circular block bootstrap (`src/bootstrap.py`)
Paired (fund, market) resampling in circular blocks of $b = 30$ trading days, $B = 1{,}000$ replications, seed 42. The resampled alphas are
re-centred on zero to impose $H_0: \alpha = 0$:
$$\tilde{\alpha}^{*,b} = \hat{\alpha}^{*,b} - \hat{\alpha}_{obs}$$
The reported p-value is **two-sided** and smoothed, $(k+1)/(B+1)$ with $k = \#\{|\tilde\alpha^{*}| \ge |\hat\alpha_{obs}|\}$; a significantly
*negative* alpha is never counted as evidence of skill (the evaluator's bootstrap flag requires a positive CAPM alpha). The 95% interval is the
percentile interval of the uncentred resampled alphas. The statistic is the raw (not studentised) alpha, each fund is tested separately, and
results near the 5% line depend on $b$.

## 5. Carhart four-factor engine (`src/factor_model.py`)
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_{mkt}(Mkt-RF)_t + \beta_{smb} SMB_t + \beta_{hml} HML_t + \beta_{mom} Mom_t + \epsilon_{i,t}$$
OLS with HAC inference for alpha (same lag rule, $t_{n-5}$ p-values). Alpha reduction: $(\alpha_{CAPM} - \alpha_{4F}) / \alpha_{CAPM}$, reported only when
the CAPM alpha is positive and not numerically close to zero (daily alpha above $10^{-5}$); otherwise it is undefined.

## 6. Performance persistence (`src/persistence.py`)
The 1,257 trading days (`2019-01-03` to `2023-12-29`) are cut into **four consecutive non-overlapping subperiods** of 314, 314, 314 and 315 days.
CAPM is re-estimated inside each (HAC p-values for significance). Metrics: positive-alpha count and rate, significantly positive count and rate, mean,
median, standard deviation, minimum and maximum of the annualised subperiod alphas.

* `alpha_positive_all_periods`: positive in 4 of 4.
* `alpha_positive_majority`: **strictly more than half**, i.e. at least 3 of 4. A 2-of-4 tie is a coin-flip outcome and does not count.
  (Earlier US releases counted a tie, which made five funds look "persistent" by chance; the rule now matches the Indian engine.)
* Limitations: four windows give little statistical power, windows are dominated by different market regimes (the 2020 crash and recovery),
  and persistence is measured on CAPM alpha, so a persistent *style* premium can pass.

---

## 7. Integrated evidence evaluator (`src/evaluator.py`, version 2.3.0)
The evaluator performs **no new estimation**. It reads the four result files, derives Boolean evidence flags, a 0-100 score and a class, and
writes `final_evaluation.csv/json`. Every rule is deterministic; every row is tagged with the version. Benjamini-Hochberg q-values are computed
across the 12 funds separately for the CAPM, bootstrap and four-factor p-values.

### 7.1 Evidence flags
| Flag | Rule |
|---|---|
| `capm_positive` | CAPM annual alpha > 0 |
| `capm_significant` | `capm_positive` AND HAC p < 0.05 (significantly **positive**) |
| `capm_marginally_significant` | `capm_positive` AND HAC p < 0.10 |
| `capm_significantly_negative` | alpha < 0 AND HAC p < 0.05 (reported as a weakness; earns no credit) |
| `bootstrap_significant` | `capm_positive` AND (bootstrap flag or two-sided p < 0.05) |
| `bootstrap_ci_positive` | bootstrap CI lower bound > 0 |
| `factor_alpha_positive` / `factor_alpha_significant` | four-factor alpha > 0 / HAC p < 0.05 |
| `factor_alpha_survives` | both of the above |
| `alpha_materially_reduced` | `alpha_reduction_pct` >= 50 |
| `style_explanation` | `capm_marginally_significant` AND `alpha_materially_reduced` |
| `persistent_positive_majority` / `persistent_all_periods` | at least 3 of 4 / 4 of 4 positive subperiods |

### 7.2 Score (0-100)
* **Performance (0-20)**: CAPM annual alpha <= 0 -> 0; (0,1%) -> 10; [1%,3%) -> 15; >= 3% -> 20.
* **Significance (0-25)**: HAC component 0-15 (p < 0.01 -> 15, < 0.05 -> 10, < 0.10 -> 5; **zero unless alpha > 0**) + bootstrap component 0-10
  (significant and CI positive -> 10, one of them -> 5).
* **Factor (0-25)**: alpha survives -> 25; positive only -> 10; else 0.
* **Persistence (0-20)**: all 4 periods -> 20; strict majority -> 12; exactly half -> 6; else 0; +3 if the significant-positive rate is >= 0.5 (cap 20).
* **Style/luck adjustment (-10..+10)**, only if `capm_marginally_significant`: reduction >= 75% -> -10; >= 50% -> -5; >= 25% -> -2.
  Otherwise +5 if reduction < 25% and the factor alpha survives, else 0.
* Total = clamp(sum, 0, 100). The score ranks evidence strength; the **class** is the verdict. The weights are expert judgement, not calibrated.

### 7.3 Classes (first match wins)
1. **Likely Style/Luck Driven**: `capm_marginally_significant` AND `alpha_reduction_pct` >= 75 AND the factor alpha does not survive.
2. **Strong Evidence of Skill**: `capm_positive`, `capm_significant`, `bootstrap_significant`, `factor_alpha_survives`, `persistent_positive_majority`,
   AND Benjamini-Hochberg q < 0.10 for the CAPM, four-factor and bootstrap tests.
3. **Strong Skill Candidate (Preliminary)**: `capm_positive`, `capm_marginally_significant`, `factor_alpha_survives`, `persistent_positive_majority`,
   and bootstrap p < 0.10. A screening tier, not proof of skill.
4. **Moderate Evidence**: any of `capm_significant`; `factor_alpha_survives`; `persistent_positive_majority` AND `capm_marginally_significant`;
   `capm_positive` AND `bootstrap_significant`.
5. **Positive but Inconclusive**: CAPM and four-factor alphas both positive and a strict persistence majority, but no higher class applies.
6. **Weak/Inconclusive**: everything else.

Explanations are generated by deterministic branching (no free text, no LLM) and never claim that skill was proved or caused the outperformance.

---

## 8. Methodological limitations (apply to every result)
1. **Sample size**: only 12 funds are analysed; results do not generalise.
2. **Window**: about 5 years (2019-2023), dominated by the 2020 crash and recovery.
3. **Persistence**: four subperiods of about 314 days and CAPM alpha only.
4. **Bootstrap**: block length (30 days) is a judgement; the statistic is the raw alpha; each fund is tested separately.
5. **Factor model completeness**: four factors omit quality, low volatility, liquidity and sector tilts.
6. **Return measurement**: adjusted NAV closes from Yahoo Finance; adjustments can be imperfect.
7. **Survivorship / selection**: the universe was chosen by hand from well-known surviving funds.
8. **Universe comparability**: AMECX (income), PRHSX (healthcare sector) and PAGRX (multi-asset) are exploratory non-peers. T. Rowe Price Growth Stock is
   PRGFX and has not been acquired.
9. **Active-management status**: funds are `UNVERIFIED_CANDIDATE`; active management is not independently confirmed.
10. **Multiple testing**: 12 funds are tested at 5%; q-values are reported and gate only the strict class.
11. **Statistical power**: a single fund's four-factor alpha has a 95% half-width of about 4 points a year, so a modest true alpha is rarely detected.
12. **Significance is not causality**, and positive alpha is consistent with skill, luck, an omitted factor or data artefacts.
13. **Benchmark**: the S&P 500 price index excludes dividends and is descriptive only; the Fama-French market is the estimation market.

Reference: Kenneth R. French Data Library, *Description of Fama/French Factors*, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html.
S&P Dow Jones Indices distinguishes the S&P 500 price-return index from its total-return versions: https://www.spglobal.com/spdji/en/indices/equity/sp-500/.
