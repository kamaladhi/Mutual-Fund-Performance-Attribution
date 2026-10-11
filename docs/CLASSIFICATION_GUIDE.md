# Classification guide

Both markets are classified by **one rule set (evaluation v2.3.0)**. The six classes below are the labels the evaluators write and the CSV
files contain. For readability the dashboard displays "Strong Skill Candidate (Preliminary)" as **Strong** (still described on screen as a
preliminary tier) and displays "Strong Evidence of Skill" and "Likely Style/Luck Driven" only if a fund falls into them. Nothing is merged
and no rule or result is changed for display.

Rules are applied top to bottom; the first one that matches decides the class. Every fund in both markets meets the same thresholds.

| # | Class | Rule (HAC = Newey-West p-value; "positive subperiods" = of 4 equal windows) |
|---|---|---|
| 1 | Likely Style/Luck Driven | CAPM alpha > 0 with HAC p < 0.10, at least 75% of it removed by the four-factor adjustment, and no significant positive four-factor alpha |
| 2 | Strong Evidence of Skill | CAPM alpha > 0 with HAC p < 0.05; bootstrap p < 0.05 (alpha > 0); four-factor alpha > 0 with HAC p < 0.05; alpha positive in at least 3 of 4 subperiods; **and** Benjamini-Hochberg q < 0.10 for the CAPM, bootstrap and four-factor tests |
| 3 | Strong Skill Candidate (Preliminary) | CAPM alpha > 0 with HAC p < 0.10; four-factor alpha > 0 with HAC p < 0.05; at least 3 of 4 positive subperiods; bootstrap p < 0.10. A screening tier, not confirmed skill |
| 4 | Moderate Evidence | Any one of: CAPM alpha > 0 with HAC p < 0.05; a surviving four-factor alpha; at least 3 of 4 positive subperiods with CAPM p < 0.10; CAPM alpha > 0 with a significantly positive bootstrap |
| 5 | Positive but Inconclusive | CAPM and four-factor alphas both positive and at least 3 of 4 positive subperiods, but no higher class applies. Direction only |
| 6 | Weak/Inconclusive | Everything else. Not proof that the manager has no skill |

Details that matter:

- **"Majority" means strictly more than half**: at least 3 of the 4 subperiods. Exactly 2 of 4 is a coin-flip tie and does not count.
- **Only positive alpha earns credit.** A significantly negative alpha earns no significance or skill points and is reported as under-performance.
- **Style/luck needs something to explain**: a positive CAPM alpha that is at least marginally significant. A fund with a negative or
  indistinguishable-from-zero CAPM alpha is never "style/luck driven".
- **Multiple testing**: Benjamini-Hochberg q-values are computed within each market (12 funds) for each of the three tests and gate only
  the strict "Strong Evidence" class. They are shown for every fund.
- **The 0-100 evidence score** ranks funds within a class (performance 20, significance 25, four-factor 25, persistence 20, style/luck
  adjustment -10 to +10). The class, not the score, is the verdict.

## What a class does and does not say

A class summarises the strength of statistical evidence under one specification (CAPM and Carhart four-factor models, Newey-West
inference, a circular block bootstrap and four-subperiod persistence). It is not proof of skill or of its absence, it is not a forecast,
and a positive alpha is an estimate that may reflect luck, an omitted risk factor or the benchmark. With twelve funds and five years of
daily data a single fund's four-factor alpha is only measured to roughly ±4 to ±7 points a year (95%), so a shortage of strong verdicts
is the expected outcome of an honest test.

No category counts are targeted; the distribution is whatever the fixed rules produce from the data. Under alternative Indian
benchmarks the same rules produce a different distribution (Benchmark robustness page).

## Fund-specific caveats shown with the classes

- **US universe**: AMECX (income), PRHSX (healthcare sector) and PAGRX (Permanent Portfolio Aggressive Growth, multi-asset) are exploratory
  comparators, not clean peers. T. Rowe Price Growth Stock is PRGFX and has not been acquired or substituted.
- **India manager attribution**: where a named manager was in charge for under 90% of the window (HDFC Flexi Cap 28%, Nippon Growth
  Mid Cap 54%, SBI Large Cap 75%, Quant Small Cap 83%) the result describes the scheme, not the person. Manager names and tenure dates
  are curated, not machine-verified.
- **PPFAS Flexi Cap** holds overseas equities and large cash/arbitrage positions that India-only factors cannot capture; its alpha
  may partly reflect those exposures.
