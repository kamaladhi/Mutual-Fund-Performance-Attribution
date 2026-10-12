# 🎓 End-Semester Project & Viva Voce Master Guide
## Financial Engineering Elective: Fund Manager Skill & Factor Attribution Engine

---

## 🎯 1. The 30-Second Elevator Pitch (Opening Your Presentation)

> *"Good morning, Professors. In asset management, investors pay active mutual fund managers significant fees—often 1.5% to 2.5% of their corpus annually—under the belief that managers possess superior stock-picking skill, known as **alpha**.*
>
> *Our project asks a fundamental quantitative finance question:* **Are mutual fund managers genuinely generating skill-based alpha, or are they simply riding market beta, size, value, and momentum styles—and charging active fees for passive factor exposure?**
>
> *To answer this, we built **AlphaLens**: an institutional-grade Quantitative Financial Engineering platform. We implemented a unified **5-Pillar Econometric Framework** across both Indian and US markets, augmented it with **Explainable AI (XGBoost + TreeSHAP)** to capture non-linear factor tilts, built a **Live On-Demand AMFI Ingestion Engine** covering 2,500+ Indian schemes, and modeled the long-term **Fee Leakage** that silently erodes investor wealth. Today, we will walk you through our data preprocessing, statistical methodology, empirical findings, and live platform demo."*

---

## 📊 2. Slide-by-Slide PPT Blueprint & Speaking Notes

| Slide # | Slide Title | Visual Elements | What You Must Say (Speaking Script) |
|---|---|---|---|
| **Slide 1** | **Title Slide: Fund Manager Skill & Factor Attribution Engine** | Title, Team members, Elective: Financial Engineering, Academic Supervisor | "Welcome everyone. Our project combines classical asset pricing econometrics with modern explainable machine learning to evaluate mutual fund outperformance." |
| **Slide 2** | **The Core Problem: Alpha vs. Beta & Fee Drag** | 2-column contrast: Active Fee Promise vs. Academic Reality (SPIVA data) | "Globally and in India, SPIVA reports show that over 70-80% of active large-cap funds fail to beat their benchmark over 5-year horizons. Yet investors pay high Total Expense Ratios (TER). We set out to rigorously test whether any flagship fund exhibits statistically significant skill." |
| **Slide 3** | **Data Engineering & Preprocessing Pipeline** | Flowchart of NAV Ingestion ➔ Date Alignment ➔ Risk-Free Rate De-annualization ➔ Excess Return Calculation | "Raw financial data is dirty and non-synchronous. We fetched daily NAVs, synchronized Indian (NSE) and US (NYSE) trading calendars, de-annualized RBI and US Treasury yields, and ensured our factor benchmarks are Total Return Series to prevent dividend distortion." |
| **Slide 4** | **Why Compare Indian and US Markets?** | Side-by-side market characteristics table (Efficiency, Retail participation, Alpha availability) | "The US market (EMH Strong-to-Semi-Strong) serves as a mature benchmark where factor indexing dominates. The Indian market (developing, semi-strong) is often hypothesized to offer greater inefficiencies for active managers to exploit. Applying the identical econometric rule-set to both reveals structural market efficiency differences." |
| **Slide 5** | **The 5-Pillar Econometric Engine** | Diagram of 5 tests: CAPM, Carhart 4-Factor, Treynor-Mazuy, Bootstrap, FDR | "We did not rely on simple Sharpe ratios. We implemented a 5-pillar statistical engine: CAPM for raw alpha, Carhart 4-Factor to strip out Size, Value, and Momentum, Treynor-Mazuy quadratic regression for market timing, a 1,000-iteration Kosowski residual bootstrap for non-normal returns, and Benjamini-Hochberg False Discovery Rate control." |
| **Slide 6** | **Empirical Findings & The 'Luck/Fee' Reality** | Summary Classification Table (India vs USA) | "Out of 24 analyzed flagship funds over 2019–2023, zero funds pass the strict FDR threshold (q < 0.10). Even Parag Parikh Flexi Cap, which shows positive 4-factor alpha, falls into the screening tier once multiple testing and cash/overseas holdings are factored in. Most apparent alpha is explained by style factor loading." |
| **Slide 7** | **Novelty 1: Explainable AI with XGBoost & TreeSHAP** | Factor Radar Chart & Return Waterfall Chart | "Classical linear regression assumes fixed factor betas. But fund managers rotate styles during market crashes and rate hikes. We trained an XGBoost model and used TreeSHAP (Shapley Additive Explanations) to decompose exact basis-point contributions from Size, Value, Momentum, and Market, showing non-linear active risk." |
| **Slide 8** | **Novelty 2: Fee Leakage & Compounding Wealth Erosion** | 25-Year Wealth Path Chart (Direct vs Passive ETF vs Gross) | "Financial engineering is about real investor outcomes. We translated abstract p-values into rupees. A 1.5% expense ratio compounded over 25 years erodes between 35% and 45% of an investor's terminal wealth. We built an interactive simulator to visualize this fee leakage." |
| **Slide 9** | **Novelty 3: Live AMFI Ingestion & Institutional Tear-Sheet** | Screenshot of Live Search & Downloaded PDF Report | "Rather than a static analysis of 24 funds, we turned our project into a live SaaS-grade product. Users can search any of 2,500+ Indian mutual funds in real time, fetch live NAVs via AMFI, run the entire 5-pillar analysis in memory, and export an Institutional Due Diligence Tear-Sheet as a PDF." |
| **Slide 10** | **Conclusion, Limitations & Future Work** | Summary points, Academic citations (Carhart 1997, Kosowski 2006, Lundberg 2017) | "In conclusion, skill in mutual fund management is exceptionally rare; factor exposure and fees dominate. Future work will extend this framework to multi-asset schemes, sector tilts, and high-frequency factor models. Thank you, and we welcome your questions." |

---

## 🛠️ 3. Deep Dive: Data Preprocessing Explained Simply

When professors ask: *"Explain your data preprocessing steps in detail"*, walk through these 5 exact steps:

```text
[Raw Fund NAVs] + [Market Benchmark Index] + [Risk-Free Yields] + [Fama-French Factor Files]
                                        │
                                        ▼
             Step 1: Daily Percentage Return Calculation
                     R_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}
                                        │
                                        ▼
             Step 2: Trading Day Calendar Synchronization
                     Inner-join on Business Dates (drop holidays/weekends)
                                        │
                                        ▼
             Step 3: Risk-Free Rate De-annualization
                     R_{f, daily} = (1 + R_{f, annual})^(1/252) - 1
                                        │
                                        ▼
             Step 4: Total Return Index (TRI) Adjustment
                     Use dividend-reinvested benchmark (avoids artificial alpha)
                                        │
                                        ▼
             Step 5: Factor Data Alignment & Orthogonality Verification
                     Align SMB, HML, WML series from IIMA / French Data Library
```

### Why each step matters mathematically:
1. **Compounding vs. Arithmetic Returns**: We use daily percentage returns $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ rather than log returns because portfolio returns are linearly additive across assets.
2. **Calendar Mismatches**: Indian equity markets celebrate festivals (Diwali, Holi) that US markets do not; US markets celebrate Thanksgiving, Independence Day. Merging them by date without an inner-join would create `NaN` values that invalidate OLS estimates.
3. **De-annualizing Risk-Free Rates**: 91-day T-Bill yields and 3-month US Treasury yields are quoted on an annualized percentage basis (e.g., 6.5%). If used directly against daily fund returns (~0.05%), the subtraction $R_t - R_f$ would result in a massive negative distortion. We convert annual rates to daily compounding rates using:
   $$R_{f,\text{daily}} = (1 + R_{f,\text{annual}})^{1/252} - 1$$
4. **Total Return Index (TRI) vs Price Index (PR)**: Mutual fund NAVs accumulate reinvested dividends. If a fund is benchmarked against a Price Return Index (which ignores dividends), the fund appears to have 1.5%–2.0% artificial "alpha" that is simply dividend yield! We enforce TRI benchmarks (IIMA for India, CRSP Value-Weighted for US).

---

## 🌍 4. Why Indian and US Datasets are Both Suited for this Project

### Why this is a brilliant comparison:
1. **Market Efficiency Spectrum**: The US market is characterized by high institutional ownership (>70%) and near-efficient pricing. India has rising retail participation, significant retail order flow, and greater dispersion in mid/small caps. Comparing them tests whether developing markets truly allow active managers to outperform.
2. **Benchmark Integrity**: The US market has the gold-standard Ken French Data Library with daily factor series since 1926. India has the Indian Institute of Management Ahmedabad (IIMA) Data Library (Agarwalla, Jacob, Varma) providing the academically validated 4-factor series.
3. **Identical Rule Parity**: The key innovation of this project is that **the same mathematical classifier runs on both**. We do not relax thresholds for India and tighten them for the US.

### The trap to avoid (Professor Gotcha):
- *Prof: "Can you directly compare an Indian fund's alpha of +4% with a US fund's alpha of +1%?"*
- *Winning Answer:* **"No, sir/ma'am. We do NOT rank them in a combined league table.** India's nominal risk-free rate is ~6.5% with higher expected market risk premia and inflation, whereas US rates were 0-5%. Furthermore, market betas represent domestic market risks. What we compare is **statistical skill evidence**—whether active managers in either market systematically survive multi-factor adjustment and false discovery rate corrections."

---

## 🔬 5. Econometric Models vs. Machine Learning

### Classical Econometrics (Why We Need It):
- **Causality & Statistical Inference**: OLS with Newey-West standard errors gives $p$-values, $t$-statistics, and confidence intervals under standard asset pricing theory.
- **Kosowski (2006) Residual Bootstrap**: Solves the problem that daily financial returns have fat tails (leptokurtosis) and non-normal residuals. By generating 1,000 synthetic return paths under the null hypothesis of $\alpha = 0$, we get an empirical $p$-value that does not assume a normal Bell curve.
- **Benjamini-Hochberg FDR**: If you test 100 funds at a 5% significance level, 5 funds will look "skilled" purely by random luck. FDR adjusts $q$-values to ensure false discoveries are limited to <10%.

### Why Explainable AI (XGBoost + SHAP) Adds Novelty:
- **Non-Linear Dynamics**: A linear regression fits a single line: $y = \alpha + \beta x$. But in reality, a manager might be defensive ($\beta = 0.7$) in market downturns and aggressive ($\beta = 1.3$) during rallies. XGBoost captures these non-linear regime shifts.
- **TreeSHAP (Lundberg & Lee, 2017)**: Grounded in cooperative game theory (Lloyd Shapley, Nobel Laureate). It fairly distributes the predicted return across the factor features, providing an intuitive breakdown of where returns originated.

---

## 💡 6. Top 10 Tough Viva Questions & Flawless Answers

### Q1: "Why did you use the Carhart 4-Factor model instead of just CAPM or Fama-French 3-Factor?"
> **Answer**: *"CAPM only controls for overall market risk ($\beta$). Fama-French (1993) added Size (SMB) and Value (HML), but Mark Carhart (1997) demonstrated that many mutual funds exhibit persistence purely because they ride price momentum (buying recent winners). Adding the WML (Winners Minus Losers) factor prevents a momentum-chasing manager from being falsely labeled as having genuine stock-picking skill."*

### Q2: "What is the difference between statistical significance and economic significance here?"
> **Answer**: *"A fund might have a 4-factor alpha of +0.30% per year with a $p$-value of 0.04 in a large sample. While statistically significant, economically it is negligible and completely consumed by the fund's 1.50% expense ratio and transaction costs. True skill requires both economic significance (alpha > fee hurdle) and statistical robustness."*

### Q3: "What is Newey-West HAC and why did you use it instead of ordinary OLS standard errors?"
> **Answer**: *"Ordinary Least Squares (OLS) assumes regression residuals are independent and identically distributed (i.i.d.). Daily financial returns violate this due to volatility clustering (heteroskedasticity) and return autocorrelation. Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors adjust the covariance matrix so our $t$-statistics and $p$-values are not artificially inflated."*

### Q4: "How does Treynor-Mazuy test for market timing?"
> **Answer**: *"Treynor and Mazuy added a quadratic market excess return term, $\gamma (R_m - R_f)^2$, to the CAPM equation. If a manager increases portfolio beta before market surges and lowers beta before crashes, the return profile curves upward like a convex parabola. A statistically significant positive $\gamma$ indicates market timing ability. In our sample, almost all funds showed $\gamma \approx 0$ or negative, showing managers do not successfully time markets."*

### Q5: "What is survivorship bias and how does it affect your sample?"
> **Answer**: *"Survivorship bias occurs when poorly performing funds are liquidated or merged into larger funds and thus disappear from historical databases, leaving only surviving, successful funds. Because our static sample analyzed established flagship funds over 2019–2023, there is inherent selection and survivorship bias. We openly document this in our limitations. However, our new Live Ingestion module helps alleviate this by allowing evaluation of any currently active scheme on demand."*

### Q6: "Why did you use TreeSHAP instead of KernelSHAP or feature importances?"
> **Answer**: *"Standard feature importances (e.g., Gini impurity or split gain) only show which feature was used most, but not whether it pushed returns up or down, and they suffer from feature correlation bias. KernelSHAP is model-agnostic but computationally slow ($O(2^M)$). TreeSHAP computes exact Shapley values in polynomial time $O(TLD^2)$ specifically for tree ensembles, guaranteeing mathematical additivity: the sum of SHAP values plus base value exactly equals the predicted fund return."*

### Q7: "Why didn't any fund pass the strict 'Strong Evidence of Skill' category?"
> **Answer**: *"Because our classification v2.3.0 applies rigorous scientific standards: it requires positive CAPM and 4-factor alpha, 3-out-of-4 year persistence, bootstrap $p < 0.10$, and crucially, Benjamini-Hochberg False Discovery Rate $q < 0.10$. When testing 12 funds, the statistical power over 5 years is limited ($\pm 4\%$ confidence intervals), and the multiple-testing penalty rightfully prevents false positives. This matches findings in top literature like Fama & French (2010)."*

### Q8: "What was Parag Parikh Flexi Cap's classification and why?"
> **Answer**: *"Parag Parikh Flexi Cap achieved 'Strong Skill Candidate (Preliminary)' under the primary IIMA benchmark. It demonstrated positive four-factor alpha and multi-year persistence. However, it did not reach the top strict tier because: (1) multiple-testing $q$-value remained above 0.10, and (2) the scheme holds international equities (e.g., Alphabet, Microsoft) and high cash reserves, which are not captured by a domestic 4-factor model, meaning part of the 'alpha' is geographic diversification rather than pure Indian stock-picking skill."*

### Q9: "How does your Fee Leakage Calculator model wealth erosion?"
> **Answer**: *"We simulate the compounding equation $W_T = W_0 \prod_{t=1}^T (1 + R_t - \text{TER})$. By running parallel trajectories with Gross Returns, Net Active Returns (with actual fund TER), and Passive Benchmark ETF Returns (with a low 0.10% TER), we show that compounding turns a seemingly small 1.5% fee difference into a 35–45% reduction in terminal accumulated wealth over 25–30 years."*

### Q10: "If you had 6 more months, how would you expand this project?"
> **Answer**: *"We would implement: (1) Style drift detection using rolling Kalman filters, (2) Holdings-based factor attribution using actual portfolio disclosure sheets rather than returns-based style analysis, and (3) Dynamic benchmark selection via automated peer-group clustering (e.g., using HDBSCAN on factor betas)."*

---

## 💻 7. Recommended 5-Minute Live Demo Flow

When giving a live demo to the evaluation committee:
1. **Launch App**: Open `http://localhost:8501`.
2. **Start at Overview**: Show the 6 Evidence Tiers and the summary counts across India and USA. Point out the zero false-positive rate.
3. **Switch to 'Live Fund Analyser 🚀'**:
   - In the search bar, type `"Parag Parikh"` or `"HDFC Flexi Cap"` or `"Axis Small Cap"`.
   - Select the fund and click **"Run Live Quantitative Analysis"**.
   - Show how the app pulls real-time NAVs from AMFI, aligns dates, and executes the 5 econometric regressions in memory in seconds.
   - Show the **Skill Verdict Badge** and the **Factor Exposure Table**.
4. **Show Explainable AI**:
   - Scroll to the **XGBoost SHAP Factor Radar** and explain the active risk loading.
   - Point out the **252-day Rolling Alpha & Beta** chart showing how beta fluctuated during COVID-19 (March 2020) and interest rate hiking cycles.
5. **Show Fee Leakage**:
   - Slide the investment horizon to 25 years and show the ₹ Rupee Wealth Drag.
6. **Download Tear-Sheet**:
   - Click **"Generate Institutional PDF Report"** and open the generated institutional due diligence tear-sheet.
