# Functional & System Requirements: Fund Manager Skill Detector

## 1. Overview
This document specifies the functional, analytical, dataset, dashboard, testing, and limitation requirements for the Fund Manager Skill Detector system, derived from the core financial engineering specifications.

---

## 2. Detailed Requirements breakdown

### 2.1 Dataset Requirements
- **FR-DAT-01**: Ingestion of historical mutual fund Price / Net Asset Value (NAV) data or return series.
- **FR-DAT-02**: Ingestion of market benchmark index return series (S&P 500 Index `^GSPC`).
- **FR-DAT-03**: Ingestion of risk-free rate series ($R_f$) derived from US Treasury Bills (`^IRX`).
- **FR-DAT-04**: Ingestion of Fama-French multi-factor series (`SMB`, `HML`, `UMD`, `RF`).
- **FR-DAT-05**: Data validation pipeline checking date alignment, frequency consistency, missing date handling, and excess return calculation ($R_{i,t} - R_{f,t}$).
- **FR-DAT-06**: Support for configurable date ranges and sampling frequencies (daily primary, monthly secondary).

### 2.2 Descriptive Statistics Requirements
- **FR-STA-01**: Compute annualized return, annualized volatility, Sharpe ratio, and Sortino ratio for all funds and benchmarks.
- **FR-STA-02**: Compute higher distribution moments: skewness and excess kurtosis.
- **FR-STA-03**: Compute maximum drawdown and drawdown duration.
- **FR-STA-04**: Perform Jarque-Bera normality test on returns and OLS residuals.
- **FR-STA-05**: Perform Ljung-Box autocorrelation test on return series and squared return series (volatility clustering check).

### 2.3 CAPM Analysis Requirements
- **FR-CAP-01**: Estimate single-factor OLS linear regression:
  $$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \epsilon_{i,t}$$
- **FR-CAP-02**: Output parameter estimates: Jensen's Alpha ($\alpha$), Beta ($\beta$), $R^2$, Adjusted $R^2$, and Residual Standard Error.
- **FR-CAP-03**: Compute standard parametric hypothesis test metrics: OLS standard error, t-statistic, and parametric p-value for $H_0: \alpha = 0$.

### 2.4 Block Bootstrap Significance Testing Requirements
- **FR-BOO-01**: Implement non-parametric Circular Block Bootstrap (CBB) resampling of residuals/returns to preserve temporal dependence and conditional heteroskedasticity.
- **FR-BOO-02**: Determine/implement block size selection mechanism (e.g., $b \approx T^{1/3}$ or fixed block length).
- **FR-BOO-03**: Generate empirical null distribution under $H_0: \alpha \le 0$ across $B \ge 1,000$ bootstrap iterations.
- **FR-BOO-04**: Calculate empirical non-parametric bootstrap p-value ($p_{boot}$).
- **FR-BOO-05**: Calculate bootstrap 95% confidence intervals using Percentile and BCa methods.

### 2.5 APT / Multi-Factor Analysis Requirements
- **FR-APT-01**: Fit Fama-French 3-Factor model ($SMB, HML$).
- **FR-APT-02**: Fit Carhart 4-Factor model (adding Momentum $UMD$).
- **FR-APT-03**: Compare single-factor CAPM $\alpha$ with multi-factor $\alpha_{APT}$ to determine whether positive CAPM alpha is absorbed by factor tilts.
- **FR-APT-04**: Compute factor attribution weights and relative contribution of each factor to overall excess returns.

### 2.6 Performance Persistence Requirements
- **FR-PER-01**: Partition the evaluation sample into non-overlapping sub-periods $T_1$ and $T_2$.
- **FR-PER-02**: Compute Spearman cross-sectional rank correlation of alphas ($\alpha_{T1}$ vs $\alpha_{T2}$).
- **FR-PER-03**: Construct $2 \times 2$ contingency table categorizing funds into Winner-Winner (WW), Winner-Loser (WL), Loser-Winner (LW), and Loser-Loser (LL).
- **FR-PER-04**: Perform Chi-Square ($\chi^2$) test of independence and compute Odds Ratio to test intertemporal performance persistence.

### 2.7 Final Skill & Confidence Evaluation Requirements
- **FR-EVA-01**: Synthesize single-factor CAPM alpha, bootstrap p-value ($p_{boot}$), multi-factor alpha ($\alpha_{APT}$), and persistence classification into a unified Skill Score.
- **FR-EVA-02**: Classify managers into four unambiguous tiers: Genuine Skill, Luck / Factor Tilt, Unskilled / Underperforming, and Inconclusive.

### 2.8 Streamlit Dashboard Requirements
- **FR-DSH-01**: Interactive UI displaying dataset overview, summary leaderboard, and metric distribution filters.
- **FR-DSH-02**: CAPM vs Multi-Factor detailed tab showing OLS regression plots, scatter diagrams, and parameter tables.
- **FR-DSH-03**: Bootstrap Explorer tab rendering dynamic histograms of bootstrap alpha distribution and percentile confidence bounds.
- **FR-DSH-04**: APT Factor Attribution tab featuring factor loading charts.
- **FR-DSH-05**: Performance Persistence tab featuring rank scatter plots and transition heatmaps.
- **FR-DSH-06**: Individual Fund Manager Deep-Dive report generator.

### 2.9 Testing Requirements
- **FR-TST-01**: Unit tests for returns computation and date alignment.
- **FR-TST-02**: Unit tests verifying CAPM OLS parameter correctness.
- **FR-TST-03**: Unit tests for Circular Block Bootstrap index generation.
- **FR-TST-04**: Unit tests for multi-factor regression coefficients.
- **FR-TST-05**: Unit tests for contingency matrix creation and Chi-Square statistics.
- **FR-TST-06**: Unit tests for skill evaluation decision matrix rules.
