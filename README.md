# 📈 AlphaLens: Fund Manager Skill & Factor Attribution Engine
### Quantitative Financial Engineering & Explainable AI Platform (India & US Mutual Funds)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50+-FF4B4B.svg)](https://streamlit.io/)
[![Econometrics](https://img.shields.io/badge/Econometrics-Carhart%204--Factor%20%7C%20Bootstrap%20%7C%20FDR-success.svg)]()
[![Explainable AI](https://img.shields.io/badge/XAI-XGBoost%20%7C%20TreeSHAP-orange.svg)]()
[![Live Ingestion](https://img.shields.io/badge/Data-AMFI%20(2500+%20Funds)%20%7C%20Yahoo%20Finance-9cf.svg)]()
[![Institutional Tear-Sheet](https://img.shields.io/badge/Export-PDF%20Due%20Diligence-purple.svg)]()

---

## 🎯 Executive Overview

Do flagship mutual funds generate genuine investment skill (true alpha), or are returns merely compensation for hidden factor exposures (market, size, value, momentum) and luck? 

**AlphaLens** is an end-to-end Quantitative Financial Engineering and Explainable AI (XAI) platform designed to separate **manager skill** from **factor beta and fee drag**. Built under rigorous econometric standards, it provides:
1. **A Unified 5-Pillar Econometric Skill Detection Framework** with rule parity across Indian (AMFI/IIMA) and US (CRSP/Fama-French) equities.
2. **Explainable AI Factor Attribution** using **XGBoost and TreeSHAP (Shapley Additive Explanations)** to capture non-linear market regimes and dynamic factor tilts.
3. **Live On-Demand Fund Ingestion**: Live search and sub-second analysis of **2,500+ Indian mutual funds via AMFI API** and any US ticker via Yahoo Finance.
4. **Investor Fee Leakage & Wealth Erosion Calculator**: Quantifies in currency terms (₹ INR / $ USD) the real drag of Total Expense Ratio (TER) and underperformance compounded over 5 to 30 years.
5. **Institutional Due Diligence Tear-Sheet**: 1-click automated PDF export generating institutional-grade evaluation sheets for asset allocators and investment committees.

---

## 🏛️ System Architecture

```text
                               ┌───────────────────────────────────────────────────┐
                               │       Live Fund Search & Real-Time Ingestion      │
                               │   AMFI API (2,500+ Indian Funds) & Yahoo Finance  │
                               └─────────────────────────┬─────────────────────────┘
                                                         │
                                                         ▼
                               ┌───────────────────────────────────────────────────┐
                               │           Rigorous Data Preprocessing             │
                               │  Trading Day Alignment, De-annualized Rf, Merges  │
                               └─────────────────────────┬─────────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
        ┌───────────────────────────────────┐                           ┌──────────────────────────────────┐
        │   5-Pillar Econometric Engine     │                           │    Explainable AI (XAI) Engine   │
        │ • CAPM Alpha & Beta (OLS)         │                           │ • Non-linear Multi-Factor Model  │
        │ • Carhart 4-Factor (SMB, HML, WML)│                           │ • XGBoost Gradient Boosting Reg. │
        │ • Treynor-Mazuy Market Timing (γ) │                           │ • TreeSHAP Shapley Attribution   │
        │ • 1,000-Iter Residual Bootstrap   │                           │ • Dynamic Factor Radar Charts    │
        │ • Benjamini-Hochberg FDR (q-val)  │                           │ • Return Contribution Waterfalls │
        └─────────────────┬─────────────────┘                           └────────────────┬─────────────────┘
                          │                                                              │
                          └──────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
                               ┌───────────────────────────────────────────────────┐
                               │       6-Tier Objective Skill Classifier           │
                               │ Strong Evidence | Candidate | Moderate | Luck/Fee │
                               └─────────────────────────┬─────────────────────────┘
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         ▼                               ▼                               ▼
        ┌───────────────────────────────────┐ ┌──────────────────────────────────┐ ┌───────────────────────────┐
        │ Interactive Streamlit Platform    │ │ Fee Leakage & Wealth Simulator   │ │ Institutional Tear-Sheet  │
        │ 7 Analytical Views + Live Analyser│ │ ₹ / $ 30-Year Compounding Drag   │ │ 1-Click Due Diligence PDF │
        └───────────────────────────────────┘ └──────────────────────────────────┘ └───────────────────────────┘
```

---

## 🔬 The 5 Econometric Pillars & Mathematical Foundations

### 1. CAPM & Jensen's Alpha
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \epsilon_{i,t}$$
* Evaluates raw excess return relative to the market benchmark. Newey-West HAC standard errors adjust for autocorrelation and heteroskedasticity.

### 2. Carhart 4-Factor Risk Adjustment
$$R_{i,t} - R_{f,t} = \alpha_{i,\text{4F}} + \beta_{i,M} (R_{m,t} - R_{f,t}) + \beta_{i,SMB} SMB_t + \beta_{i,HML} HML_t + \beta_{i,WML} WML_t + \epsilon_{i,t}$$
* Strips away passive style factor premia (Small minus Big size tilt, High minus Low value tilt, Winners minus Losers momentum tilt). True alpha must survive this decomposition.

### 3. Treynor-Mazuy Market Timing
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \gamma_i (R_{m,t} - R_{f,t})^2 + \epsilon_{i,t}$$
* Tests whether the manager dynamically adjusts market beta ahead of bull/bear phases ($\gamma > 0$ with $p < 0.05$).

### 4. 1,000-Iteration Residual Bootstrap (Kosowski et al. 2006)
* Re-samples regression residuals under the null hypothesis of zero skill ($\alpha = 0$) to generate non-parametric empirical $t$-distribution, immune to fat tails and non-normality.

### 5. Benjamini-Hochberg False Discovery Rate (FDR) Control
$$q_{(i)} = \min_{k \ge i} \left\{ \frac{m \cdot p_{(k)}}{k} \right\} \le 0.10$$
* Controls the false positive rate across multiple fund comparisons, ensuring apparent outperformance is not a mathematical artifact of selection bias.

---

## 🤖 Modern Explainable AI: XGBoost + TreeSHAP

Classical multi-factor regression assumes fixed, linear relationships across market cycles. In real market conditions:
- Factor loadings change dynamically across monetary policy and volatility regimes.
- Interaction effects between factors (e.g., Value performing differently in High-Beta vs Low-Beta regimes) create non-linear risk surfaces.

AlphaLens integrates an **XGBoost Regressor** trained on factor return series, coupled with **TreeSHAP (Lundberg & Lee, 2017)**:
- **Exact Additive Shapley Values**: Mathematically decomposes each period's fund return into exact basis-point contributions from Market, Size, Value, and Momentum.
- **Factor Radar**: Visualizes where the manager actually took active risk versus passive style loading.
- **Return Waterfall**: Highlights the precise factors driving monthly performance.

---

## 💰 Fee Leakage & Compounding Wealth Erosion Simulator

A mutual fund charging a 2.00% Total Expense Ratio (TER) while delivering only style factor exposure silently destroys long-term retail and institutional wealth.
- Models SIP and lump-sum investments over 5 to 30 years comparing:
  1. **Direct Active Fund Path** (Net returns after fees)
  2. **Low-Cost Passive Benchmark Alternative** (e.g., 0.10% TER Index ETF)
  3. **Gross Manager Return** (Unconstrained wealth trajectory before fee deduction)
- Exposes the **"Silent Wealth Destroyer"**: Even a 1.5% fee difference erodes over **35% to 45% of total wealth over a 25-year compounding horizon**.

---

## 📑 Automated Institutional Due Diligence Tear-Sheet

With one click, export an executive PDF report containing:
- Executive Summary & Overall Skill Verdict
- 5-Pillar Statistical Scorecard (Alpha, P-values, Bootstrap P, Market Timing $\gamma$, FDR $q$)
- Multi-Factor Exposure Table (Loadings, $t$-statistics, Economic Significance)
- Wealth Erosion & Fee Drag Assessment
- Mandatory Methodological Disclaimers & Governance Caveats

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher

### Installation & Launch

```bash
# Clone the repository
git clone https://github.com/<your-username>/fund-manager-skill-detector.git
cd fund-manager-skill-detector

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows (macOS/Linux: source .venv/bin/activate)

# Install dependencies
pip install -r requirements.txt

# Run the interactive platform
streamlit run app.py
```

Or on Windows, simply double-click `run_dashboard.bat`.

---

## 📊 Complete Analytical Suite

| Page | Purpose |
|---|---|
| **Overview** | Cross-market high-level evidence summary, classification breakdown, and alpha distributions. |
| **Live Fund Analyser 🚀** | **On-demand analysis** for 2,500+ Indian funds (AMFI) & US tickers. Features SHAP radar, rolling alpha, fee calculator, and PDF export. |
| **India** | Deep-dive into 12 flagship Indian schemes against IIMA total-return factor series. |
| **USA** | Deep-dive into 12 US mutual funds against Ken French 4-factor daily series. |
| **Cross-Market Comparison** | Rigorous side-by-side analysis contrasting fee structures, factor efficiency, and alpha survival rates. |
| **Benchmark Robustness** | Sensitivity analysis showing how benchmark choice (Price Index vs Total Return Index) impacts skill classifications. |
| **Fund Explorer** | Granular inspection of individual funds across all 5 statistical pillars. |
| **Methodology & Data** | Mathematical formulation, data lineage, and academic transparency documentation. |

---

## 💼 Resume Bullet Points (Copy & Paste for Portfolios)

### Quant / Financial Engineering Role
> **Quantitative Asset Management & Explainable AI Platform (Python, Streamlit, Econometrics, XGBoost, SHAP)**
> - Engineered an institutional fund manager skill detection platform implementing Carhart 4-Factor risk adjustment, Treynor-Mazuy market timing, Kosowski (2006) 1,000-iteration residual bootstrap, and Benjamini-Hochberg False Discovery Rate corrections.
> - Built on-demand ingestion pipelines integrating the AMFI real-time API (2,500+ funds) and Yahoo Finance, parsing historical NAV series and aligning multi-market trading calendars.
> - Integrated XGBoost and TreeSHAP to calculate dynamic factor Shapley attributions, decoupling non-linear factor tilts from genuine managerial alpha.
> - Designed a 30-year compounding fee leakage simulation engine and automated Jinja2/WeasyPrint institutional PDF tear-sheet generation.

### Data Science / FinTech Role
> **Fund Manager Skill & Factor Attribution Engine (Python, Pandas, Plotly, Streamlit, REST APIs)**
> - Developed an interactive financial analytics platform that classifies mutual funds across 6 statistical evidence tiers to detect whether excess returns stem from skill, market beta, or style luck.
> - Formulated automated data pipelines with Newey-West HAC standard error estimation and non-parametric bootstrap simulations to mitigate fat-tail bias.
> - Built interactive factor radar, rolling CAPM beta time series, and fee erosion visualizations using Plotly and Streamlit.

---

## 📜 Academic Integrity & Limitations

- **Survivorship & Selection Bias**: Five-year windows on prominent schemes cannot represent entire fund universes; survivorship bias is naturally present in open-market samples.
- **Benchmark Selection**: The Indian primary benchmark is the IIMA Total Return portfolio; testing against dividend-free indices shows significant classification drift (highlighted on the Benchmark Robustness page).
- **Manager Tenure Attribution**: Results describe scheme performance; where portfolio managers have less than 90% tenure across the analysis window, results reflect scheme governance rather than an individual.
- **Regulatory Disclaimer**: Statistical evidence summaries do not constitute financial or investment advice.
