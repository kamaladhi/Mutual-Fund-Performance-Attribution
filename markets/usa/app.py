"""
Streamlit Fund Manager Skill Detector Dashboard
===============================================
CAPM & APT Performance Evaluation with Bootstrap Significance Testing.

Central Question:
"Is observed fund outperformance consistent with genuine manager skill,
or is it more plausibly explained by market/style exposure and random variation?"

This dashboard is an evidence-synthesis diagnostic tool over existing, validated
analytical results (Tasks 1-8). It does not recompute analytical engines, predict
returns, or offer investment recommendations.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None


# -----------------------------------------------------------------------------
# 1. Robust Data Loader
# -----------------------------------------------------------------------------
def load_dashboard_data(base_dir="."):
    """
    Loads all required processed analytical artifacts for the dashboard.
    Returns a dictionary of DataFrames / metadata structures and an error string.
    """
    proc_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    files = {
        "final_eval": os.path.join(proc_dir, "final_evaluation.csv"),
        "final_json": os.path.join(proc_dir, "final_evaluation.json"),
        "capm": os.path.join(proc_dir, "capm_results.csv"),
        "bootstrap": os.path.join(proc_dir, "bootstrap_results.csv"),
        "factor": os.path.join(proc_dir, "factor_model_results.csv"),
        "pers_summary": os.path.join(proc_dir, "persistence_summary.csv"),
        "pers_periods": os.path.join(proc_dir, "persistence_period_results.csv"),
        "screening": os.path.join(raw_dir, "screening_metadata.json"),
    }
    
    missing_files = [k for k, v in files.items() if not os.path.exists(v) and k != "screening"]
    if missing_files:
        return None, f"Missing required processed data files: {', '.join(missing_files)}"
    
    try:
        data = {
            "final_eval": pd.read_csv(files["final_eval"]),
            "capm": pd.read_csv(files["capm"]),
            "bootstrap": pd.read_csv(files["bootstrap"]),
            "factor": pd.read_csv(files["factor"]),
            "pers_summary": pd.read_csv(files["pers_summary"]),
            "pers_periods": pd.read_csv(files["pers_periods"]),
            "screening": None,
        }
        
        if os.path.exists(files["screening"]):
            with open(files["screening"], "r") as f:
                data["screening"] = json.load(f)
                
        # Validate required columns in final_eval
        req_cols = ["symbol", "final_category", "total_score", "evaluation_version", "final_explanation"]
        missing_cols = [c for c in req_cols if c not in data["final_eval"].columns]
        if missing_cols:
            return None, f"final_evaluation.csv missing required columns: {', '.join(missing_cols)}"
            
        return data, None
    except Exception as e:
        return None, f"Failed loading dashboard data: {str(e)}"


# Decorate with st.cache_data if Streamlit is available
if HAS_STREAMLIT:
    load_dashboard_data = st.cache_data(show_spinner=False)(load_dashboard_data)


# -----------------------------------------------------------------------------
# 2. Formatting Helpers
# -----------------------------------------------------------------------------
def fmt_pct(val, decimals=2, include_sign=False):
    if pd.isna(val):
        return "N/A"
    sign = "+" if include_sign and val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"

def fmt_val(val, decimals=4):
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"

def fmt_pval(val):
    if pd.isna(val):
        return "N/A"
    if val < 0.0001:
        return "< 0.0001"
    return f"{val:.4f}"

def get_category_badge(category):
    if category == "Strong Evidence of Skill":
        return f'<span class="category-badge-strong">🟢 {category}</span>'
    elif category == "Strong Skill Candidate (Preliminary)":
        return f'<span class="category-badge-promising">🟢 {category}</span>'
    elif category == "Moderate Evidence":
        return f'<span class="category-badge-moderate">🔵 {category}</span>'
    elif category == "Positive but Inconclusive":
        return f'<span class="category-badge-positive">🔷 {category}</span>'
    elif category == "Weak/Inconclusive":
        return f'<span class="category-badge-weak">🟠 {category}</span>'
    else:
        return f'<span class="category-badge-style">🔴 {category}</span>'

def get_flag_indicator(condition, true_text="✓ Yes", false_text="✗ No"):
    if condition:
        return f'<span style="color: #059669; font-weight: 600;">{true_text}</span>'
    else:
        return f'<span style="color: #DC2626; font-weight: 600;">{false_text}</span>'


# -----------------------------------------------------------------------------
# 3. Main Streamlit Layout Function
# -----------------------------------------------------------------------------
def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Please run: pip install streamlit")
        return

    # -------------------------------------------------------------------------
    # Page Configuration & CSS Setup
    # -------------------------------------------------------------------------
    st.set_page_config(
        page_title="Fund Manager Skill Detector",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            font-weight: 400;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .pipeline-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 4px;
        }
        .warning-card {
            background-color: #FFFBEB;
            border: 1px solid #FCD34D;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 20px;
            color: #92400E;
            font-size: 0.92rem;
        }
        .category-badge-strong {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .category-badge-promising {
            background-color: #D1FAE5;
            color: #065F46;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .category-badge-moderate {
            background-color: #E1EFFE;
            color: #1E429F;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .category-badge-positive {
            background-color: #E0F2FE;
            color: #075985;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .category-badge-weak {
            background-color: #FFE8D6;
            color: #A04000;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .category-badge-style {
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load Data
    data, err = load_dashboard_data(".")
    if err:
        st.error(f"❌ Dashboard Initialization Error: {err}")
        st.info("Please run `python -m src.evaluator .` to generate required output files.")
        st.stop()
        
    final_df = data["final_eval"]
    capm_df = data["capm"]
    boot_df = data["bootstrap"]
    factor_df = data["factor"]
    pers_sum_df = data["pers_summary"]
    pers_per_df = data["pers_periods"]
    screening_meta = data["screening"]

    # Fund List & Mapping
    fund_symbols = list(final_df["symbol"].unique())
    fund_names = {}
    if screening_meta:
        for item in screening_meta:
            fund_names[item.get("ticker")] = item.get("fund_name", item.get("ticker"))
            
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=64)
    st.sidebar.title("Navigation")
    
    st.sidebar.subheader("🎯 Select Fund for Analysis")
    fund_options = [f"{sym} - {fund_names.get(sym, sym)}" for sym in fund_symbols]
    selected_option = st.sidebar.selectbox("Choose Mutual Fund:", fund_options, index=0)
    selected_symbol = selected_option.split(" - ")[0]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Details")
    st.sidebar.markdown("""
    - **Analyzed Period**: 2019-01-03 to 2023-12-29
    - **Total Observations**: 1,257 trading days
    - **Candidate Funds**: 12 funds
    - **Market for CAPM and 4-factor**: Fama-French Mkt-RF (total-return market excess return)
    - **S&P 500 (^GSPC)**: descriptive only (price index, excludes dividends)
    - **Risk-Free Rate**: Kenneth French Daily Rf
    - **Factors**: Fama-French 3-Factor + Momentum
    - **Rules version**: shared with the India market (evaluation v2.3.0)
    """)

    # -------------------------------------------------------------------------
    # Header & Overview
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">Fund Manager Skill Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CAPM & APT Performance Evaluation with Circular Block Bootstrap Significance Testing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-box">
    <b>Methodology Pipeline:</b> Data Acquisition & Alignment ➔ CAPM OLS Regression ➔ Circular Block Bootstrap Significance ➔ Carhart 4-Factor Risk Adjustment ➔ Intertemporal Performance Persistence ➔ Integrated Rule-Based Skill Evaluation Engine
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Active-Fund Verification Warning (REQUIRED SECTION 15)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="warning-card">
    <b>⚠️ Active-Management Verification Status Warning:</b><br/>
    Some candidate funds in this sample remain classified as <code>UNVERIFIED_CANDIDATE</code> under the project's screening protocol. 
    Therefore, these classifications must be interpreted strictly as statistical evidence within the analyzed fund sample, and not as definitive proof of active manager skill or investment advisory recommendations.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Executive Summary (SECTION 5 & 14)
    # -------------------------------------------------------------------------
    st.subheader("📈 Executive Summary & Category Distribution")
    st.caption("Six evidence classes, applied by the same rules as the India market. "
               "Strong Skill Candidate (Preliminary) is a screening tier: marginal CAPM evidence, a significant positive four-factor alpha, "
               "alpha positive in at least 3 of 4 subperiods and a bootstrap p-value below 0.10. It is not confirmed skill.")

    # Category counts dynamically
    all_categories = [
        "Strong Evidence of Skill",
        "Strong Skill Candidate (Preliminary)",
        "Moderate Evidence",
        "Positive but Inconclusive",
        "Weak/Inconclusive",
        "Likely Style/Luck Driven"
    ]
    cat_counts = final_df["final_category"].value_counts().to_dict()
    for cat in all_categories:
        cat_counts.setdefault(cat, 0)

    # Dynamic metrics
    total_funds = len(final_df)
    max_score = final_df["total_score"].max()
    min_score = final_df["total_score"].min()
    median_score = final_df["total_score"].median()

    # KPI Layout
    kpis = [
        ("Evaluated Funds", total_funds, "#0F172A"),
        ("Strong Evidence", cat_counts["Strong Evidence of Skill"], "#10B981"),
        ("Skill Candidate (Prelim.)", cat_counts["Strong Skill Candidate (Preliminary)"], "#059669"),
        ("Moderate Evidence", cat_counts["Moderate Evidence"], "#2563EB"),
        ("Positive but Inconclusive", cat_counts["Positive but Inconclusive"], "#0369A1"),
        ("Weak / Inconclusive", cat_counts["Weak/Inconclusive"], "#D97706"),
        ("Style / Luck Driven", cat_counts["Likely Style/Luck Driven"], "#DC2626"),
    ]
    for col, (label, value, color) in zip(st.columns(len(kpis)), kpis):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color};">{value}</div></div>', unsafe_allow_html=True)

    st.caption("Every class is always shown, including those with zero funds. The classes keep different evidence levels separate; they are evidence summaries, not a ranking of managers.")
    st.markdown("<br/>", unsafe_allow_html=True)

    # Category Distribution Chart & Score Summary
    sum_col1, sum_col2 = st.columns([3, 2])
    
    with sum_col1:
        cat_data = pd.DataFrame({
            "Category": all_categories,
            "Count": [cat_counts[c] for c in all_categories]
        })
        colors = ["#10B981", "#059669", "#3B82F6", "#0EA5E9", "#F59E0B", "#EF4444"]
        fig_cat = px.bar(
            cat_data, x="Category", y="Count", text="Count",
            color="Category",
            color_discrete_map=dict(zip(all_categories, colors)),
            title="Fund Count by Skill Evidence Category"
        )
        fig_cat.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with sum_col2:
        st.markdown("#### 📊 Score Summary")
        max_sym = final_df.loc[final_df['total_score'].idxmax(), 'symbol']
        min_sym = final_df.loc[final_df['total_score'].idxmin(), 'symbol']
        st.markdown(f"- **Highest Evidence Score**: `{max_score:.1f} / 100.0` ({max_sym})")
        st.markdown(f"- **Lowest Evidence Score**: `{min_score:.1f} / 100.0` ({min_sym})")
        st.markdown(f"- **Median Evidence Score**: `{median_score:.1f} / 100.0` Across 12 Funds")
        st.markdown("""
        > **Evidence Score Note:** The 0–100 score is a transparent, rule-based quantitative metric with documented weights. It evaluates evidence strength, **not** a predictive machine learning score.
        """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Selected Fund Deep-Dive (SECTIONS 6, 7, 8, 9, 10, 11, 12)
    # -------------------------------------------------------------------------
    fund_eval = final_df[final_df["symbol"] == selected_symbol].iloc[0]
    fund_capm = capm_df[capm_df["symbol"] == selected_symbol].iloc[0] if "symbol" in capm_df.columns else capm_df.iloc[fund_symbols.index(selected_symbol)]
    fund_boot = boot_df[boot_df["symbol"] == selected_symbol].iloc[0] if "symbol" in boot_df.columns else boot_df.iloc[fund_symbols.index(selected_symbol)]
    fund_fact = factor_df[factor_df["symbol"] == selected_symbol].iloc[0] if "symbol" in factor_df.columns else factor_df.iloc[fund_symbols.index(selected_symbol)]
    fund_pers = pers_sum_df[pers_sum_df["symbol"] == selected_symbol].iloc[0] if "symbol" in pers_sum_df.columns else pers_sum_df.iloc[fund_symbols.index(selected_symbol)]
    
    st.header(f"🔍 Deep-Dive Evaluation: {selected_symbol} - {fund_names.get(selected_symbol, selected_symbol)}")

    # Category & Explanation Header
    head_col1, head_col2, head_col3 = st.columns([3, 1, 1])
    with head_col1:
        badge_html = get_category_badge(fund_eval["final_category"])
        st.markdown(f"### Category: {badge_html}", unsafe_allow_html=True)
    with head_col2:
        st.metric("Total Evidence Score", f"{fund_eval['total_score']:.1f} / 100")
    with head_col3:
        st.metric("Evaluation Version", str(fund_eval.get("evaluation_version", "2.3.0")))

    st.info(f"**Deterministic Explanation**: {fund_eval['final_explanation']}")

    # Key Evidence Matrix
    st.subheader("📋 Core Evidence Matrix")
    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
    
    with ev_col1:
        capm_pos = bool(fund_eval.get("capm_positive", fund_eval.get("capm_alpha_annual_pct", 0) > 0))
        capm_sig = bool(fund_eval.get("capm_significant", False))
        st.markdown("**1. CAPM Evidence**")
        st.markdown(f"- Alpha > 0: {get_flag_indicator(capm_pos, '✓ Positive', '✗ Non-Positive')}", unsafe_allow_html=True)
        st.markdown(f"- Statistical Sig.: {get_flag_indicator(capm_sig, '✓ Significant (HAC p<0.05)', '✗ Not Significant')}", unsafe_allow_html=True)
        
    with ev_col2:
        boot_sig = bool(fund_eval.get("bootstrap_significant_flag", fund_eval.get("bootstrap_significant", False)))
        boot_ci_pos = bool(fund_eval.get("bootstrap_ci_positive", False))
        st.markdown("**2. Circular Bootstrap**")
        st.markdown(f"- Bootstrap Sig.: {get_flag_indicator(boot_sig, '✓ Significant (p<0.05)', '✗ Not Significant')}", unsafe_allow_html=True)
        st.markdown(f"- 95% CI > 0: {get_flag_indicator(boot_ci_pos, '✓ Entirely > 0', '✗ Crosses Zero')}", unsafe_allow_html=True)

    with ev_col3:
        fact_surv = bool(fund_eval.get("factor_alpha_survives", False))
        red_pct = fund_eval.get("alpha_reduction_pct", 0.0)
        st.markdown("**3. Carhart 4-Factor**")
        st.markdown(f"- Survives Factor Model: {get_flag_indicator(fact_surv, '✓ Alpha Survives', '⚠️ Materially Reduced')}", unsafe_allow_html=True)
        if pd.isna(red_pct):
            st.markdown("- Alpha Reduction: `n/a` (CAPM alpha is not a positive baseline)")
        else:
            st.markdown(f"- Alpha Reduction: `{red_pct:.1f}%`")

    with ev_col4:
        pers_maj = bool(fund_eval.get("persistent_positive_majority", fund_eval.get("alpha_positive_majority", False)))
        pers_rate = fund_eval.get("positive_alpha_persistence_rate", 0.0)
        st.markdown("**4. Persistence (4 Periods)**")
        st.markdown(f"- Majority Positive: {get_flag_indicator(pers_maj, '✓ Persistent', '✗ Non-Persistent')}", unsafe_allow_html=True)
        st.markdown(f"- Persistence Rate: `{pers_rate*100:.0f}%`")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Detailed Analytical Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 CAPM Analysis",
        "🔄 Bootstrap Significance",
        "🎯 Carhart 4-Factor Model",
        "⏳ Performance Persistence",
        "💯 Score Breakdown"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: CAPM Analysis (SECTION 8)
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("CAPM Single-Factor Model Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized CAPM Alpha", fmt_pct(fund_capm.get("alpha_annual_pct", 0.0), include_sign=True))
        c2.metric("Market Beta (β)", fmt_val(fund_capm.get("beta", 1.0)))
        c3.metric("CAPM R-Squared (R²)", fmt_val(fund_capm.get("r_squared", 0.0)))
        c4.metric("Alpha p-value (HAC, used)", fmt_pval(fund_capm.get("alpha_pvalue_hac", fund_capm.get("alpha_pvalue", 1.0))))

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # CAPM Details Table
        ci_lower = fund_capm.get("alpha_annual_pct_ci95_lower", fund_capm.get("alpha_annual_lower_pct", np.nan))
        ci_upper = fund_capm.get("alpha_annual_pct_ci95_upper", fund_capm.get("alpha_annual_upper_pct", np.nan))
        
        capm_summary_df = pd.DataFrame({
            "Metric": ["Daily Alpha", "Annualized Alpha", "Market Beta", "R-Squared", "HAC p-value (used)", "OLS p-value (diagnostic)", "95% OLS Confidence Interval", "Observations"],
            "Value": [
                fmt_val(fund_capm.get("alpha_daily", 0.0)),
                fmt_pct(fund_capm.get("alpha_annual_pct", 0.0), include_sign=True),
                fmt_val(fund_capm.get("beta", 1.0)),
                fmt_val(fund_capm.get("r_squared", 0.0)),
                fmt_pval(fund_capm.get("alpha_pvalue_hac", fund_capm.get("alpha_pvalue", 1.0))),
                fmt_pval(fund_capm.get("alpha_pvalue", 1.0)),
                f"[{fmt_pct(ci_lower)} , {fmt_pct(ci_upper)}]" if not pd.isna(ci_lower) else "N/A",
                f"{int(fund_capm.get('n_observations', 1257))}"
            ]
        })
        st.table(capm_summary_df)

        st.caption("ℹ️ **Methodological Note**: CAPM alpha measures average performance unexplained by market exposure within the CAPM specification. It is not by itself proof of manager skill.")

    # -------------------------------------------------------------------------
    # TAB 2: Circular Block Bootstrap (SECTION 9)
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Circular Block Bootstrap (CBB) Significance Testing")
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Observed Alpha (Annualized)", fmt_pct(fund_boot.get("observed_alpha_annual_pct", fund_boot.get("observed_alpha", 0.0)), include_sign=True))
        b2.metric("Bootstrap p-value", fmt_pval(fund_boot.get("p_value", 1.0)))
        b3.metric("Block Length", f"{int(fund_boot.get('block_length', 30))} Days")
        b4.metric("Replications", f"{int(fund_boot.get('n_bootstrap', 1000)):,}")

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Bootstrap CI Visualization
        b_ci_lower = fund_boot.get("ci_annual_lower_pct", fund_boot.get("ci_lower", np.nan))
        b_ci_upper = fund_boot.get("ci_annual_upper_pct", fund_boot.get("ci_upper", np.nan))
        b_obs = fund_boot.get("observed_alpha_annual_pct", fund_boot.get("observed_alpha", 0.0))
        
        fig_boot = go.Figure()
        fig_boot.add_trace(go.Bar(
            y=["95% Bootstrap CI"],
            x=[b_ci_upper - b_ci_lower],
            base=[b_ci_lower],
            orientation='h',
            name='95% Confidence Interval',
            marker_color='#93C5FD',
            opacity=0.7
        ))
        fig_boot.add_trace(go.Scatter(
            y=["95% Bootstrap CI"],
            x=[b_obs],
            mode='markers',
            name='Observed Alpha',
            marker=dict(color='#1E429F', size=14, symbol='diamond')
        ))
        fig_boot.add_shape(
            type="line", x0=0, y0=-0.5, x1=0, y1=0.5,
            line=dict(color="Red", width=2, dash="dash")
        )
        fig_boot.update_layout(
            title=f"Bootstrap 95% Confidence Interval for {selected_symbol} Alpha (% p.a.)",
            xaxis_title="Annualized Alpha (%)",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_boot, use_container_width=True)

        st.caption("ℹ️ **Methodological Note**: The Circular Block Bootstrap (Politis & Romano, 1992) resamples contiguous blocks of paired daily returns (fund and market kept together) to preserve short-range autocorrelation and volatility clustering. The resampled alphas are re-centred on zero to impose H0: alpha = 0, and the p-value is two-sided. The shaded bar is the uncentred 95% interval; the dashed red line marks zero alpha.")

    # -------------------------------------------------------------------------
    # TAB 3: Carhart 4-Factor Model (SECTION 10)
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Carhart 4-Factor Multi-Factor Risk Adjustment")
        
        f1, f2, f3, f4 = st.columns(4)
        capm_a = fund_fact.get("capm_alpha_annual_pct", fund_capm.get("alpha_annual_pct", 0.0))
        fact_a = fund_fact.get("alpha_annual_pct", 0.0)
        red_p = fund_eval.get("alpha_reduction_pct", 0.0)
        
        f1.metric("CAPM Alpha", fmt_pct(capm_a, include_sign=True))
        f2.metric("4-Factor Alpha", fmt_pct(fact_a, include_sign=True))
        f3.metric("Alpha Reduction %", f"{red_p:.1f}%")
        f4.metric("4-Factor R²", fmt_val(fund_fact.get("r_squared", 0.0)))

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Alpha Comparison Chart
        col_chart, col_betas = st.columns([1, 1])
        
        with col_chart:
            comp_df = pd.DataFrame({
                "Model": ["CAPM Alpha", "Carhart 4-Factor Alpha"],
                "Alpha (% p.a.)": [capm_a, fact_a]
            })
            fig_comp = px.bar(
                comp_df, x="Model", y="Alpha (% p.a.)", color="Model",
                text="Alpha (% p.a.)",
                color_discrete_sequence=["#3B82F6", "#8B5CF6"],
                title="CAPM Alpha vs. 4-Factor Alpha"
            )
            fig_comp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_comp.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_betas:
            st.markdown("#### Factor Betas (Exposures)")
            beta_df = pd.DataFrame({
                "Factor": ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)", "Momentum (Mom)"],
                "Beta Exposure": [
                    fmt_val(fund_fact.get("beta_market", 1.0)),
                    fmt_val(fund_fact.get("beta_smb", 0.0)),
                    fmt_val(fund_fact.get("beta_hml", 0.0)),
                    fmt_val(fund_fact.get("beta_momentum", 0.0))
                ]
            })
            st.table(beta_df)

        if bool(fund_eval.get("style_explanation", False)):
            st.caption("ℹ️ **Interpretation**: factor adjustment removes at least half of a positive, at-least-marginally significant CAPM alpha, "
                       "so systematic style exposures (size, value, momentum) explain a material part of the apparent outperformance.")
        else:
            st.caption("ℹ️ **Interpretation**: the Carhart model controls for market, size, value and momentum tilts. "
                       "Here the factor adjustment does not remove a material part of a noteworthy positive CAPM alpha (or there is no such alpha to explain).")

    # -------------------------------------------------------------------------
    # TAB 4: Performance Persistence (SECTION 11)
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Intertemporal Performance Persistence Across Subperiods")
        
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Persistence Rate", f"{fund_pers.get('positive_alpha_persistence_rate', 0.0)*100:.0f}%")
        p2.metric("Sig. Persistence Rate", f"{fund_pers.get('significant_positive_alpha_persistence_rate', 0.0)*100:.0f}%")
        p3.metric("Subperiod Mean Alpha", fmt_pct(fund_pers.get("alpha_mean_annual_pct", 0.0), include_sign=True))
        p4.metric("Subperiod Std. Dev.", fmt_pct(fund_pers.get("alpha_std_annual_pct", 0.0)))

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Subperiod Chart
        fund_sub = pers_per_df[pers_per_df["symbol"] == selected_symbol] if "symbol" in pers_per_df.columns else pd.DataFrame()
        
        if not fund_sub.empty:
            fig_pers = px.bar(
                fund_sub, x="period_id", y="alpha_annual_pct",
                text="alpha_annual_pct",
                title=f"Subperiod CAPM Annualized Alpha (% p.a.) for {selected_symbol}",
                labels={"period_id": "Subperiod", "alpha_annual_pct": "Annualized Alpha (%)"},
                color_discrete_sequence=["#10B981" if x > 0 else "#EF4444" for x in fund_sub["alpha_annual_pct"]]
            )
            fig_pers.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_pers.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pers, use_container_width=True)

        st.caption("ℹ️ **Methodological Note**: The total 1,257 trading days are partitioned into 4 non-overlapping chronological subperiods (~314 days each). Persistence measures sign consistency across subperiods. "
                   "\"Majority\" means strictly more than half (at least 3 of 4); exactly 2 of 4 is a coin-flip tie and does not count.")

    # -------------------------------------------------------------------------
    # TAB 5: Score Breakdown (SECTION 12)
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Transparent Evidence Score Component Breakdown")
        
        score_df = pd.DataFrame({
            "Component": [
                "1. Performance Score (Raw CAPM Alpha)",
                "2. Statistical Significance Score (HAC p-value & Bootstrap; positive alpha only)",
                "3. Factor-Model Score (4-Factor Survival)",
                "4. Persistence Score (Subperiod Repeatability)",
                "5. Style/Luck Adjustment (Deduction for Excessive Factor Attenuation)"
            ],
            "Max Points": ["20.0", "25.0", "25.0", "20.0", "±10.0"],
            "Earned Points": [
                f"{fund_eval.get('performance_score', 0.0):.1f}",
                f"{fund_eval.get('significance_score', 0.0):.1f}",
                f"{fund_eval.get('factor_score', 0.0):.1f}",
                f"{fund_eval.get('persistence_score', 0.0):.1f}",
                f"{fund_eval.get('style_luck_adjustment', 0.0):.1f}"
            ]
        })
        
        col_tbl, col_hbar = st.columns([1, 1])
        with col_tbl:
            st.table(score_df)
            st.markdown(f"### **Total Evidence Score**: `{fund_eval['total_score']:.1f} / 100.0`")
            
        with col_hbar:
            comp_plot_df = pd.DataFrame({
                "Component": ["Performance", "Significance", "Factor Model", "Persistence", "Style/Luck Adj."],
                "Score": [
                    fund_eval.get("performance_score", 0.0),
                    fund_eval.get("significance_score", 0.0),
                    fund_eval.get("factor_score", 0.0),
                    fund_eval.get("persistence_score", 0.0),
                    fund_eval.get("style_luck_adjustment", 0.0)
                ]
            })
            fig_score = px.bar(
                comp_plot_df, x="Score", y="Component", orientation="h",
                text="Score",
                color="Score",
                color_continuous_scale="Blues",
                title="Component Score Contribution"
            )
            fig_score.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_score, use_container_width=True)

        st.caption("ℹ️ **Rule-Based System**: The evidence score is calculated strictly using fixed weights defined in Task 8 architecture. It is fully deterministic, transparent, and reproducible.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Cross-Fund Comparison Table & Visualizations (SECTION 13)
    # -------------------------------------------------------------------------
    st.header("📊 Cross-Fund Evidence Comparison")
    
    comp_display_df = final_df[[
        "symbol", "final_category", "total_score", "capm_alpha_annual_pct",
        "factor_alpha_annual_pct", "alpha_reduction_pct", "bootstrap_p_value",
        "positive_alpha_persistence_rate"
    ]].copy()
    
    comp_display_df.columns = [
        "Fund Ticker", "Final Category", "Total Score", "CAPM Alpha (% p.a.)",
        "4-Factor Alpha (% p.a.)", "Alpha Reduction (%)", "Bootstrap p-value", "Persistence Rate"
    ]
    comp_display_df["Persistence Rate"] = (comp_display_df["Persistence Rate"] * 100).round(0).astype(str) + "%"
    comp_display_df["CAPM Alpha (% p.a.)"] = comp_display_df["CAPM Alpha (% p.a.)"].round(2)
    comp_display_df["4-Factor Alpha (% p.a.)"] = comp_display_df["4-Factor Alpha (% p.a.)"].round(2)
    comp_display_df["Alpha Reduction (%)"] = comp_display_df["Alpha Reduction (%)"].round(1)
    comp_display_df["Bootstrap p-value"] = comp_display_df["Bootstrap p-value"].round(4)
    comp_display_df["Total Score"] = comp_display_df["Total Score"].round(1)

    st.dataframe(comp_display_df.sort_values(by="Total Score", ascending=False), use_container_width=True)

    # Score Comparison Chart
    fig_comp_scores = px.bar(
        final_df.sort_values(by="total_score", ascending=True),
        x="total_score", y="symbol", orientation="h",
        color="final_category",
        text="total_score",
        title="Evidence Score Comparison Across Evaluated Funds",
        labels={"total_score": "Total Evidence Score (0–100)", "symbol": "Fund Ticker", "final_category": "Category"},
        color_discrete_map={
            "Strong Evidence of Skill": "#10B981",
            "Strong Skill Candidate (Preliminary)": "#059669",
            "Moderate Evidence": "#3B82F6",
            "Positive but Inconclusive": "#0EA5E9",
            "Weak/Inconclusive": "#F59E0B",
            "Likely Style/Luck Driven": "#EF4444"
        }
    )
    fig_comp_scores.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_comp_scores, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Methodology & Documented Limitations Section (SECTION 16)
    # -------------------------------------------------------------------------
    st.header("📚 Methodology & Documented Project Limitations")
    
    with st.expander("📖 Quantitative Methodology Summary (CAPM, Bootstrap, Factor Model, Persistence, Rules)"):
        st.markdown("""
        ### 1. CAPM Engine
        Single-index OLS regression of daily fund excess returns on market excess returns:
        $$R_{fund, t} - R_{f, t} = \\alpha + \\beta (R_{m, t} - R_{f, t}) + \\varepsilon_t$$
        Annualized alpha is computed as $\\alpha_{annual} = (1 + \\alpha_{daily})^{252} - 1$.
        
        ### 2. Circular Block Bootstrap (CBB)
        Non-parametric significance test preserving temporal dependency via contiguous circular blocks of paired fund/market returns (block length $b=30$ days, $B=1,000$ replications, fixed seed). The resampled alphas are re-centred on zero to impose $H_0: \\alpha = 0$ and the empirical p-value is two-sided (smoothed, $(k+1)/(B+1)$).
        
        ### 3. Carhart 4-Factor APT Model
        Risk adjustment controlling for systematic market, size, value, and momentum factor exposures:
        $$R_{fund, t} - R_{f, t} = \\alpha_{4F} + \\beta_m (Mkt-RF)_t + \\beta_s SMB_t + \\beta_h HML_t + \\beta_m MOM_t + \\varepsilon_t$$
        Alpha reduction percentage evaluates factor absorption: $1 - \\frac{\\alpha_{4F}}{\\alpha_{CAPM}}$.
        
        ### 4. Performance Persistence
        Partitioning trading days into 4 chronological subperiods to calculate sign consistency and stability ($s_{\\alpha}$).
        
        ### 5. Rule-Based Evaluation Engine
        Precedence-based Boolean decision tree (the same rules as the India market): Likely Style/Luck Driven is checked first, then Strong Evidence of Skill,
        Strong Skill Candidate (Preliminary), Moderate Evidence, Positive but Inconclusive, and finally Weak/Inconclusive. Inference uses Newey-West HAC p-values,
        and the Strong tier additionally requires Benjamini-Hochberg q-values below 0.10 across the 12-fund sample.
        """)

    with st.expander("⚠️ Documented Methodological Limitations & Risk Warnings"):
        st.markdown("""
        The evaluation framework and findings carry the following documented limitations:
        
        1. **Sample Size Constraints**: Evaluated on 12 candidate funds over a 5-year sample window (2019–2023). Results cannot be generalized outside this cohort.
        2. **Subperiod Partition Sensitivity**: Persistence analysis uses 4 chronological subperiods (~314 days each); alternative subperiod boundaries may affect local persistence rates.
        3. **Bootstrap Block Length**: Circular block bootstrap results depend on chosen block length ($b=30$ days).
        4. **Omitted Factor Risk**: Carhart 4-factor model does not capture specialized factor risks (e.g., quality, low volatility, sector tilts, liquidity).
        5. **Survivorship & Selection Bias**: Sample consists of surviving active mutual funds; dead/merged funds are excluded.
        6. **Mutual Fund Return Measurement**: Net asset value (NAV) returns reflect expense ratios and cash drag but omit front-end loads or individual tax friction.
        7. **Statistical Non-Causality**: Statistical significance ($p < 0.05$) indicates pattern unlikelihood under random chance, NOT definitive causality or manager talent.
        8. **Factor Exposure vs. Intentional Style**: High factor loadings indicate risk tilts but cannot distinguish passive factor exposure from active tactical timing.
        9. **Active-Management Verification**: Funds marked `UNVERIFIED_CANDIDATE` have not undergone full secondary mandate verification.
        10. **Non-peer funds**: AMECX (income), PRHSX (healthcare sector) and PAGRX (multi-asset) are exploratory comparators, not clean peers for a broad US equity universe.
        """)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Fund Manager Skill Detector • US market • Rules v2.3.0 • Diagnostic Framework for Empirical Performance Evaluation • Non-Commercial Academic/Educational Tool</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
