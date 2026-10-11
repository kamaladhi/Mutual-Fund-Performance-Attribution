"""
Indian Fund Manager Skill Detector Dashboard
Institutional Asset Pricing & Econometric Diagnostic Platform
==============================================================
CAPM, Carhart 4-Factor APT Decomposition, and Circular Block Bootstrap Significance Framework.
Evaluates actively managed Indian equity mutual funds against a total-return market proxy
(IIM Ahmedabad market portfolio by default; a genuine NIFTY 500 TRI if supplied) and the
IIM Ahmedabad academic multi-factor series.
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

CHART_FONT = "Roboto, Segoe UI, Arial, sans-serif"

MARKET_PROXY_LABELS = {
    "iima_market": "IIMA value-weighted market portfolio (total return; Mkt-RF + RF)",
    "nifty500_tri": "NIFTY 500 Total Return Index",
    "nifty500_price": "NIFTY 500 price index (excludes dividends)",
}


# -----------------------------------------------------------------------------
# 1. Institutional Data Loader
# -----------------------------------------------------------------------------
def load_dashboard_data(base_dir="."):
    """
    Loads all required processed analytical artifacts for the dashboard.
    Returns a dictionary of DataFrames / metadata structures and an error string.
    Optional artifacts (sensitivity tables, descriptive statistics, run manifest) are loaded when present.
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
    optional_csv = {
        "capm_nifty_price": "capm_sensitivity_nifty500_price.csv",
        "block_sensitivity": "bootstrap_block_sensitivity.csv",
        "benchmark_sensitivity": "benchmark_sensitivity.csv",
        "descriptive": "descriptive_statistics.csv",
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
            "manifest": None,
        }

        for key, fname in optional_csv.items():
            path = os.path.join(proc_dir, fname)
            data[key] = pd.read_csv(path) if os.path.exists(path) else None

        # Cast symbols to string across all tables
        for df_key in ["final_eval", "capm", "bootstrap", "factor", "pers_summary", "pers_periods",
                       "capm_nifty_price", "block_sensitivity", "benchmark_sensitivity"]:
            if data.get(df_key) is not None and "symbol" in data[df_key].columns:
                data[df_key]["symbol"] = data[df_key]["symbol"].astype(str)
        if data["descriptive"] is not None and "series" in data["descriptive"].columns:
            data["descriptive"]["series"] = data["descriptive"]["series"].astype(str)

        if os.path.exists(files["screening"]):
            with open(files["screening"], "r", encoding="utf-8") as f:
                data["screening"] = json.load(f)

        manifest_path = os.path.join(proc_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data["manifest"] = json.load(f)

        req_cols = ["symbol", "final_category", "total_score", "evaluation_version", "final_explanation"]
        missing_cols = [c for c in req_cols if c not in data["final_eval"].columns]
        if missing_cols:
            return None, f"final_evaluation.csv missing required columns: {', '.join(missing_cols)}"

        return data, None
    except Exception as e:
        return None, f"Failed loading dashboard data: {str(e)}"


if HAS_STREAMLIT:
    load_dashboard_data = st.cache_data(show_spinner=False, ttl=300)(load_dashboard_data)


# -----------------------------------------------------------------------------
# 2. Institutional Formatting Helpers
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
    if val < 0.001:
        return f"{val:.4f} (p < 0.001, Sig. at 0.1%)"
    elif val < 0.01:
        return f"{val:.4f} (p < 0.01, Sig. at 1%)"
    elif val < 0.05:
        return f"{val:.4f} (p < 0.05, Sig. at 5%)"
    else:
        return f"{val:.4f} (p >= 0.05, Not Significant)"

def fmt_p(val):
    """Compact p-value for metric cards (never truncated by the card width)."""
    if pd.isna(val):
        return "N/A"
    return "<0.0001" if val < 0.0001 else f"{val:.4f}"

def p_label(val, alpha_positive=None):
    """Plain-language significance label shown under a p-value metric."""
    if pd.isna(val):
        return "n/a"
    level = ("significant at 1%" if val < 0.01 else
             "significant at 5%" if val < 0.05 else
             "marginal (10%)" if val < 0.10 else "not significant")
    if alpha_positive is False and val < 0.10:
        return level + ", NEGATIVE alpha"
    return level

def get_category_badge(cat):
    if cat == "Strong Evidence of Skill":
        return f'<span class="badge badge-strong">{cat.upper()}</span>'
    elif cat == "Strong Skill Candidate (Preliminary)":
        return f'<span class="badge badge-candidate">{cat.upper()}</span>'
    elif cat == "Positive but Inconclusive":
        return f'<span class="badge badge-positive">{cat.upper()}</span>'
    elif cat == "Moderate Evidence":
        return f'<span class="badge badge-moderate">{cat.upper()}</span>'
    elif cat == "Weak/Inconclusive":
        return f'<span class="badge badge-weak">{cat.upper()}</span>'
    else:
        return f'<span class="badge badge-style">{cat.upper()}</span>'

def get_flag_indicator(condition, true_text="AFFIRMATIVE", false_text="NEGATIVE", is_warning=False):
    if condition:
        return f'<span class="indicator-pass">[{true_text}]</span>'
    elif is_warning:
        return f'<span class="indicator-warn">[{false_text}]</span>'
    else:
        return f'<span class="indicator-fail">[{false_text}]</span>'

def build_sidebar_facts(data):
    """Sample / benchmark facts derived from the loaded results (nothing hard-coded)."""
    facts = {}
    pers = data["pers_periods"]
    capm = data["capm"]
    facts["sample_start"] = str(pers["start_date"].min())
    facts["sample_end"] = str(pers["end_date"].max())
    facts["obs_min"] = int(capm["n_observations"].min())
    facts["obs_max"] = int(capm["n_observations"].max())
    facts["n_funds"] = int(capm["symbol"].nunique())
    manifest = data.get("manifest") or {}
    proxy = manifest.get("market_proxy", "iima_market")
    facts["market_proxy"] = proxy
    facts["market_label"] = MARKET_PROXY_LABELS.get(proxy, proxy)
    facts["block_length"] = int(data["bootstrap"]["block_length"].iloc[0])
    facts["n_bootstrap"] = int(data["bootstrap"]["n_bootstrap"].iloc[0])
    return facts


# -----------------------------------------------------------------------------
# 3. Streamlit Main Presentation
# -----------------------------------------------------------------------------
def _table_height(n_rows, row_px=35, header_px=38):
    """Pixel height that shows every row of a short table without an inner scrollbar."""
    return int(header_px + row_px * n_rows + 3)


def _style_fig(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family=CHART_FONT, size=11, color="#0F172A"),
    )
    return fig


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Please run: pip install streamlit")
        return

    st.set_page_config(
        page_title="Indian Fund Manager Skill Detector",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Formal Institutional Typography & Clean Palette.
    # NOTE: no blanket background-color on div/span -- it blanks every st.dataframe canvas.
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

    .stApp { background-color: #f8f9fa; }
    [data-testid="stHeader"] { background-color: #f8f9fa; }

    html, body, .stApp, .stMarkdown, .stMarkdown p, label, button, h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    code, pre, .mono-val {
        font-family: 'Source Code Pro', monospace !important;
    }

    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #212529;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #ced4da;
        padding-bottom: 0.3rem;
    }

    .sub-header {
        font-size: 0.9rem;
        font-weight: 500;
        color: #495057;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .pipeline-box {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-left: 4px solid #343a40;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 1rem;
        font-size: 0.86rem;
        color: #495057;
        line-height: 1.5;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 12px 10px;
        text-align: left;
        box-shadow: 0 1px 2px rgba(33, 37, 41, 0.05);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #212529;
        font-family: 'Source Code Pro', monospace;
    }

    .profile-container {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 1rem;
        font-size: 0.84rem;
        color: #495057;
    }

    .profile-item-label {
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.73rem;
        letter-spacing: 0.03em;
    }

    .profile-item-value {
        color: #212529;
        font-weight: 600;
    }

    .notice-card {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 1rem;
        color: #495057;
        font-size: 0.83rem;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-strong { background-color: #e6f4ea; color: #155724; border: 1px solid #c3e6cb; }
    .badge-candidate { background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
    .badge-positive { background-color: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
    .badge-moderate { background-color: #e7f1ff; color: #0c5460; border: 1px solid #b8daff; }
    .badge-weak { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .badge-style { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

    .indicator-pass { color: #155724; font-weight: 600; font-family: 'Source Code Pro', monospace; font-size: 0.78rem; }
    .indicator-warn { color: #856404; font-weight: 600; font-family: 'Source Code Pro', monospace; font-size: 0.78rem; }
    .indicator-fail { color: #721c24; font-weight: 600; font-family: 'Source Code Pro', monospace; font-size: 0.78rem; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #ced4da;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.84rem;
        padding: 8px 14px;
        color: #495057;
        border-radius: 4px 4px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #212529 !important;
        border-bottom: 2px solid #212529 !important;
    }
</style>
    """, unsafe_allow_html=True)

    data, err = load_dashboard_data(".")
    if err:
        st.error(f"Dashboard Initialization Error: {err}")
        st.info("Execute `python run_pipeline.py` to generate validated output datasets.")
        st.stop()

    final_df = data["final_eval"]
    capm_df = data["capm"]
    boot_df = data["bootstrap"]
    factor_df = data["factor"]
    pers_sum_df = data["pers_summary"]
    pers_per_df = data["pers_periods"]
    screening_meta = data["screening"]
    facts = build_sidebar_facts(data)

    meta_dict = {}
    if screening_meta:
        for item in screening_meta:
            ticker = str(item.get("ticker"))
            meta_dict[ticker] = item

    fund_symbols = list(final_df["symbol"].unique())

    # -------------------------------------------------------------------------
    # Sidebar: Clean Institutional Controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### PORTFOLIO UNIVERSE")

    fund_options = []
    for sym in fund_symbols:
        m = meta_dict.get(sym, {})
        s_name = m.get("short_name", sym)
        m_name = m.get("lead_manager", "Manager")
        fund_options.append(f"{sym} | {s_name} [{m_name}]")

    selected_option = st.sidebar.selectbox("Select Candidate Scheme:", fund_options, index=0)
    selected_symbol = selected_option.split(" | ")[0]
    curr_meta = meta_dict.get(selected_symbol, {})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### BENCHMARK & FACTOR PARAMETERS")
    obs_text = (f"{facts['obs_max']:,}" if facts["obs_min"] == facts["obs_max"]
                else f"{facts['obs_min']:,}-{facts['obs_max']:,}")
    st.sidebar.markdown(f"""
    - **Sample Period**: {facts['sample_start']} to {facts['sample_end']}
    - **Return Observations**: {obs_text} synchronous daily returns per fund
    - **Active Universe**: {facts['n_funds']} Direct-Growth equity schemes
    - **Market Proxy (CAPM)**: {facts['market_label']}
    - **Asset Pricing Factors**: IIMA Academic Data Library (Agarwalla, Jacob, Varma)
    - **Risk-Free Rate**: GoI 91-Day T-Bill (daily IIMA Rf)
    - **Inference**: Newey-West HAC; circular block bootstrap (b={facts['block_length']}, B={facts['n_bootstrap']:,})
    """)

    # -------------------------------------------------------------------------
    # Main Header & Institutional Metadata Box
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">Mutual Fund Manager Skill & Performance Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CAPM, Carhart Multi-Factor APT & Circular Block Bootstrap Significance Framework</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pipeline-box">
    <b>Econometric Architecture:</b> Synchronous-Interval Return Alignment ➔ Single-Factor CAPM OLS (Newey-West HAC inference) ➔ Politis & Romano Circular Block Bootstrap Resampling (B={facts['n_bootstrap']:,}, b={facts['block_length']}) ➔ Carhart 4-Factor (Mkt-RF, SMB, HML, Momentum) APT Decomposition ➔ Chronological 4-Regime Subperiod Persistence Audit ➔ Deterministic Evidence Synthesis Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="notice-card">
    <b>Regulatory & Attribution Advisory:</b> All evaluated schemes are Direct Plan - Growth. Classifications communicate statistical evidence strength under the specified model; they are not forward return forecasts or investment advice. Twelve hand-picked schemes over a ~5-year window cannot support general conclusions, no multiple-testing correction is built into the rules (Benjamini-Hochberg q-values are shown), and conclusions depend on the market proxy chosen (see the Leaderboard tab).
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Executive Summary: KPI Layout
    # -------------------------------------------------------------------------
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

    total_funds = len(final_df)
    max_score = final_df["total_score"].max()
    min_score = final_df["total_score"].min()
    median_score = final_df["total_score"].median()

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Universe Size</div><div class="metric-value">{total_funds}</div></div>', unsafe_allow_html=True)
    with col2:
        skill_signal_count = cat_counts["Strong Evidence of Skill"] + cat_counts["Strong Skill Candidate (Preliminary)"]
        st.markdown(f'<div class="metric-card"><div class="metric-label">Skill Candidates Identified</div><div class="metric-value" style="color:#047857;">{skill_signal_count}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confirmed Strong Evidence</div><div class="metric-value" style="color:#15803D;">{cat_counts["Strong Evidence of Skill"]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Moderate Evidence</div><div class="metric-value" style="color:#1D4ED8;">{cat_counts["Moderate Evidence"]}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Positive but Inconclusive</div><div class="metric-value" style="color:#0369A1;">{cat_counts["Positive but Inconclusive"]}</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Weak / Inconclusive</div><div class="metric-value" style="color:#B45309;">{cat_counts["Weak/Inconclusive"]}</div></div>', unsafe_allow_html=True)
    with col7:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Style / Luck Driven</div><div class="metric-value" style="color:#B91C1C;">{cat_counts["Likely Style/Luck Driven"]}</div></div>', unsafe_allow_html=True)
    st.caption("Skill Candidates Identified includes confirmed evidence plus preliminary candidates. A preliminary candidate is a promising statistical signal, not proof of persistent manager skill.")

    bsens = data.get("benchmark_sensitivity")
    if bsens is not None:
        at_least_moderate = (bsens[bsens["final_category"].isin(["Strong Evidence of Skill", "Strong Skill Candidate (Preliminary)", "Moderate Evidence"])]
                             .groupby("market_proxy").size().reindex(bsens["market_proxy"].unique(), fill_value=0))
        st.caption("Robustness - funds with at least Moderate evidence under each market proxy: "
                   + " · ".join(f"{proxy}: {int(n)}" for proxy, n in at_least_moderate.items())
                   + ". The headline counts above use the primary proxy; see the Leaderboard tab.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Clean Chart & Score Summary
    sum_col1, sum_col2 = st.columns([3, 2])
    with sum_col1:
        cat_data = pd.DataFrame({
            "Category": all_categories,
            "Count": [cat_counts[c] for c in all_categories]
        })
        formal_colors = ["#15803D", "#047857", "#1D4ED8", "#0369A1", "#B45309", "#B91C1C"]
        fig_cat = px.bar(
            cat_data, x="Category", y="Count", text="Count",
            color="Category",
            color_discrete_map=dict(zip(all_categories, formal_colors)),
            title="Classification Distribution Across Universe"
        )
        _style_fig(fig_cat, 270)
        fig_cat.update_layout(
            showlegend=False,
            yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(size=10),
                       range=[0, max(1, cat_data["Count"].max()) * 1.25],
                       dtick=1 if cat_data["Count"].max() <= 6 else 2),
            xaxis=dict(tickfont=dict(size=10), title=None),
        )
        fig_cat.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=12),
                              marker_line_width=1, marker_line_color="#E2E8F0")
        st.plotly_chart(fig_cat, width="stretch")

    with sum_col2:
        st.markdown("#### SCORE RANGE & STATISTICAL PROFILE")
        top_row = final_df.loc[final_df['total_score'].idxmax()]
        top_meta = meta_dict.get(str(top_row['symbol']), {})
        st.markdown(f"- **Highest Evidence Index**: `{max_score:.1f} / 100` ({top_meta.get('short_name', top_row['symbol'])})")
        st.markdown(f"- **Median Universe Index**: `{median_score:.1f} / 100`")
        st.markdown(f"- **Lowest Evidence Index**: `{min_score:.1f} / 100`")
        st.markdown("""
        The evidence index is a transparent five-component rubric: performance (0-20) + significance (0-25) + factor survival (0-25) + persistence (0-20) + a style/luck adjustment (-10 to +10). It ranks evidence strength; the category, not the score, drives the classification.
        """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Selected Scheme Deep-Dive
    # -------------------------------------------------------------------------
    fund_eval = final_df[final_df["symbol"] == selected_symbol].iloc[0]
    fund_capm = capm_df[capm_df["symbol"] == selected_symbol].iloc[0]
    fund_boot = boot_df[boot_df["symbol"] == selected_symbol].iloc[0]
    fund_fact = factor_df[factor_df["symbol"] == selected_symbol].iloc[0]
    fund_pers = pers_sum_df[pers_sum_df["symbol"] == selected_symbol].iloc[0]

    fund_full_name = curr_meta.get("fund_name", f"Scheme {selected_symbol}")
    manager_name = curr_meta.get("lead_manager", "Lead Portfolio Manager")
    amc_name = curr_meta.get("amc", "Asset Management Company")
    category_name = curr_meta.get("category", "Equity")
    tenure_start = curr_meta.get("tenure_start", "N/A")
    ter = curr_meta.get("expense_ratio", "N/A")

    st.markdown(f"### DIAGNOSTIC REPORT: {fund_full_name}")

    # Formal Metadata Bar
    st.markdown(f"""
    <div class="profile-container">
        <span class="profile-item-label">Named Manager:</span> <span class="profile-item-value">{manager_name}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="profile-item-label">AMC:</span> <span class="profile-item-value">{amc_name}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="profile-item-label">SEBI Category:</span> <span class="profile-item-value">{category_name}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="profile-item-label">Manager Since:</span> <span class="profile-item-value">{tenure_start}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="profile-item-label">Direct TER (current):</span> <span class="profile-item-value">{ter}%</span>
    </div>
    """, unsafe_allow_html=True)

    attribution_note = str(fund_eval.get("attribution_note", "") or "")
    if attribution_note and attribution_note.lower() != "nan":
        st.warning(f"**Attribution caveat:** {attribution_note}")

    head_col1, head_col2, head_col3 = st.columns([3, 1, 1])
    with head_col1:
        badge_html = get_category_badge(fund_eval["final_category"])
        st.markdown(f"**Classification Status:** &nbsp; {badge_html}", unsafe_allow_html=True)
    with head_col2:
        st.metric("Evidence Index", f"{fund_eval['total_score']:.1f} / 100")
    with head_col3:
        st.metric("Rule Version", str(fund_eval.get("evaluation_version", "2.0.0")))

    st.caption(f"**Attribution Summary**: {fund_eval['final_explanation']}")

    # 4-Pillar Decision Matrix
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### PRIMARY ECONOMETRIC DECISION MATRIX")
    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)

    with ev_col1:
        capm_pos = bool(fund_eval.get("capm_positive", False))
        capm_sig = bool(fund_eval.get("capm_significant", False))
        st.markdown("**1. Single-Factor CAPM**")
        st.markdown(f"Excess Alpha > 0: {get_flag_indicator(capm_pos, 'POSITIVE', 'NON-POSITIVE')}", unsafe_allow_html=True)
        st.markdown(f"HAC Significance (alpha>0): {get_flag_indicator(capm_sig, 'p < 0.05', 'p >= 0.05')}", unsafe_allow_html=True)

    with ev_col2:
        boot_sig = bool(fund_eval.get("bootstrap_significant_flag", fund_eval.get("bootstrap_significant", False)))
        boot_ci_pos = bool(fund_eval.get("bootstrap_ci_positive", False))
        st.markdown("**2. Block Bootstrap (CBB)**")
        st.markdown(f"Significantly Positive: {get_flag_indicator(boot_sig, 'p < 0.05', 'p >= 0.05')}", unsafe_allow_html=True)
        st.markdown(f"95% CI Lower Bound: {get_flag_indicator(boot_ci_pos, '> 0', '<= 0')}", unsafe_allow_html=True)

    with ev_col3:
        fact_surv = bool(fund_eval.get("factor_alpha_survives", False))
        red_pct = float(fund_eval.get("alpha_reduction_pct", 0.0))
        red_pts = float(fund_eval.get("alpha_reduction_pts_annual", np.nan))
        st.markdown("**3. Carhart 4-Factor Model**")
        st.markdown(f"Factor Survival: {get_flag_indicator(fact_surv, 'SURVIVES', 'ATTRIBUTED', is_warning=True)}", unsafe_allow_html=True)
        pts_txt = "" if np.isnan(red_pts) else f" ({red_pts:+.2f} pts/yr)"
        st.markdown(f"Alpha Absorption: <span class='mono-val'>{red_pct:.1f}%{pts_txt}</span>", unsafe_allow_html=True)

    with ev_col4:
        pers_maj = bool(fund_eval.get("persistent_positive_majority", fund_eval.get("alpha_positive_majority", False)))
        pers_rate = fund_eval.get("positive_alpha_persistence_rate", 0.0)
        st.markdown("**4. Intertemporal Persistence**")
        st.markdown(f"Majority Positive (>=3 of 4): {get_flag_indicator(pers_maj, 'PERSISTENT', 'DECAYED')}", unsafe_allow_html=True)
        st.markdown(f"Positive-Alpha Subperiods: <span class='mono-val'>{pers_rate*100:.0f}%</span>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Formal Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "CAPM REGRESSION",
        "BLOCK BOOTSTRAP RESAMPLING",
        "CARHART 4-FACTOR DECOMPOSITION",
        "SUBPERIOD PERSISTENCE",
        "UNIVERSE LEADERBOARD"
    ])

    with tab1:
        st.markdown(f"##### CAPM Single-Factor Specification vs {facts['market_label']}")
        alpha_ann = fund_capm.get("alpha_annual_pct", 0.0)
        p_hac = fund_capm.get("alpha_pvalue_hac", np.nan)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized Jensen Alpha (α)", fmt_pct(alpha_ann, include_sign=True))
        c2.metric("Market Sensitivity (β)", fmt_val(fund_capm.get("beta", 1.0)))
        c3.metric("Coefficient of Determination (R²)", fmt_val(fund_capm.get("r_squared", 0.0)))
        c4.metric("Alpha p-value (HAC)", fmt_p(p_hac))
        c4.caption(p_label(p_hac, alpha_positive=bool(alpha_ann > 0)))

        capm_df_table = pd.DataFrame({
            "Specification Metric": [
                "Annualized Alpha (Jensen's α)",
                "Daily Intercept Parameter",
                "Market Beta (β)",
                "Alpha Standard Error (OLS)",
                "Alpha Standard Error (Newey-West HAC)",
                "t-Statistic (OLS)",
                "t-Statistic (HAC)",
                "Two-Tailed p-value (OLS, i.i.d. errors)",
                "Two-Tailed p-value (HAC) - used in decisions",
                "R-Squared (R²)",
                "Durbin-Watson Autocorrelation",
                "Jarque-Bera Normality Test p-value"
            ],
            "Estimate": [
                fmt_pct(alpha_ann, include_sign=True),
                f"{fund_capm.get('alpha_daily', 0.0):.6f}",
                f"{fund_capm.get('beta', 1.0):.4f}",
                f"{fund_capm.get('alpha_stderr', 0.0):.6f}",
                f"{fund_capm.get('alpha_stderr_hac', np.nan):.6f}",
                f"{fund_capm.get('alpha_tstat', 0.0):.3f}",
                f"{fund_capm.get('alpha_tstat_hac', np.nan):.3f}",
                fmt_pval(fund_capm.get("alpha_pvalue", 1.0)),
                fmt_pval(p_hac),
                f"{fund_capm.get('r_squared', 0.0):.4f}",
                f"{fund_capm.get('durbin_watson', 2.0):.3f}",
                f"{fund_capm.get('jarque_bera_pvalue', 0.0):.4e}"
            ]
        })
        st.dataframe(capm_df_table, width="stretch", hide_index=True)

        alt = data.get("capm_nifty_price")
        if alt is not None and (alt["symbol"] == selected_symbol).any():
            a_alt = float(alt.loc[alt["symbol"] == selected_symbol, "alpha_annual_pct"].iloc[0])
            st.caption(
                f"Sensitivity: against the NIFTY 500 *price* index (dividends excluded) the same fund shows an "
                f"annual alpha of {a_alt:+.2f}% — upward-biased by the index dividend yield, which is why it is not the primary benchmark."
            )

        desc = data.get("descriptive")
        if desc is not None:
            st.markdown("##### Return & Risk Profile (full sample, daily data annualised)")
            sel = desc[desc["series"].isin([selected_symbol, "iima_market_return", "nifty500_price_return"])].copy()
            label_map = {selected_symbol: "This fund",
                         "iima_market_return": "IIMA market (total return)",
                         "nifty500_price_return": "NIFTY 500 price index"}
            sel["Series"] = sel["series"].map(label_map)
            sel = sel.rename(columns={
                "annualized_return_cagr_pct": "CAGR %", "annualized_volatility_pct": "Volatility %",
                "sharpe_ratio": "Sharpe", "max_drawdown_pct": "Max drawdown %"})
            st.dataframe(sel[["Series", "CAGR %", "Volatility %", "Sharpe", "Max drawdown %"]].round(2),
                         width="stretch", hide_index=True)

    with tab2:
        st.markdown("##### Non-Parametric Circular Block Bootstrap Testing (Politis & Romano, Kosowski 2006)")
        boot_pval = fund_boot.get("p_value", 1.0)
        boot_p1 = fund_boot.get("p_value_one_sided", np.nan)
        boot_bh = fund_boot.get("p_value_bh", np.nan)
        boot_ci_low = fund_boot.get("ci_annual_lower_pct", 0.0)
        boot_ci_up = fund_boot.get("ci_annual_upper_pct", 0.0)

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Two-sided p-value", fmt_p(boot_pval))
        b1.caption(p_label(boot_pval, alpha_positive=bool(fund_boot.get("observed_alpha_daily", 0) > 0)))
        b2.metric("One-sided p (H1: α > 0)", fmt_p(boot_p1))
        b3.metric("FDR-adjusted p (BH)", fmt_p(boot_bh))
        b3.caption(f"Benjamini-Hochberg across {len(boot_df)} funds")
        b4.metric("95% CI lower (annual α)", f"{boot_ci_low:+.1f}%")
        b5.metric("95% CI upper (annual α)", f"{boot_ci_up:+.1f}%")

        positive_flag = bool(fund_boot.get("significantly_positive_5pct", False))
        if positive_flag:
            sig_status = "Alpha is significantly POSITIVE at the 5% level under residual dependency."
        elif bool(fund_boot.get("significant_at_5pct", False)):
            sig_status = "Alpha differs from zero at the 5% level but is NEGATIVE — evidence of under-performance, not skill."
        else:
            sig_status = "Fails the 5% test; the data cannot reject zero alpha."
        block_len = int(fund_boot.get("block_length", 30))
        n_boot = int(fund_boot.get("n_bootstrap", 1000))
        st.markdown(f"""
        - **Resampling Method**: Circular Block Bootstrap (CBB) of paired fund/market returns, block length $b = {block_len}$ trading days (the $T^{{1/3}} \\approx 11$ rule of thumb is deliberately exceeded; sensitivity shown below).
        - **Bootstrap Iterations**: $B = {n_boot:,}$ resamples preserving autocorrelation and volatility clustering.
        - **Hypothesis Test**: bootstrap distribution re-centred on zero ($H_0: \\alpha = 0$, two-sided). A fund counts as significantly positive only if $p < 0.05$ **and** the estimated alpha is positive.
        - **Decision**: {sig_status}
        """)

        sens = data.get("block_sensitivity")
        if sens is not None and (sens["symbol"] == selected_symbol).any():
            s_sub = sens[sens["symbol"] == selected_symbol].sort_values("block_length")
            fig_b = px.line(s_sub, x="block_length", y="p_value_two_sided", markers=True,
                            labels={"block_length": "Block length b (b=1: i.i.d. bootstrap)",
                                    "p_value_two_sided": "Two-sided bootstrap p-value"},
                            title="Sensitivity of the bootstrap p-value to block length")
            fig_b.add_hline(y=0.05, line_dash="dash", line_color="#B91C1C", annotation_text="5%")
            _style_fig(fig_b, 300)
            fig_b.update_traces(line_color="#1E3A8A")
            fig_b.update_xaxes(type="category", categoryorder="array",
                               categoryarray=[str(int(b)) for b in s_sub["block_length"]])
            fig_b.update_yaxes(range=[0, max(0.12, float(s_sub["p_value_two_sided"].max()) * 1.15)],
                               gridcolor="#F1F5F9")
            st.plotly_chart(fig_b, width="stretch")

    with tab3:
        st.markdown("##### Multi-Factor Attribution (IIM Ahmedabad Academic Series)")
        beta_mkt_val = fund_fact.get("beta_market", 1.0)
        beta_smb_val = fund_fact.get("beta_smb", 0.0)
        beta_hml_val = fund_fact.get("beta_hml", 0.0)
        beta_mom_val = fund_fact.get("beta_momentum", 0.0)
        p_fac_hac = fund_fact.get("alpha_pvalue_hac", np.nan)

        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("4-Factor Annualized Alpha", fmt_pct(fund_fact.get("alpha_annual_pct", 0.0), include_sign=True))
        f1.caption(f"HAC p = {fmt_p(p_fac_hac)} ({p_label(p_fac_hac, alpha_positive=bool(fund_fact.get('alpha_annual_pct', 0) > 0))})")
        f2.metric("Size Exposure (SMB)", fmt_val(beta_smb_val))
        f3.metric("Value Exposure (HML)", fmt_val(beta_hml_val))
        f4.metric("Momentum Exposure (WML)", fmt_val(beta_mom_val))
        f5.metric("R² (4-factor)", fmt_val(fund_fact.get("r_squared", 0.0)))

        factors = ["Market Excess (Mkt-RF)", "Size Tilt (SMB)", "Value Tilt (HML)", "Momentum Tilt (WML)"]
        keys = ["beta_market", "beta_smb", "beta_hml", "beta_momentum"]
        loadings = [float(fund_fact[k]) for k in keys]
        lo = [float(fund_fact[f"{k}_ci_lower"]) for k in keys]
        hi = [float(fund_fact[f"{k}_ci_upper"]) for k in keys]
        fig_factors = go.Figure(go.Bar(
            x=factors, y=loadings, marker_color="#1E3A8A", marker_line_width=1, marker_line_color="#CBD5E1",
            error_y=dict(type="data", symmetric=False,
                         array=[h - v for h, v in zip(hi, loadings)],
                         arrayminus=[v - l for v, l in zip(loadings, lo)],
                         color="#475569"),
        ))
        fig_factors.update_layout(title="Systematic Factor Exposures (95% OLS confidence intervals)",
                                  yaxis_title="Regression Coefficient (Beta)", xaxis_title=None,
                                  yaxis=dict(gridcolor="#F1F5F9"))
        _style_fig(fig_factors, 300)
        st.plotly_chart(fig_factors, width="stretch")

        load_table = pd.DataFrame({
            "Factor": factors,
            "Loading": [f"{v:.4f}" for v in loadings],
            "t-stat": [f"{float(fund_fact[k + '_tstat']):.2f}" for k in keys],
            "p-value (OLS)": [fmt_p(float(fund_fact[k + '_pvalue'])) for k in keys],
        })
        st.dataframe(load_table, width="stretch", hide_index=True)
        st.caption(
            f"Alpha bridge: CAPM α {float(fund_capm['alpha_annual_pct']):+.2f}% → 4-factor α "
            f"{float(fund_fact['alpha_annual_pct']):+.2f}% "
            f"(factor tilts account for {float(fund_eval.get('alpha_reduction_pts_annual', np.nan)):+.2f} pts/yr). "
            "Both models use the same IIMA market series, so the difference isolates SMB/HML/momentum exposure."
        )

    with tab4:
        st.markdown("##### Intertemporal Performance Stability Across Chronological Regimes")
        p_sub = pers_per_df[pers_per_df["symbol"] == selected_symbol].sort_values("period_id")

        fig_pers = px.bar(
            p_sub, x="period_id", y="alpha_annual_pct",
            labels={"period_id": "Chronological Subperiod", "alpha_annual_pct": "Annualized Alpha (%)"},
            title="Subperiod Annualized Alpha Evolution"
        )
        _style_fig(fig_pers, 280)
        fig_pers.update_layout(yaxis=dict(gridcolor="#F1F5F9"))
        fig_pers.update_traces(marker_color="#0F766E", marker_line_width=1, marker_line_color="#CBD5E1")
        st.plotly_chart(fig_pers, width="stretch")

        p_cols = [c for c in ["period_id", "start_date", "end_date", "n_observations", "alpha_annual_pct",
                              "beta", "alpha_pvalue", "alpha_pvalue_hac"] if c in p_sub.columns]
        p_names = {"period_id": "Period", "start_date": "Start", "end_date": "End", "n_observations": "Obs.",
                   "alpha_annual_pct": "Alpha % p.a.", "beta": "Beta", "alpha_pvalue": "p (OLS)",
                   "alpha_pvalue_hac": "p (HAC)"}
        st.dataframe(p_sub[p_cols].round(4).rename(columns=p_names), width="stretch", hide_index=True,
                     height=_table_height(len(p_sub)))
        st.caption("A 'majority' means strictly more than half of the four subperiods (at least 3). "
                   "Four short windows give limited statistical power; treat persistence as descriptive.")

    with tab5:
        st.markdown("##### Cross-Sectional Universe Leaderboard & Skill Ranking")
        leaderboard_df = final_df.copy()
        leaderboard_df["Scheme Identifier"] = leaderboard_df["symbol"]
        leaderboard_df["Fund Name"] = leaderboard_df["symbol"].apply(lambda s: meta_dict.get(str(s), {}).get("short_name", s))
        leaderboard_df["Named Manager"] = leaderboard_df["symbol"].apply(lambda s: meta_dict.get(str(s), {}).get("lead_manager", "N/A"))
        leaderboard_df["SEBI Category"] = leaderboard_df["symbol"].apply(lambda s: meta_dict.get(str(s), {}).get("category", "N/A"))
        leaderboard_df["Index (0-100)"] = leaderboard_df["total_score"]
        leaderboard_df["Classification"] = leaderboard_df["final_category"]
        board_cols = ["Scheme Identifier", "Fund Name", "Named Manager", "SEBI Category", "Index (0-100)", "Classification"]
        if "tenure_overlap_pct" in leaderboard_df.columns:
            leaderboard_df["Manager in charge (% of sample)"] = leaderboard_df["tenure_overlap_pct"].round(0)
            board_cols.append("Manager in charge (% of sample)")

        st.dataframe(
            leaderboard_df[board_cols].sort_values("Index (0-100)", ascending=False),
            width="stretch",
            hide_index=True,
            height=_table_height(len(leaderboard_df)),
        )

        bsens = data.get("benchmark_sensitivity")
        if bsens is not None:
            st.markdown("##### Robustness: classification under alternative market proxies")
            short = {"Strong Evidence of Skill": "Strong", "Strong Skill Candidate (Preliminary)": "Potential",
                     "Moderate Evidence": "Moderate", "Positive but Inconclusive": "Positive?",
                     "Weak/Inconclusive": "Weak", "Likely Style/Luck Driven": "Style/Luck"}
            piv = bsens.assign(cell=bsens["final_category"].map(short) + " (" + bsens["total_score"].round(0).astype(int).astype(str) + ")") \
                       .pivot(index="symbol", columns="market_proxy", values="cell")
            piv.insert(0, "Fund", [meta_dict.get(str(s), {}).get("short_name", s) for s in piv.index])
            piv = piv.reset_index().rename(columns={"symbol": "Scheme"})
            st.dataframe(piv, width="stretch", hide_index=True, height=_table_height(len(piv)))
            st.caption(
                "Same rules, different market portfolio. 'nifty500_price' excludes dividends (alpha biased upward); "
                "'..._plus_assumed_dividend_yield' adds an ASSUMED 1.3% p.a. yield as a rough TRI stand-in; "
                "'iima_market' is the primary, total-return, same-market-as-4-factor specification. "
                "Conclusions that change with the proxy should not be treated as established."
            )

    with st.expander("Methodology notes, assumptions and limitations"):
        st.markdown("""
        - **Market proxy**: the CAPM and the 4-factor model use the *same* IIMA market portfolio, so the alpha reduction isolates size/value/momentum tilts. The NIFTY 500 TRI is not freely downloadable; drop a TRI file at `data/raw/raw_benchmark_nifty500_tri.csv` to use it.
        - **Inference**: significance uses Newey-West HAC standard errors; the bootstrap provides a non-parametric check. Several funds change significance with the bootstrap block length.
        - **Multiple testing**: 12 funds are tested at 5%; BH q-values are reported but the classification rules do not use them.
        - **Selection / survivorship**: the 12 funds are hand-picked, well-known survivors; results are biased towards flattering outcomes.
        - **Manager attribution**: a scheme's record is the AMC's, not necessarily the named manager's (see the attribution caveat). Manager names/tenures are curated registry data, not machine-verified.
        - **Model scope**: India-only factors cannot capture overseas holdings, large cash/arbitrage books, or other risk premia.
        - **Causality**: a statistically significant alpha is not proof of manager skill.
        """)


if __name__ == "__main__":
    main()
