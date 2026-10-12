"""
live_search_page.py — The "Live Fund Analyser" Page
=====================================================
A new Streamlit page that:
  1. Lets the user search any of 2,500+ Indian funds or any US ticker
  2. Fetches live NAV/price data on the fly
  3. Runs the full 5-pillar econometric pipeline in memory
  4. Displays: Evidence Class, Key Stats, Fee Leakage Calculator,
     SHAP Radar, Rolling Alpha, and 1-click PDF download

Called from app.py via:
    from live_search_page import render_live_search_page
    render_live_search_page()
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from components.live_ingestion import (
    search_india_funds,
    fetch_india_nav,
    fetch_us_fund,
    fetch_all_scheme_codes,
    get_fund_meta_from_api,
)
from components.fee_calculator import (
    compute_wealth_paths,
    plot_wealth_paths,
    plot_annual_fee_drag,
    fee_summary_metrics,
)
from components.shap_attribution import (
    run_shap_analysis,
    plot_factor_radar,
    plot_return_waterfall,
    shap_summary_table,
    SHAP_AVAILABLE,
)
from components.rolling_analysis import (
    compute_rolling_capm,
    plot_alpha_beta_combined,
    rolling_summary_stats,
)
from components.report_generator import build_report_data, generate_pdf_bytes

# Category colours for badge rendering
CATEGORY_COLORS = {
    "Strong Evidence of Skill":              "#166534",
    "Strong Skill Candidate (Preliminary)":  "#059669",
    "Moderate Evidence":                     "#2563eb",
    "Positive but Inconclusive":             "#0ea5e9",
    "Weak/Inconclusive":                     "#64748b",
    "Likely Style/Luck Driven":              "#d97706",
    "Unclassified":                          "#94a3b8",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _badge(label: str) -> str:
    color = CATEGORY_COLORS.get(label, "#94a3b8")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 14px;'
        f'border-radius:999px;font-weight:700;font-size:.88rem">{label}</span>'
    )

def _pct(v, digits=2):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{float(v):+.{digits}f}%"

def _p(v):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return "<0.001" if float(v) < 0.001 else f"{float(v):.3f}"


# ─── Main page renderer ────────────────────────────────────────────────────────

def render_live_search_page() -> None:
    """Renders the full Live Fund Analyser page."""

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:1.2rem 1.5rem;border-radius:14px;
                background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;margin-bottom:1rem">
      <h2 style="margin:0 0 .3rem 0;color:#fff;font-size:1.6rem">🔍 Live Fund Analyser</h2>
      <p style="margin:0;color:#93c5fd;font-size:.95rem">
        Search any of 2,500+ Indian mutual funds or any US fund/ETF ticker —
        run the full 5-pillar econometric analysis in real time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Market selector ────────────────────────────────────────────────────────
    market = st.radio(
        "Select Market",
        ["🇮🇳 India (AMFI / mfapi.in)", "🇺🇸 USA (Yahoo Finance)"],
        horizontal=True,
        key="live_market_selector",
    )
    is_india = market.startswith("🇮🇳")

    # ── Fund search ─────────────────────────────────────────────────────────────
    if is_india:
        col_search, col_info = st.columns([3, 1])
        with col_search:
            search_q = st.text_input(
                "Search Indian Fund",
                placeholder="e.g.  parag parikh   or   hdfc flexi   or   quant small",
                key="india_search_input",
            )
        with col_info:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Searches AMFI scheme names (Direct Plan preferred)")

        scheme_code  = None
        display_name = None

        if search_q and len(search_q) >= 3:
            with st.spinner("Searching AMFI fund database..."):
                results = search_india_funds(search_q)

            if not results:
                st.warning(f"No funds found matching **{search_q}**. Try a different keyword.")
            else:
                options = {f"{name}  [{code}]": code for code, name in results.items()}
                chosen = st.selectbox(
                    f"Select from {len(options)} results",
                    list(options.keys()),
                    key="india_fund_select",
                )
                scheme_code  = options[chosen]
                display_name = chosen.split("  [")[0]

    else:
        col_ticker, col_hint = st.columns([2, 2])
        with col_ticker:
            ticker = st.text_input(
                "Enter US Fund / ETF Ticker",
                placeholder="e.g.  FCNTX  or  PRGFX  or  SPY",
                key="us_ticker_input",
            ).strip().upper()
        with col_hint:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Use Yahoo Finance ticker symbols (mutual funds, ETFs, indices)")

        scheme_code  = ticker if ticker else None
        display_name = ticker if ticker else None

    # ── Date range ──────────────────────────────────────────────────────────────
    col_d1, col_d2, col_run = st.columns([1, 1, 1])
    with col_d1:
        start_date = st.text_input("Start Date", value="2019-01-01", key="live_start")
    with col_d2:
        end_date = st.text_input("End Date", value="2023-12-31", key="live_end")
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Full Analysis", type="primary", use_container_width=True,
                            disabled=scheme_code is None)

    if not scheme_code or not run_btn:
        if not scheme_code:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem;color:#64748b;font-size:.95rem">
              ☝️ Search for a fund above and click <strong>Run Full Analysis</strong> to begin.
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Fetch data + run pipeline ───────────────────────────────────────────────
    with st.spinner(f"Fetching {'NAV' if is_india else 'price'} data for **{display_name}**..."):
        if is_india:
            nav_df = fetch_india_nav(scheme_code, start=start_date, end=end_date)
        else:
            nav_df = fetch_us_fund(scheme_code, start=start_date, end=end_date)

    if nav_df.empty:
        st.error(f"Could not fetch data for **{display_name}** in the date range {start_date} → {end_date}.")
        return

    st.success(f"✅ Fetched {len(nav_df):,} NAV observations for **{display_name}** ({start_date} → {end_date})")

    # Run in-memory pipeline
    with st.spinner("Running 5-pillar econometric analysis (CAPM → Bootstrap → 4-Factor → Persistence → Classification)..."):
        try:
            if is_india:
                from components.live_pipeline import run_india_live
                result = run_india_live(nav_df, fund_name=display_name,
                                         start=start_date, end=end_date)
            else:
                from components.live_pipeline import run_usa_live
                result = run_usa_live(nav_df, fund_name=display_name,
                                       start=start_date, end=end_date)
        except Exception as exc:
            result = {"error": str(exc)}

    if result.get("error"):
        st.error(f"**Pipeline Error**: {result['error']}")
        st.info("💡 The live analysis requires the benchmark and factor CSV files from the initial pipeline run. "
                "Make sure you have run the India or USA market pipeline at least once.")
        return

    # ─── Results section ────────────────────────────────────────────────────────
    ev      = result.get("evaluation", {})
    capm    = result.get("capm", {})
    boot    = result.get("bootstrap", {})
    factor  = result.get("factor", {})
    pers    = result.get("persistence", {})
    aligned = result.get("aligned_df", pd.DataFrame())

    evidence_class = ev.get("evidence_class", "Unclassified")
    score          = ev.get("score", 0)
    explanation    = ev.get("explanation", "")

    st.divider()

    # Evidence badge + score
    col_badge, col_score, col_time = st.columns([3, 1, 1])
    with col_badge:
        st.markdown(f"### {_badge(evidence_class)}", unsafe_allow_html=True)
        st.caption(explanation[:180] + "..." if len(explanation) > 180 else explanation)
    with col_score:
        st.metric("Evidence Score", f"{score}/100")
    with col_time:
        st.metric("Pipeline Time", f"{result.get('elapsed_sec', 0):.1f}s")

    st.divider()

    # ── Key Stats Row ───────────────────────────────────────────────────────────
    st.subheader("📊 Core Statistical Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    capm_alpha = capm.get("alpha_annual_pct", 0.0) or 0.0
    factor_alpha_d = factor.get("alpha_daily", 0.0) or 0.0
    factor_alpha = factor_alpha_d * 252 * 100
    capm_p    = float(capm.get("alpha_pvalue_hac", 1.0) or 1.0)
    boot_p    = float(boot.get("bootstrap_pvalue", 1.0) or 1.0)
    beta_val  = float(capm.get("beta", 1.0) or 1.0)

    c1.metric("CAPM Alpha (p.a.)", _pct(capm_alpha),
              delta="positive" if capm_alpha >= 0 else "negative")
    c2.metric("4-Factor Alpha (p.a.)", _pct(factor_alpha),
              delta="positive" if factor_alpha >= 0 else "negative")
    c3.metric("CAPM HAC p-value", _p(capm_p))
    c4.metric("Bootstrap p-value", _p(boot_p))
    c5.metric("Market Beta (β)", f"{beta_val:.3f}")

    # ── Fee Leakage Calculator ──────────────────────────────────────────────────
    st.divider()
    st.subheader("💰 Fee Leakage & Wealth Erosion Calculator")
    st.caption("Enter your investment details to see what active fees cost you over time.")

    currency = "₹" if is_india else "$"
    default_ter = 1.20 if is_india else 0.80

    col_inv, col_ter, col_idx_ter, col_yrs, col_mkt = st.columns(5)
    with col_inv:
        invested = st.number_input(
            f"Investment ({currency})", min_value=10000, max_value=100_000_000,
            value=1_000_000, step=100_000, key="fee_invested"
        )
    with col_ter:
        fund_ter = st.number_input(
            "Fund TER (%/yr)", min_value=0.01, max_value=3.0,
            value=default_ter, step=0.05, key="fee_ter"
        )
    with col_idx_ter:
        index_ter = st.number_input(
            "Index Fund TER (%/yr)", min_value=0.01, max_value=0.5,
            value=0.10, step=0.01, key="fee_index_ter"
        )
    with col_yrs:
        years = st.slider("Projection Years", min_value=5, max_value=30, value=10, key="fee_years")
    with col_mkt:
        mkt_return = st.number_input(
            "Gross Market Return (%/yr)", min_value=1.0, max_value=25.0,
            value=12.0 if is_india else 10.0, step=0.5, key="fee_mkt"
        )

    fee_result = compute_wealth_paths(
        invested=float(invested),
        years=years,
        fund_net_alpha_pct=capm_alpha,
        fund_ter_pct=fund_ter,
        index_ter_pct=index_ter,
        gross_market_return_pct=mkt_return,
        currency_symbol=currency,
    )

    metrics = fee_summary_metrics(fee_result)
    mcols = st.columns(len(metrics))
    for col, m in zip(mcols, metrics):
        if m["delta"]:
            col.metric(m["label"], m["value"], delta=m["delta"],
                       delta_color=m.get("delta_color", "normal"))
        else:
            col.metric(m["label"], m["value"])

    tab_fee1, tab_fee2 = st.tabs(["📈 Wealth Trajectory", "📊 Annual Fee Drag"])
    with tab_fee1:
        st.plotly_chart(plot_wealth_paths(fee_result, fund_name=display_name),
                        use_container_width=True)
    with tab_fee2:
        st.plotly_chart(plot_annual_fee_drag(fee_result), use_container_width=True)

    # ── AI / SHAP Attribution ───────────────────────────────────────────────────
    if not aligned.empty and SHAP_AVAILABLE:
        st.divider()
        st.subheader("🤖 Explainable AI — Factor Attribution (XGBoost + SHAP)")
        st.caption(
            "XGBoost trained on the 4 Carhart factors. SHAP values decompose each day's "
            "return into factor contributions. The residual is the manager's alpha."
        )

        with st.spinner("Computing SHAP values..."):
            shap_result = run_shap_analysis(aligned)

        if shap_result.get("available"):
            col_radar, col_fall = st.columns(2)
            with col_radar:
                st.plotly_chart(plot_factor_radar(shap_result, fund_name=display_name),
                                use_container_width=True)
            with col_fall:
                st.plotly_chart(
                    plot_return_waterfall(shap_result, capm_alpha_annual_pct=capm_alpha,
                                          fund_name=display_name),
                    use_container_width=True
                )
            r2 = shap_result.get("r2", 0.0)
            st.caption(f"XGBoost R² = {r2:.4f} (proportion of return variation explained by the 4 Carhart factors)")
            st.dataframe(shap_summary_table(shap_result), use_container_width=True, hide_index=True)
        else:
            st.info(shap_result.get("error", "SHAP not available."))
    elif not SHAP_AVAILABLE:
        st.info("💡 Install `xgboost` and `shap` to enable the AI Factor Attribution section.")

    # ── Rolling Alpha / Beta ────────────────────────────────────────────────────
    if not aligned.empty:
        st.divider()
        st.subheader("📈 Rolling Alpha & Beta (1-Year Sliding Window)")
        st.caption(
            "Static 5-year alphas hide the story. Did skill vary during the COVID crash, "
            "recovery, or the 2022 rate-hike cycle?"
        )

        with st.spinner("Computing rolling CAPM..."):
            rolling_df = compute_rolling_capm(aligned, window=252)

        if not rolling_df.empty:
            st.plotly_chart(plot_alpha_beta_combined(rolling_df, fund_name=display_name),
                            use_container_width=True)
            stats = rolling_summary_stats(rolling_df)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Avg Rolling Alpha", f"{stats.get('alpha_mean', 0):+.2f}% p.a.")
            s2.metric("% Time Alpha > 0", f"{stats.get('alpha_positive_pct', 0):.0f}%")
            s3.metric("Avg Rolling Beta", f"{stats.get('beta_mean', 1):.3f}")
            s4.metric("Rolling Periods", str(stats.get("n_periods", 0)))
        else:
            st.info("Need at least 504 trading days (2 years) of aligned data for rolling analysis.")

    # ── PDF Download ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Download Institutional Tear-Sheet")
    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        with st.spinner("Generating PDF..."):
            report_data = build_report_data(
                fund_name=display_name,
                market="India" if is_india else "USA",
                result=result,
            )
            pdf_bytes = generate_pdf_bytes(report_data)

        filename = f"diligence_{display_name.replace(' ', '_')[:30]}.pdf"
        mime = "application/pdf" if pdf_bytes and pdf_bytes[:4] == b"%PDF" else "text/html"
        ext  = "pdf" if mime == "application/pdf" else "html"
        st.download_button(
            label="📥 Download Tear-Sheet",
            data=pdf_bytes,
            file_name=f"diligence_{display_name.replace(' ', '_')[:30]}.{ext}",
            mime=mime,
            use_container_width=True,
            type="primary",
        )
    with col_info:
        st.caption(
            "**What's in the tear-sheet?**  "
            "Evidence class badge, score, full pillar results table, "
            "4-factor loadings table, and explanation. "
            "Statistical evidence only — not investment advice."
        )

    st.caption(
        f"Analysis run at {result.get('elapsed_sec', 0):.1f}s | "
        f"Observations: {result.get('n_obs', 0):,} | "
        f"Evaluation v{ev.get('evaluation_version', '2.3.0')} | "
        "Not investment advice."
    )
