"""
Combined Fund Manager Skill Detector - one dashboard for the Indian and US mutual-fund analyses.

The dashboard only READS the saved results of the two market projects (markets/*/data/processed). It never
estimates anything: both markets are classified by one shared rule set, so a class means exactly the same thing
in India and the USA. Every number on screen comes from combined_data.py.

Presentation only (the saved results, the CSV and the rules are untouched): the preliminary tier "Strong Skill
Candidate (Preliminary)" is displayed under the short name "Strong", and "Strong Evidence of Skill" and "Likely
Style/Luck Driven" are displayed only if at least one fund actually falls into them.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import combined_data as cd

ORDER = cd.CATEGORY_ORDER                                  # the full rule set, strongest first (data layer, unchanged)
CANDIDATE = "Strong Skill Candidate (Preliminary)"
UI_NAME = {CANDIDATE: "Strong"}                            # how a class is named on screen
HIDDEN_UNLESS_POPULATED = ("Strong Evidence of Skill", "Likely Style/Luck Driven")   # no card/row/bar unless a fund is in it
COLORS = {c: cd.CATEGORY_INFO[c]["color"] for c in ORDER} | {"Unclassified": "#94a3b8"}
COLORS |= {UI_NAME[c]: COLORS[c] for c in UI_NAME}         # colours also work for the on-screen names
MARKET_COLORS = {"India": "#f59e0b", "USA": "#3b82f6"}
SHORT_NAME = {c: cd.CATEGORY_INFO[c]["short"] for c in ORDER} | {CANDIDATE: "Strong"}   # compact labels for cards and charts
POSITIVE_RANK = ORDER.index("Positive but Inconclusive")   # classes up to here carry some positive evidence
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def ui(label: str) -> str:
    """The name of a class as shown on screen."""
    return UI_NAME.get(label, label)


def shown_classes(labels: "pd.Series | None" = None) -> list[str]:
    """Classes to display, strongest first: the main scale, plus a normally hidden class if a fund falls into it."""
    present = set(labels.dropna()) if labels is not None else set()
    return [c for c in ORDER if c not in HIDDEN_UNLESS_POPULATED or c in present]


# ----------------------------------------------------------------------------------------------------
# data (cached; the cache key changes whenever an output file changes)
# ----------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all(fingerprint: tuple):
    return cd.all_results()


@st.cache_data(show_spinner=False)
def load_sensitivity(fingerprint: tuple):
    return cd.load_benchmark_sensitivity()


# ----------------------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------------------

def pct(value, digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):+.{digits}f}%"


def pval(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def badge(label: str) -> str:
    return (f'<span style="background:{COLORS.get(label, "#94a3b8")};color:#fff;padding:2px 11px;'
            f'border-radius:999px;font-weight:600;font-size:.82rem;white-space:nowrap">{html.escape(ui(label))}</span>')


def style_page() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding: 1.4rem 1.5rem 2.5rem 1.5rem; max-width: 1800px; }
        .hero { padding: 1.3rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#10243a,#1e4960);
                color: #fff; margin-bottom: 1rem; }
        .hero h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .3rem 0; color: #fff; }
        .hero p { margin: 0; color: #d8e7ef; font-size: 1rem; }
        .note { border-left: 4px solid #0f766e; background: rgba(15,118,110,.09); padding: .8rem 1rem;
                border-radius: 8px; margin: .4rem 0 1rem 0; }
        .warnbox { border-left: 4px solid #d97706; background: rgba(217,119,6,.11); padding: .8rem 1rem;
                   border-radius: 8px; margin: .4rem 0 1rem 0; }
        .card { border-top: 5px solid var(--c); background: rgba(128,128,128,.08); border-radius: 10px;
                padding: .65rem .9rem .55rem .9rem; height: 100%; }
        .card .n { font-size: 2rem; font-weight: 700; line-height: 1.1; }
        .card .l { font-size: .84rem; font-weight: 600; }
        .card .m { font-size: .72rem; opacity: .75; }
        table.scale { width: 100%; border-collapse: collapse; font-size: .9rem; }
        table.scale th { text-align: left; padding: .35rem .5rem; border-bottom: 2px solid rgba(128,128,128,.4); }
        table.scale td { padding: .45rem .5rem; border-bottom: 1px solid rgba(128,128,128,.25); vertical-align: top; }
        table.scale td.c { text-align: center; font-weight: 700; width: 3.2rem; }
        table.scale td:nth-child(2) { min-width: 22rem; }
        table.scale td.z { opacity: .35; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------

def class_cards(df: pd.DataFrame, order: list[str]) -> None:
    counts = df["category"].value_counts()
    for col, label in zip(st.columns(len(order)), order):
        with col:
            sub = f"of {len(df)} funds" + (" · preliminary tier" if label == CANDIDATE else "")
            st.markdown(
                f'<div class="card" style="--c:{COLORS[label]}"><div class="n">{int(counts.get(label, 0))}</div>'
                f'<div class="l">{html.escape(SHORT_NAME[label])}</div>'
                f'<div class="m">{sub}</div></div>',
                unsafe_allow_html=True)


def class_footnote(order: list[str]) -> None:
    """One line that keeps the on-screen name 'Strong' honest, and mentions any class that is not shown."""
    text = ("“Strong” is the rule set's preliminary tier (Strong Skill Candidate): it passes the screening checks but not the strict "
            "multiple-testing standard, so it is a screening result, not confirmed skill.")
    absent = [c for c in HIDDEN_UNLESS_POPULATED if c not in order]
    if absent:
        names = " and ".join(f"“{c}”" for c in absent)
        text += f" The rule set also defines {names}; no fund falls into {'it' if len(absent) == 1 else 'them'}, so {'it is' if len(absent) == 1 else 'they are'} not shown."
    st.caption(text)


def scale_table(df: pd.DataFrame, order: list[str]) -> None:
    """The classes in use with a plain-English meaning and the number of funds in each market (zeros shown)."""
    counts = cd.category_counts(df)
    markets = [m for m in cd.MARKETS if m in counts.columns]
    head = "".join(f"<th>{m}</th>" for m in markets) + "<th>Total</th>"
    rows = []
    for label in order:
        cells = "".join(
            f'<td class="c{" z" if counts.loc[label, m] == 0 else ""}">{counts.loc[label, m]}</td>' for m in markets + ["Total"])
        rows.append(f'<tr><td style="white-space:nowrap">{badge(label)}</td>'
                    f'<td>{html.escape(cd.CATEGORY_INFO[label]["meaning"])}</td>{cells}</tr>')
    st.markdown(f'<table class="scale"><tr><th>Class</th><th>What it means</th>{head}</tr>{"".join(rows)}</table>',
                unsafe_allow_html=True)


def class_chart(df: pd.DataFrame, *, by_market: bool, title: str, key: str, order: list[str]) -> None:
    """Horizontal bars for the classes in use (zeros included) in the fixed strongest-first order."""
    counts = cd.category_counts(df).loc[order]
    names = [SHORT_NAME[c] for c in order]
    fig = go.Figure()
    if by_market:
        for market in [m for m in cd.MARKETS if m in counts.columns]:
            fig.add_trace(go.Bar(y=names, x=counts[market], name=market, orientation="h", text=counts[market],
                                 textposition="outside", marker_color=MARKET_COLORS[market], cliponaxis=False))
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(go.Bar(y=names, x=counts["Total"], orientation="h", text=counts["Total"], textposition="outside",
                             marker_color=[COLORS[c] for c in order], cliponaxis=False, showlegend=False))
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Number of funds", dtick=1, rangemode="tozero")
    fig.update_layout(title=title, height=340 if by_market else 300, margin=dict(l=10, r=30, t=50, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key=key)


def fund_table(d: pd.DataFrame, *, show_market: bool = True, key: str = "") -> None:
    if d.empty:
        st.info("No funds to display.")
        return
    d = d.sort_values(["category_rank", "score", "symbol"], ascending=[True, False, True])
    labels = {
        "market": "Market", "symbol": "Code", "short_name": "Fund", "category": "Class", "score": "Score",
        "capm_alpha_pct": "CAPM α %/yr", "factor_alpha_pct": "4-factor α %/yr", "capm_pvalue": "CAPM p",
        "factor_pvalue": "4-factor p", "bootstrap_pvalue": "Bootstrap p", "persistence_label": "Positive periods",
        "caveat": "Caveat",
    }
    view = d[list(labels)].rename(columns=labels).reset_index(drop=True)
    view["Class"] = view["Class"].map(ui)
    if not show_market:
        view = view.drop(columns="Market")
    config = {
        "Market": st.column_config.TextColumn(width=56),
        "Code": st.column_config.TextColumn(width=60, help="Ticker (USA) or AMFI scheme code (India)."),
        "Fund": st.column_config.TextColumn(width=300),
        "Class": st.column_config.TextColumn(width=180, help="Evidence class from the shared rule set. A class describes the strength "
                                                             "of statistical evidence, not the quality of the manager. "
                                                             "“Strong” is a preliminary screening tier, not confirmed skill."),
        "Score": st.column_config.NumberColumn(width=58, format="%d",
                                               help="Evidence score, 0-100: a summary of the evidence. The class, not the score, is the verdict."),
        "CAPM α %/yr": st.column_config.NumberColumn(width=92, format="%+.2f", help="CAPM alpha, % per year: annualised excess return over what market exposure alone explains."),
        "4-factor α %/yr": st.column_config.NumberColumn(width=104, format="%+.2f", help="Four-factor alpha, % per year: the same, after also controlling for size, value and momentum."),
        "CAPM p": st.column_config.NumberColumn(width=66, format="%.3f", help="Newey-West (HAC) p-value for H0: alpha = 0. Below 0.05 = statistically significant."),
        "4-factor p": st.column_config.NumberColumn(width=76, format="%.3f", help="Newey-West (HAC) p-value of the four-factor alpha."),
        "Bootstrap p": st.column_config.NumberColumn(width=84, format="%.3f", help="Two-sided circular block-bootstrap p-value for CAPM alpha."),
        "Positive periods": st.column_config.TextColumn(width=104, help="Positive subperiods: in how many of the 4 equal time windows the CAPM alpha was positive. 3 or 4 counts as persistent; 2 is a tie."),
        "Caveat": st.column_config.TextColumn(width=260, help="Read this before interpreting the row (open the fund's explanation for the full note)."),
    }
    styler = view.style
    mapper = getattr(styler, "map", None) or styler.applymap
    styler = mapper(lambda v: f"color:{COLORS.get(v, 'inherit')};font-weight:600", subset=["Class"])
    st.dataframe(styler, hide_index=True, width="stretch", column_config=config,
                 height=int(min(38 + 35 * len(view), 900)), key=key or None)


def evidence_checklist(row: pd.Series) -> pd.DataFrame:
    def tick(flag) -> str:
        return "✓ yes" if bool(flag) else "✗ no"

    qs = [row.get("capm_q"), row.get("factor_q"), row.get("bootstrap_q")]
    q_ok = all(pd.notna(q) and q < 0.10 for q in qs)
    red = row.get("alpha_reduction_pct", np.nan)
    rows = [
        ("CAPM alpha is positive", tick(row.get("capm_positive")), f"{pct(row['capm_alpha_pct'])} per year"),
        ("CAPM alpha is statistically significant (HAC p < 0.05)", tick(row.get("capm_significant")), f"p = {pval(row['capm_pvalue'])}"),
        ("Bootstrap agrees (p < 0.05 and alpha > 0)", tick(row.get("bootstrap_significant_flag")), f"p = {pval(row['bootstrap_pvalue'])}"),
        ("Four-factor alpha is positive and significant (p < 0.05)", tick(row.get("factor_alpha_survives")),
         f"{pct(row['factor_alpha_pct'])} per year, p = {pval(row['factor_pvalue'])}"),
        ("Alpha positive in a strict majority of subperiods (3 or 4 of 4)", tick(row.get("persistent_positive_majority")),
         f"{row['persistence_label']} positive"),
        ("Passes the multiple-testing check (all three BH q < 0.10)", tick(q_ok),
         "q = " + " / ".join(pval(q) for q in qs) + "  (CAPM / 4-factor / bootstrap)"),
        ("Factor adjustment removes at least half of the CAPM alpha", tick(pd.notna(red) and red >= 50),
         "n/a (CAPM alpha is not a positive baseline)" if pd.isna(red) else f"{red:.0f}% removed"),
    ]
    return pd.DataFrame(rows, columns=["Test", "Result", "Value"])


def fund_detail(row: pd.Series) -> None:
    st.markdown(badge(row["category"]), unsafe_allow_html=True)
    if row["category"] == CANDIDATE:
        st.caption("Preliminary tier: the screening checks are passed, the strict multiple-testing standard is not. See the tests below.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence score", f"{row['score']:.0f} / 100" if pd.notna(row["score"]) else "n/a",
              help="Summary of the evidence. The class is the verdict; the score only ranks funds within it.")
    c2.metric("CAPM alpha", pct(row["capm_alpha_pct"]), help="Per year, after market exposure.")
    c3.metric("4-factor alpha", pct(row["factor_alpha_pct"]), help="Per year, after market, size, value and momentum.")
    c4.metric("Positive subperiods", row["persistence_label"], help="Of 4 equal time windows.")
    if row["caveat_detail"]:
        note(f"<b>Read this first:</b> {html.escape(row['caveat_detail'])}", warn=True)
    st.dataframe(evidence_checklist(row), hide_index=True, width="stretch")
    explanation = row.get("final_explanation", "")
    if isinstance(explanation, str) and explanation.strip():
        st.write(explanation.replace(CANDIDATE, "Strong (preliminary)"))
    for field, title in (("primary_strengths", "Evidence in favour"), ("primary_weaknesses", "Evidence gaps")):
        text = row.get(field, "")
        if isinstance(text, str) and text.strip() and text.strip() != "None identified.":
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {part.strip()}" for part in text.split(";") if part.strip()))


def alpha_dumbbell(df: pd.DataFrame, key: str) -> None:
    d = df.dropna(subset=["capm_alpha_pct", "factor_alpha_pct"]).copy()
    d["label"] = [("IN · " if m == "India" else "US · ") + str(n) for m, n in zip(d["market"], d["short_name"])]
    d = d.sort_values("factor_alpha_pct")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r["capm_alpha_pct"], r["factor_alpha_pct"]], y=[r["label"]] * 2, mode="lines",
                                 line=dict(color="rgba(128,128,128,.55)", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["capm_alpha_pct"], y=d["label"], mode="markers", name="CAPM alpha",
                             marker=dict(symbol="circle-open", size=10, color="#64748b", line=dict(width=2)),
                             hovertemplate="%{y}<br>CAPM alpha %{x:+.2f}% per year<extra></extra>"))
    for label in ORDER:
        part = d[d["category"] == label]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["factor_alpha_pct"], y=part["label"], mode="markers", name=f"4-factor alpha: {ui(label)}",
            marker=dict(symbol="diamond", size=11, color=COLORS[label]),
            customdata=np.stack([part["factor_pvalue"], part["category"].map(ui)], axis=-1),
            hovertemplate="%{y}<br>4-factor alpha %{x:+.2f}% per year<br>HAC p = %{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fig.update_layout(title="Alpha before (circle) and after (diamond) the factor adjustment, % per year",
                      height=max(420, 26 * len(d) + 120), margin=dict(l=10, r=10, t=60, b=10),
                      xaxis_title="Annualised alpha (%)", yaxis_title="", legend=dict(orientation="h", y=-0.08),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key=key)


def approx_halfwidth(df: pd.DataFrame) -> pd.Series:
    """Rough 95% half-width of a single fund's four-factor alpha (percentage points per year), per market.

    Backed out of each fund's own alpha and p-value (z = alpha / SE, so SE = |alpha| / z); a median over funds.
    """
    from scipy.stats import norm
    d = df[["market", "factor_alpha_pct", "factor_pvalue"]].dropna()
    d = d[(d["factor_pvalue"] > 1e-6) & (d["factor_pvalue"] < 0.999999) & (d["factor_alpha_pct"].abs() > 1e-9)]
    se = d["factor_alpha_pct"].abs() / norm.isf(d["factor_pvalue"] / 2.0)
    return (1.96 * se).groupby(d["market"]).median()


# ----------------------------------------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------------------------------------

def rules_banner(df: pd.DataFrame) -> None:
    version = cd.rules_version(df)
    if version:
        st.success(f"One rule set for both markets: all {len(df)} funds are classified by the same rules (evaluation v{version}), "
                   "so a class means the same thing in India and the USA.")
    else:
        st.error("The two markets were NOT classified by the same rule version, so their classes are not comparable. "
                 "Re-run both evaluators before presenting these results.")


def overview(df: pd.DataFrame) -> None:
    hero("Fund Manager Skill Detector", "Do 24 flagship Indian and US mutual funds show skill, or only market exposure, style and luck?")
    if df.empty:
        st.error("No market results loaded. Check the two markets/*/data/processed/final_evaluation.csv files.")
        return
    rules_banner(df)
    order = shown_classes(df["category"])
    words = NUMBER_WORDS.get(len(order), str(len(order)))
    note("<b>How to read this.</b> Each fund is tested four ways (market-adjusted alpha, a block bootstrap, a four-factor alpha, and "
         f"whether the alpha repeats across time windows) and placed in <b>one of {words} evidence classes</b>. A class describes how strong "
         "the statistical evidence is, not how good the manager is. <i>Weak / inconclusive</i> means &ldquo;skill was not shown&rdquo;, "
         "never &ldquo;skill was ruled out&rdquo;.")
    class_cards(df, order)
    class_footnote(order)
    st.write("")
    left, right = st.columns([1.9, 1])
    with left:
        st.subheader(f"The {words} evidence classes")
        scale_table(df, order)
    with right:
        class_chart(df, by_market=True, title="Funds per class", key="ov_classes", order=order)

    positive = df[df["category_rank"] <= POSITIVE_RANK]
    st.subheader("Funds with a positive-evidence class, strongest first")
    st.caption(f"{len(positive)} of {len(df)} funds. The other {len(df) - len(positive)} are Weak/Inconclusive "
               "(see the market pages or the Fund explorer for every fund).")
    fund_table(positive, key="ov_positive")

    st.subheader("Where the alphas come from")
    alpha_dumbbell(df, key="ov_alpha")

    hw = approx_halfwidth(df)
    if not hw.empty:
        spans = ", ".join(f"about ±{v:.1f} points a year in {m}" for m, v in hw.items())
        note("<b>Why most funds are inconclusive.</b> Daily returns over five years pin a single fund's four-factor alpha down only to "
             f"{spans} (95% interval, approximate, backed out of each fund's own alpha and p-value). A true alpha of 2% a year would "
             "therefore rarely be detected. A shortage of strong verdicts is the expected outcome of an honest test, not a failure of the method.")


def market_page(market: str, df: pd.DataFrame) -> None:
    cfg = cd.MARKETS[market]
    sub = df[df["market"] == market]
    hero(f"{market}: fund-by-fund evidence", cfg["universe_note"])
    if sub.empty:
        st.error(f"No results available for {market}.")
        return
    rules_banner(df)
    order = shown_classes(df["category"])
    class_cards(sub, order)
    class_footnote(order)
    st.caption(f"Market used for CAPM and the four-factor model: {cfg['market_note']} Factors: {cfg['factor_note']}")
    if market == "USA":
        note("<b>Universe caution.</b> AMECX is income-oriented, PRHSX is a healthcare-sector fund and PAGRX is a multi-asset "
             "(Permanent Portfolio) strategy. They are exploratory comparators, not clean peers for a broad US equity study. "
             "T. Rowe Price Growth Stock is PRGFX and has not been substituted because its data was not acquired.", warn=True)
    flagged = sub[sub["caveat"] != ""].sort_values(["category_rank", "score"], ascending=[True, False])
    if not flagged.empty:
        items = "".join(f"<li><b>{html.escape(str(r.short_name))}</b>: {html.escape(r.caveat)}</li>" for r in flagged.itertuples())
        note(f"<b>Read before interpreting these funds</b> (details are in each fund's explanation below):<ul style='margin:.3rem 0 0 0'>{items}</ul>", warn=True)
    left, right = st.columns([0.9, 1.1])
    with left:
        class_chart(sub, by_market=False, title=f"{market}: funds per class", key=f"{market}_classes", order=order)
    with right:
        plot = sub.sort_values("factor_alpha_pct").assign(category=lambda x: x["category"].map(ui))
        fig = px.bar(plot, x="factor_alpha_pct", y="short_name", orientation="h", color="category",
                     color_discrete_map=COLORS, category_orders={"category": [ui(c) for c in order]},
                     hover_data={"symbol": True, "factor_pvalue": ":.3f", "short_name": False, "factor_alpha_pct": ":+.2f"},
                     labels={"factor_alpha_pct": "4-factor alpha (% per year)", "short_name": "", "category": "Class",
                             "factor_pvalue": "HAC p"},
                     title="Four-factor alpha by fund")
        fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=max(360, 32 * len(plot) + 110), margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key=f"{market}_alpha")

    st.subheader(f"All {len(sub)} {market} funds")
    fund_table(sub, show_market=False, key=f"{market}_table")

    st.subheader("Why each fund got its class")
    st.caption("Open a fund to see the tests it passed and failed, in plain English.")
    for _, row in sub.sort_values(["category_rank", "score"], ascending=[True, False]).iterrows():
        with st.expander(f"{row['symbol']} · {row['fund_name']} · {ui(row['category'])}"):
            fund_detail(row)


def comparison_page(df: pd.DataFrame) -> None:
    hero("Cross-market comparison", "Same rules, different markets: compare the evidence patterns, not raw alpha league tables")
    rules_banner(df)
    note("<b>What is and is not comparable.</b> The classification rules are identical, so classes and p-values can be compared "
         "directly. The raw alphas cannot be ranked like-for-like: the two markets use different market portfolios, factor sets, "
         "trading calendars and 12-fund samples, and three US funds are non-peers.")
    order = shown_classes(df["category"])
    class_chart(df, by_market=True, title="Funds per class and market", key="cmp_classes", order=order)

    st.subheader("Side-by-side summary")
    rows = []
    for market in cd.MARKETS:
        m = df[df["market"] == market]
        rows.append({
            "Market": market,
            "Funds": len(m),
            "In a positive class": int((m["category_rank"] <= POSITIVE_RANK).sum()),
            "Strong (preliminary)": int((m["category"] == CANDIDATE).sum()),
            "4-factor α > 0": int((m["factor_alpha_pct"] > 0).sum()),
            "4-factor p < 0.05": int((m["factor_pvalue"] < 0.05).sum()),
            "Smallest BH q": float(m["factor_q"].min()),
            "Median CAPM α": float(m["capm_alpha_pct"].median()),
            "Median 4-factor α": float(m["factor_alpha_pct"].median()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Smallest BH q": st.column_config.NumberColumn(format="%.3f", help="Smallest Benjamini-Hochberg q-value of the four-factor alpha across the 12 funds, i.e. after correcting for testing 12 funds at once. Below 0.10 would be needed to pass the strict multiple-testing standard; a value above that is why 'Strong' is only a preliminary tier."),
        "Strong (preliminary)": st.column_config.NumberColumn(help="Funds in the preliminary 'Strong' tier: screening checks passed, strict multiple-testing standard not met."),
        "In a positive class": st.column_config.NumberColumn(help="Funds in Strong, Moderate or Positive but Inconclusive (any class with some positive evidence)."),
        "4-factor α > 0": st.column_config.NumberColumn(help="Funds whose four-factor alpha is positive, whether or not it is statistically significant."),
        "Median CAPM α": st.column_config.NumberColumn(format="%+.2f", help="Median CAPM alpha across the 12 funds, % per year."),
        "Median 4-factor α": st.column_config.NumberColumn(format="%+.2f", help="Median four-factor alpha across the 12 funds, % per year."),
        "4-factor p < 0.05": st.column_config.NumberColumn(help="Number of funds with a significant four-factor alpha, before any multiple-testing correction. With 12 tests, about 0.6 funds per market would pass by chance alone."),
    })

    st.subheader("Fund table")
    selected = st.selectbox("Market", ["India", "USA", "Both markets"], index=2, key="cmp_market")
    names = [ui(c) for c in order]
    chosen = st.multiselect("Show classes", names, default=names, key="cmp_classes_filter")
    view = df[df["category"].map(ui).isin(chosen)]
    if selected != "Both markets":
        view = view[view["market"] == selected]
    st.caption(f"Showing {len(view)} funds.")
    fund_table(view, show_market=selected == "Both markets", key="cmp_table")


def robustness_page(sens: pd.DataFrame | None, df: pd.DataFrame) -> None:
    hero("Benchmark robustness (India)", "How much does the answer depend on which market portfolio the CAPM alpha is measured against?")
    if sens is None or sens.empty:
        st.warning("benchmark_sensitivity.csv was not found for India. Run `python -m src.sensitivity` inside markets/india.")
        return
    note("<b>Why this matters.</b> A fund's CAPM alpha is its excess return over what the chosen market portfolio explains. A price index "
         "(which leaves out dividends) makes every fund look better than a total-return market does. The four-factor alpha does not depend "
         "on this choice. The table re-runs the <i>same rules</i> under each benchmark; the first row is the main result.")
    proxies = list(sens["market_proxy"].unique())
    order = shown_classes(sens["final_category"])        # a normally hidden class appears here if a fund lands in it
    rows = []
    for proxy in proxies:
        counts = sens.loc[sens["market_proxy"] == proxy, "final_category"].value_counts()
        rows.append({"Benchmark": cd.SENSITIVITY_PROXY_LABELS.get(proxy, proxy), **{c: int(counts.get(c, 0)) for c in order}})
    table = pd.DataFrame(rows)
    st.subheader("Funds per class under each benchmark")
    st.dataframe(table.rename(columns={c: SHORT_NAME[c] for c in order}), hide_index=True, width="stretch",
                 column_config={SHORT_NAME[c]: st.column_config.NumberColumn(help=cd.CATEGORY_INFO[c]["meaning"]) for c in order})

    long = table.melt(id_vars="Benchmark", var_name="Class", value_name="Funds")
    long["Class"] = long["Class"].map(ui)
    fig = px.bar(long, x="Funds", y="Benchmark", color="Class", orientation="h", color_discrete_map=COLORS,
                 category_orders={"Class": [ui(c) for c in order]}, title="Classification mix by benchmark (India, 12 funds)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="")
    st.plotly_chart(fig, width="stretch", key="rb_mix")

    st.subheader("Each fund under each benchmark")
    names = df.loc[df["market"] == "India", ["symbol", "short_name"]]
    merged = sens.merge(names, on="symbol", how="left")
    merged["final_category"] = merged["final_category"].map(ui)
    wide =merged.pivot_table(index=["symbol", "short_name"], columns="market_proxy", values="final_category", aggfunc="first").reset_index()
    wide = wide.rename(columns=cd.SENSITIVITY_PROXY_LABELS).rename(columns={"short_name": "Fund", "symbol": "Code"})
    st.dataframe(wide[["Fund", "Code"] + [cd.SENSITIVITY_PROXY_LABELS.get(p, p) for p in proxies]], hide_index=True, width="stretch",
                 height=int(38 + 35 * len(wide) + 3))
    note("<b>Reading it.</b> The benchmark dependence is itself a finding: against a dividend-free price index several funds look "
         "<i>Moderate</i> and one looks <i>Style/Luck driven</i>; against a total-return market they fall back to <i>Positive but "
         "Inconclusive</i> or <i>Weak</i>. The strongest-looking fund (PPFAS Flexi Cap) is rated <i>Strong</i> under every benchmark, but only "
         "as a preliminary tier: its four-factor q-value (0.54) fails the multiple-testing check, so it is a screening result, not confirmed "
         "skill. A genuine NIFTY 500 TRI series is the most valuable missing input.")


def explorer_page(df: pd.DataFrame) -> None:
    hero("Fund explorer", "One fund at a time: its class, the tests it passed and failed, and any caveat that applies")
    market = st.selectbox("Market", list(cd.MARKETS), key="fx_market")
    choices = df[df["market"] == market].copy()
    choices["option"] = choices["symbol"] + " · " + choices["fund_name"]
    choices = choices.sort_values("option")
    if choices.empty:
        st.warning("No funds are available in this market.")
        return
    option = st.selectbox("Fund", choices["option"].tolist(), key="fx_fund")
    row = choices.loc[choices["option"] == option].iloc[0]
    st.subheader(str(row["fund_name"]))
    meta = [f"{market} · {row['symbol']}"]
    if isinstance(row.get("amc"), str) and row["amc"].strip():
        meta.append(row["amc"])
    if isinstance(row.get("lead_manager"), str) and row["lead_manager"].strip():
        meta.append(f"named manager: {row['lead_manager']} (since {row.get('tenure_start', 'n/a')})")
    st.caption(" · ".join(meta))
    fund_detail(row)


def methodology_page(df: pd.DataFrame, issues: list[str]) -> None:
    hero("Methodology and data quality", "How a fund gets its class, where the data come from, and what the results cannot tell you")
    rules_banner(df)
    st.subheader("How a fund is classified")
    st.markdown(
        "The rules are applied top to bottom; the first one that matches decides the class. Nothing is tuned to a target distribution: "
        "the same thresholds are applied to every fund in both markets.\n\n"
        "1. **Likely Style/Luck Driven**: a positive CAPM alpha with p < 0.10 that falls by at least 75% after the factor adjustment, with no significant four-factor alpha.\n"
        "2. **Strong Evidence of Skill**: significant CAPM alpha, significant bootstrap, significant four-factor alpha, positive in at least 3 of 4 subperiods, **and** Benjamini-Hochberg q < 0.10 for all three tests.\n"
        "3. **Strong Skill Candidate (Preliminary)** (shown as “Strong” in this dashboard): CAPM p < 0.10, significant four-factor alpha, positive in at least 3 of 4 subperiods, bootstrap p < 0.10.\n"
        "4. **Moderate Evidence**: a significant CAPM alpha, or a surviving four-factor alpha, or persistence with CAPM p < 0.10, or a significant bootstrap.\n"
        "5. **Positive but Inconclusive**: both alphas positive and positive in at least 3 of 4 subperiods, nothing significant.\n"
        "6. **Weak/Inconclusive**: everything else.")
    order = shown_classes(df["category"] if not df.empty else None)
    scale_table(df, order)
    class_footnote(order)

    st.subheader("Data sources and versions")
    for market, cfg in cd.MARKETS.items():
        sub = df[df["market"] == market]
        with st.expander(f"{market} project"):
            st.write(f"**Funds:** {len(sub)} · **Rule version:** {', '.join(sorted(set(sub['version']))) if not sub.empty else 'n/a'}")
            st.write(f"**Folder:** `markets/{cfg['key']}/` · **Results read:** `markets/{cfg['key']}/data/processed/final_evaluation.csv`")
            st.write(f"**Market portfolio:** {cfg['market_note']}")
            st.write(f"**Factors:** {cfg['factor_note']}")
            st.write(f"**Universe:** {cfg['universe_note']}")

    st.subheader("Integrity checks run on load")
    if not issues:
        st.success("All checks passed: both result files loaded, every class label is recognised, no duplicate funds, "
                   "and both markets share one rule version.")
    for issue in issues:
        st.warning(issue)
    st.caption("These checks cover file structure and consistency. The numbers themselves are re-verified by the test suite "
               "(including an independent statsmodels cross-check), not on every page load.")

    st.subheader("What the results cannot tell you")
    st.markdown(
        "- A class is a statement about statistical evidence. It is not proof of skill, nor proof of its absence, and it is not investment advice.\n"
        "- Twelve hand-picked, well-known funds per market over about five years (2019-2023) cannot represent all funds; survivorship and selection bias are likely.\n"
        "- A single fund's alpha is measured imprecisely (see the Overview), so only a very large true alpha would be detected.\n"
        "- Benchmark choice matters (see Benchmark robustness). The Indian primary market is a total-return portfolio from IIMA; a genuine NIFTY 500 TRI series has not been added.\n"
        "- Manager names and tenure dates are curated, not machine-verified. Where a named manager ran a scheme for under 90% of the window, the result describes the scheme, not the person.\n"
        "- The four-factor model leaves out quality, liquidity, sector tilts and, for Parag Parikh Flexi Cap, overseas equity and large cash holdings.\n"
        "- Multiple testing: with 12 funds per market some funds would look significant by chance, which is why the strict class needs q < 0.10.")


# ----------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------

PAGES = ["Overview", "India", "USA", "Cross-market comparison", "Benchmark robustness", "Fund explorer", "Methodology & data"]


def main() -> None:
    st.set_page_config(page_title="Fund Manager Skill Detector", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    style_page()
    fingerprint = cd.output_fingerprint()
    df, _, issues = load_all(fingerprint)

    with st.sidebar:
        st.markdown("## 📈 Fund Skill Detector")
        st.caption("India + United States · one rule set")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="nav")
        st.divider()
        st.markdown("**Evidence classes**")
        st.markdown("<br>".join(
            f'<span style="color:{COLORS[c]}">●</span> {html.escape(ui(c))}'
            for c in shown_classes(df["category"] if not df.empty else None)), unsafe_allow_html=True)
        st.divider()
        if not df.empty:
            version = cd.rules_version(df)
            st.caption(f"{len(df)} funds loaded · rules " + (f"v{version}" if version else "MISMATCH"))
        for issue in issues:
            st.warning(issue)
        st.caption("Classes are evidence summaries, not investment advice.")

    if page == "Overview":
        overview(df)
    elif page in cd.MARKETS:
        market_page(page, df)
    elif page == "Cross-market comparison":
        comparison_page(df)
    elif page == "Benchmark robustness":
        robustness_page(load_sensitivity(fingerprint), df)
    elif page == "Fund explorer":
        explorer_page(df)
    else:
        methodology_page(df, issues)

    st.divider()
    st.caption("Fund Manager Skill Detector · reads the saved market results; nothing is re-estimated here.")


if __name__ == "__main__":
    main()
