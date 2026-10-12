"""
fee_calculator.py — Phase 2: Fee Leakage & Wealth Erosion Calculator
======================================================================
Translates abstract p-values and alpha estimates from the econometric engine
into actionable rupee/dollar projections the investor can immediately understand.

The key insight: even a fund with positive (but insignificant) alpha often costs
the investor more in active fees than it returns in net alpha.

Usage:
    from components.fee_calculator import compute_wealth_paths, plot_wealth_paths, fee_summary_card
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ─── Core computation ──────────────────────────────────────────────────────────

def compute_wealth_paths(
    invested: float,
    years: int,
    fund_net_alpha_pct: float,  # 4-Factor net alpha in % per year (can be negative)
    fund_ter_pct: float,         # Active fund TER in % per year, e.g. 1.80
    index_ter_pct: float = 0.10, # Index fund TER (Nifty 500 / S&P 500 ETF), e.g. 0.10
    gross_market_return_pct: float = 12.0,  # Assumed gross market return % per year
    currency_symbol: str = "₹",
) -> dict:
    """
    Projects investor wealth under two strategies over `years`:
      - Active Fund:  gross_market + fund_net_alpha - fund_ter
      - Index Fund:   gross_market - index_ter

    All inputs in % per year (e.g. 1.80 not 0.018).
    Returns a dict of arrays + summary scalars for charting and display.
    """
    yrs           = np.arange(0, years + 1)
    active_net    = (gross_market_return_pct + fund_net_alpha_pct - fund_ter_pct) / 100.0
    index_net     = (gross_market_return_pct - index_ter_pct) / 100.0

    active_wealth = invested * ((1 + active_net) ** yrs)
    index_wealth  = invested * ((1 + index_net)  ** yrs)
    leakage       = index_wealth - active_wealth   # positive = index wins

    return {
        "years":             yrs,
        "active_wealth":     active_wealth,
        "index_wealth":      index_wealth,
        "leakage":           leakage,
        "total_leakage":     float(leakage[-1]),
        "annual_leakage":    float(leakage[-1] / years),
        "active_cagr_pct":   float(active_net * 100),
        "index_cagr_pct":    float(index_net  * 100),
        "final_active":      float(active_wealth[-1]),
        "final_index":       float(index_wealth[-1]),
        "currency":          currency_symbol,
        "invested":          invested,
        "years_n":           years,
    }


# ─── Plotly chart ──────────────────────────────────────────────────────────────

def plot_wealth_paths(result: dict, fund_name: str = "Active Fund") -> go.Figure:
    """
    Animated dual-line wealth trajectory chart.
    Green area = additional wealth from index switching.
    """
    y_active = result["active_wealth"]
    y_index  = result["index_wealth"]
    xr       = result["years"].tolist()
    cur      = result["currency"]

    # Determine which series is on top
    index_wins = result["total_leakage"] > 0

    fig = go.Figure()

    # Bottom series first (fill goes from bottom to top)
    if index_wins:
        fig.add_trace(go.Scatter(
            x=xr, y=y_active.tolist(),
            name=f"{fund_name} (Active)",
            line=dict(color="#ef4444", width=2.5),
            fill=None,
            hovertemplate=f"Year %{{x}}<br>{cur}%{{y:,.0f}}<extra>Active Fund</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=xr, y=y_index.tolist(),
            name="Index Fund Equivalent",
            line=dict(color="#10b981", width=2.5),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.12)",
            hovertemplate=f"Year %{{x}}<br>{cur}%{{y:,.0f}}<extra>Index Fund</extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=xr, y=y_index.tolist(),
            name="Index Fund Equivalent",
            line=dict(color="#10b981", width=2.5),
            fill=None,
            hovertemplate=f"Year %{{x}}<br>{cur}%{{y:,.0f}}<extra>Index Fund</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=xr, y=y_active.tolist(),
            name=f"{fund_name} (Active)",
            line=dict(color="#3b82f6", width=2.5),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.12)",
            hovertemplate=f"Year %{{x}}<br>{cur}%{{y:,.0f}}<extra>Active Fund</extra>",
        ))

    leakage_label = (
        f"Index wins by {cur}{abs(result['total_leakage']):,.0f} over {result['years_n']} years"
        if index_wins else
        f"Active fund wins by {cur}{abs(result['total_leakage']):,.0f} over {result['years_n']} years"
    )

    fig.update_layout(
        title=dict(text=f"Projected Wealth Trajectory — {leakage_label}", font=dict(size=14)),
        xaxis_title="Year",
        yaxis_title=f"Portfolio Value ({cur})",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                     tickformat=f",.0f")
    return fig


def plot_annual_fee_drag(result: dict) -> go.Figure:
    """
    Bar chart showing annual wealth leakage each year.
    Useful to show that fee drag compounds over time.
    """
    annual_drag = np.diff(result["leakage"])  # change in leakage per year
    years_range = result["years"][1:].tolist()
    cur         = result["currency"]

    colors = ["#10b981" if v >= 0 else "#3b82f6" for v in annual_drag]

    fig = go.Figure(go.Bar(
        x=years_range,
        y=annual_drag.tolist(),
        marker_color=colors,
        text=[f"{cur}{abs(v):,.0f}" for v in annual_drag],
        textposition="outside",
        hovertemplate="Year %{x}<br>Annual drag: " + cur + "%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Annual Fee Drag: Cost of Active Management Each Year",
        xaxis_title="Year",
        yaxis_title=f"Opportunity Cost ({cur})",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=30),
        showlegend=False,
    )
    return fig


# ─── Summary card data ─────────────────────────────────────────────────────────

def fee_summary_metrics(result: dict) -> list[dict]:
    """
    Returns a list of metric dicts for st.metric() display.
    """
    cur = result["currency"]
    invested = result["invested"]
    total_leakage = result["total_leakage"]
    annual_leakage = result["annual_leakage"]
    index_wins = total_leakage > 0

    return [
        {
            "label": "Your Investment",
            "value": f"{cur}{invested:,.0f}",
            "delta": None,
        },
        {
            "label": f"Active Fund CAGR",
            "value": f"{result['active_cagr_pct']:+.2f}% p.a.",
            "delta": None,
        },
        {
            "label": f"Index Fund CAGR",
            "value": f"{result['index_cagr_pct']:+.2f}% p.a.",
            "delta": None,
        },
        {
            "label": f"{result['years_n']}-Year Outcome",
            "value": f"{'Index wins' if index_wins else 'Active wins'}",
            "delta": f"{cur}{abs(total_leakage):,.0f} {'saved' if index_wins else 'gained'} by switching",
            "delta_color": "normal" if not index_wins else "inverse",
        },
        {
            "label": "Annual Fee Drag",
            "value": f"{cur}{abs(annual_leakage):,.0f}/year",
            "delta": f"{'Wasted on active fees' if index_wins else 'Justified by alpha'}",
            "delta_color": "inverse" if index_wins else "normal",
        },
    ]
