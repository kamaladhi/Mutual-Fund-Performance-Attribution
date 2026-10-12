"""
rolling_analysis.py — Phase 3: Rolling Alpha & Beta Time-Series
================================================================
Computes 1-year (252-day) rolling CAPM alpha and beta over the aligned dataset,
revealing how manager skill varies across market regimes.

Key insight: A static 5-year alpha hides the story. Did the manager outperform
during COVID? During the rate-hike cycle? Or only in calm markets?

Usage:
    from components.rolling_analysis import compute_rolling_capm, plot_rolling_alpha, plot_rolling_beta
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Market regime annotations ────────────────────────────────────────────────
MARKET_REGIMES = [
    {
        "label":    "COVID Crash",
        "start":    "2020-02-15",
        "end":      "2020-04-30",
        "color":    "#ef4444",
        "opacity":  0.08,
    },
    {
        "label":    "COVID Recovery",
        "start":    "2020-05-01",
        "end":      "2021-10-31",
        "color":    "#10b981",
        "opacity":  0.06,
    },
    {
        "label":    "Rate Hike Cycle",
        "start":    "2022-01-01",
        "end":      "2023-06-30",
        "color":    "#f59e0b",
        "opacity":  0.08,
    },
]


# ─── Core rolling computation ──────────────────────────────────────────────────

def compute_rolling_capm(aligned_df: pd.DataFrame,
                          window: int = 252) -> pd.DataFrame:
    """
    Computes rolling OLS CAPM alpha and beta over a sliding `window`-day window.

    Uses numpy OLS (no HAC) for speed — this is for visualisation only,
    the definitive inference remains the full-sample Newey-West result.

    Args:
        aligned_df: The aligned dataset from the econometric pipeline.
                    Must contain: Date, excess_fund_return, excess_market_return.
        window:     Sliding window in trading days (default: 252 = 1 year).

    Returns:
        DataFrame with columns: Date, rolling_alpha_annual_pct, rolling_beta, n_obs
    """
    required = {"Date", "excess_fund_return", "excess_market_return"}
    if not required.issubset(aligned_df.columns):
        return pd.DataFrame()

    df = aligned_df[["Date", "excess_fund_return", "excess_market_return"]].dropna().copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < window + 10:
        return pd.DataFrame()

    results = []
    excess_ret = df["excess_fund_return"].values
    market_ret = df["excess_market_return"].values
    dates      = df["Date"].values

    for i in range(window, len(df)):
        y = excess_ret[i - window:i]
        x = market_ret[i - window:i]

        # OLS via numpy: [alpha_daily, beta]
        X_mat = np.column_stack([np.ones(window), x])
        try:
            coeffs, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
            alpha_daily = float(coeffs[0])
            beta        = float(coeffs[1])
        except Exception:
            alpha_daily = np.nan
            beta        = np.nan

        # Annualise alpha: (1 + alpha_daily)^252 - 1
        if not np.isnan(alpha_daily):
            alpha_annual_pct = (((1 + alpha_daily) ** 252) - 1) * 100
        else:
            alpha_annual_pct = np.nan

        results.append({
            "Date":                    dates[i],
            "rolling_alpha_annual_pct": alpha_annual_pct,
            "rolling_beta":             beta,
        })

    return pd.DataFrame(results)


# ─── Rolling Alpha chart ────────────────────────────────────────────────────────

def _add_regime_bands(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Adds COVID and rate-hike regime shading to a Plotly figure."""
    for regime in MARKET_REGIMES:
        fig.add_vrect(
            x0=regime["start"],
            x1=regime["end"],
            fillcolor=regime["color"],
            opacity=regime["opacity"],
            layer="below",
            line_width=0,
            annotation_text=regime["label"],
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=regime["color"],
            row=row, col=col,
        )


def plot_rolling_alpha(rolling_df: pd.DataFrame,
                        fund_name: str = "Fund",
                        window: int = 252) -> go.Figure:
    """
    Line chart of 1-year rolling annualised CAPM alpha with market regime shading.

    Time above the zero line = periods when the manager added value relative to
    the CAPM-implied return.
    """
    if rolling_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Insufficient data for rolling analysis")
        return fig

    dates  = pd.to_datetime(rolling_df["Date"])
    alphas = rolling_df["rolling_alpha_annual_pct"]

    # Colour the line green/red based on sign
    pos_alpha = alphas.clip(lower=0)
    neg_alpha = alphas.clip(upper=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=alphas.tolist(),
        mode="lines",
        name=f"{window}-day Rolling Alpha",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="Date: %{x|%b %Y}<br>Rolling Alpha: %{y:+.2f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.6)", line_width=1.5)

    # Regime bands
    _add_regime_bands(fig)

    fig.update_layout(
        title=dict(
            text=f"1-Year Rolling CAPM Alpha: {fund_name} (% per annum)",
            font=dict(size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Rolling Alpha (% p.a.)",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                     zeroline=True, zerolinecolor="rgba(100,116,139,0.6)")
    return fig


def plot_rolling_beta(rolling_df: pd.DataFrame,
                       fund_name: str = "Fund",
                       window: int = 252) -> go.Figure:
    """
    Line chart of 1-year rolling market beta.

    Beta > 1 = fund takes more market risk than benchmark.
    Beta < 1 = fund is defensive; manager may be timing the market.
    Beta rising into downturns = poor timing.
    """
    if rolling_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Insufficient data")
        return fig

    dates = pd.to_datetime(rolling_df["Date"])
    betas = rolling_df["rolling_beta"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=betas.tolist(),
        mode="lines",
        name=f"{window}-day Rolling Beta",
        line=dict(color="#8b5cf6", width=2),
        hovertemplate="Date: %{x|%b %Y}<br>Rolling Beta: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(100,116,139,0.6)", line_width=1.5,
                  annotation_text="β = 1.0 (market)", annotation_position="bottom right")

    _add_regime_bands(fig)

    fig.update_layout(
        title=dict(
            text=f"1-Year Rolling Market Beta: {fund_name}",
            font=dict(size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Rolling Beta (β)",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)")
    return fig


def plot_alpha_beta_combined(rolling_df: pd.DataFrame,
                              fund_name: str = "Fund") -> go.Figure:
    """
    Two-panel subplot: Rolling Alpha (top) + Rolling Beta (bottom).
    Useful for seeing market-timing behaviour: does beta drop when alpha drops?
    """
    if rolling_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Insufficient data")
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("1-Year Rolling Alpha (% p.a.)", "1-Year Rolling Beta"),
        vertical_spacing=0.08,
    )

    dates  = pd.to_datetime(rolling_df["Date"])
    alphas = rolling_df["rolling_alpha_annual_pct"].tolist()
    betas  = rolling_df["rolling_beta"].tolist()

    fig.add_trace(go.Scatter(
        x=dates, y=alphas,
        name="Rolling Alpha",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="%{x|%b %Y}: %{y:+.2f}%<extra>Alpha</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=betas,
        name="Rolling Beta",
        line=dict(color="#8b5cf6", width=2),
        hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>Beta</extra>",
    ), row=2, col=1)

    fig.add_hline(y=0,   line_dash="dash", line_color="rgba(100,116,139,0.5)", row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(100,116,139,0.5)", row=2, col=1)

    for regime in MARKET_REGIMES:
        for row in [1, 2]:
            fig.add_vrect(
                x0=regime["start"], x1=regime["end"],
                fillcolor=regime["color"], opacity=regime["opacity"],
                layer="below", line_width=0,
                row=row, col=1,
            )

    fig.update_layout(
        title=dict(text=f"Rolling CAPM Diagnostics: {fund_name}", font=dict(size=14)),
        height=520,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=70, b=30),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.10)")
    return fig


def rolling_summary_stats(rolling_df: pd.DataFrame) -> dict:
    """Returns key summary statistics from the rolling series."""
    if rolling_df.empty:
        return {}
    alphas = rolling_df["rolling_alpha_annual_pct"].dropna()
    betas  = rolling_df["rolling_beta"].dropna()
    return {
        "alpha_mean":    float(alphas.mean()),
        "alpha_median":  float(alphas.median()),
        "alpha_positive_pct": float((alphas > 0).mean() * 100),
        "alpha_min":     float(alphas.min()),
        "alpha_max":     float(alphas.max()),
        "beta_mean":     float(betas.mean()),
        "beta_min":      float(betas.min()),
        "beta_max":      float(betas.max()),
        "n_periods":     len(alphas),
    }
