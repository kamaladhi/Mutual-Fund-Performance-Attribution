"""
shap_attribution.py — Phase 3: Explainable AI Factor Attribution
=================================================================
Augments classical linear OLS factor analysis with XGBoost + TreeSHAP,
replacing static OLS betas with dynamic Shapley value decompositions.

Produces:
  1. Factor Radar Chart  — which factor dominated this fund's return profile?
  2. Return Waterfall    — decomposes annualised return into factor contributions + residual alpha
  3. SHAP summary stats  — for dashboard metric display

Usage:
    from components.shap_attribution import run_shap_analysis, plot_factor_radar, plot_return_waterfall
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import xgboost as xgb
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─── Factor Labels ─────────────────────────────────────────────────────────────
FACTOR_COLS    = ["excess_market_return", "SMB", "HML", "Mom"]
FACTOR_LABELS  = {
    "excess_market_return": "Market (Mkt-RF)",
    "SMB":                  "Size (SMB)",
    "HML":                  "Value (HML)",
    "Mom":                  "Momentum",
}
FACTOR_COLORS = {
    "Market (Mkt-RF)": "#3b82f6",
    "Size (SMB)":      "#8b5cf6",
    "Value (HML)":     "#f59e0b",
    "Momentum":        "#10b981",
}


# ─── Core SHAP pipeline ────────────────────────────────────────────────────────

def run_shap_analysis(aligned_df: pd.DataFrame) -> dict:
    """
    Trains XGBoost on the 4 Carhart factors and computes SHAP values.

    Args:
        aligned_df: The aligned dataset from the econometric pipeline.
                    Must contain columns: excess_fund_return + all FACTOR_COLS.

    Returns:
        dict with:
          - shap_df:       DataFrame of average SHAP contributions per factor
          - shap_values:   raw SHAP matrix (n_obs × 4)
          - model:         trained XGBoost model
          - X:             feature DataFrame used for training
          - r2:            XGBoost R²
          - available:     bool (False if xgboost/shap not installed)
    """
    if not SHAP_AVAILABLE:
        return {"available": False, "error": "xgboost and shap are not installed."}

    available_factors = [c for c in FACTOR_COLS if c in aligned_df.columns]
    if len(available_factors) < 2:
        return {"available": False, "error": "Not enough factor columns in aligned dataset."}

    X = aligned_df[available_factors].fillna(0)
    y = aligned_df["excess_fund_return"].fillna(0)

    if len(X) < 60:
        return {"available": False, "error": "Need at least 60 observations for SHAP analysis."}

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # R² score
    y_pred = model.predict(X)
    ss_res = float(np.sum((y.values - y_pred) ** 2))
    ss_tot = float(np.sum((y.values - y.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)   # shape: (n_obs, n_factors)

    # Build contribution summary
    labels = [FACTOR_LABELS.get(c, c) for c in available_factors]
    mean_abs   = np.abs(shap_values).mean(axis=0)
    mean_dir   = shap_values.mean(axis=0)

    shap_df = pd.DataFrame({
        "col":         available_factors,
        "label":       labels,
        "shap_abs":    mean_abs,
        "shap_signed": mean_dir,
        "direction":   np.sign(mean_dir),
    })
    shap_df = shap_df.sort_values("shap_abs", ascending=False).reset_index(drop=True)

    return {
        "available":   True,
        "shap_df":     shap_df,
        "shap_values": shap_values,
        "model":       model,
        "X":           X,
        "r2":          r2,
        "n_obs":       len(X),
    }


# ─── Radar Chart ───────────────────────────────────────────────────────────────

def plot_factor_radar(shap_result: dict, fund_name: str = "Fund") -> go.Figure:
    """
    Spider / Radar chart showing the average absolute SHAP contribution of each factor.
    Larger area = that factor dominates this fund's return.
    """
    if not shap_result.get("available"):
        fig = go.Figure()
        fig.update_layout(title="SHAP not available")
        return fig

    df = shap_result["shap_df"]
    labels  = df["label"].tolist()
    values  = df["shap_abs"].tolist()
    dirs    = df["direction"].tolist()

    # Close the loop for radar
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    # Direction-aware color: green if positive contribution, red if negative
    fill_color = "rgba(59, 130, 246, 0.25)"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color="#3b82f6", width=2.5),
        name=fund_name,
        hovertemplate="%{theta}<br>SHAP contribution: %{r:.5f}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.35],
                tickfont=dict(size=9),
                gridcolor="rgba(128,128,128,0.20)",
            ),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=f"AI Factor Attribution Radar — {fund_name}",
            font=dict(size=14),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


# ─── Return Waterfall ──────────────────────────────────────────────────────────

def plot_return_waterfall(shap_result: dict,
                           capm_alpha_annual_pct: float,
                           fund_name: str = "Fund") -> go.Figure:
    """
    Waterfall chart decomposing annualised return into:
      Factor bets (SHAP signed × 252 × 100) + Residual Manager Alpha

    Bars going up = positive contribution (green)
    Bars going down = negative contribution (red)
    """
    if not shap_result.get("available"):
        fig = go.Figure()
        fig.update_layout(title="SHAP not available")
        return fig

    df = shap_result["shap_df"]
    labels      = df["label"].tolist()
    shap_signed = df["shap_signed"].tolist()
    # Annualise: daily SHAP contribution × 252 trading days × 100 (to %)
    contribs_ann = [s * 252 * 100 for s in shap_signed]

    all_labels = labels + ["Manager Alpha (Residual)"]
    all_values = contribs_ann + [capm_alpha_annual_pct]
    measures   = ["relative"] * len(labels) + ["total"]

    text_labels = [f"{v:+.2f}%" for v in all_values]

    fig = go.Figure(go.Waterfall(
        name="Return Attribution",
        orientation="v",
        measure=measures,
        x=all_labels,
        y=all_values,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11),
        connector=dict(line=dict(color="rgba(128,128,128,0.3)", dash="dot")),
        increasing=dict(marker=dict(color="#10b981")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#3b82f6", line=dict(color="#1d4ed8", width=1.5))),
        hovertemplate="%{x}<br>Contribution: %{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Annual Return Attribution Waterfall — {fund_name}",
            font=dict(size=14),
        ),
        yaxis_title="Annual Return Contribution (%)",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=40),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)", zeroline=True,
                     zerolinecolor="rgba(128,128,128,0.4)", zerolinewidth=1.5)
    return fig


# ─── SHAP summary table ────────────────────────────────────────────────────────

def shap_summary_table(shap_result: dict) -> pd.DataFrame:
    """
    Returns a clean DataFrame for st.dataframe() display.
    """
    if not shap_result.get("available"):
        return pd.DataFrame()

    df = shap_result["shap_df"].copy()
    df["Daily SHAP (×10⁻⁴)"] = (df["shap_abs"] * 10000).round(3)
    df["Direction"]           = df["direction"].map({1: "↑ Positive", -1: "↓ Negative", 0: "Neutral"})
    df["Annual Impact (%)"]   = (df["shap_signed"] * 252 * 100).round(2).astype(str) + "%"
    return df[["label", "Daily SHAP (×10⁻⁴)", "Direction", "Annual Impact (%)"]].rename(
        columns={"label": "Factor"}
    )
