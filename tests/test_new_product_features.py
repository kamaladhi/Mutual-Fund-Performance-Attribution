"""
test_new_product_features.py
Tests the new quantitative and product components:
  1. Fee Leakage Calculator
  2. Rolling CAPM Analysis
  3. Institutional PDF Report Data Builder
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from components.fee_calculator import compute_wealth_paths
from components.rolling_analysis import compute_rolling_capm, rolling_summary_stats
from components.report_generator import build_report_data, generate_pdf_bytes


def test_wealth_paths_simulation():
    # 20 years, 1.5% net alpha, 1.8% TER vs 0.10% passive ETF
    res = compute_wealth_paths(
        invested=100000.0,
        years=20,
        fund_net_alpha_pct=1.5,
        fund_ter_pct=1.8,
        index_ter_pct=0.10,
        gross_market_return_pct=12.0,
        currency_symbol="₹",
    )
    assert isinstance(res, dict)
    assert len(res["years"]) == 21  # Year 0 to Year 20
    assert "active_wealth" in res
    assert "index_wealth" in res
    assert "leakage" in res
    assert "total_leakage" in res
    assert res["final_active"] > 0
    assert res["final_index"] > 0


def test_rolling_capm_synthetic():
    np.random.seed(42)
    n = 350
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt = np.random.normal(0.0004, 0.01, size=n)
    rf = np.full(n, 0.0001)
    ret = rf + 0.0002 + 1.1 * (mkt - rf) + np.random.normal(0, 0.005, size=n)

    aligned = pd.DataFrame({
        "Date": dates,
        "excess_fund_return": ret - rf,
        "excess_market_return": mkt - rf,
    })

    rolling_df = compute_rolling_capm(aligned, window=252)
    assert not rolling_df.empty
    assert "rolling_alpha_annual_pct" in rolling_df.columns
    assert "rolling_beta" in rolling_df.columns

    stats = rolling_summary_stats(rolling_df)
    assert "alpha_mean" in stats
    assert "beta_mean" in stats
    assert abs(stats["beta_mean"] - 1.1) < 0.3


def test_report_builder_and_pdf():
    pipeline_result = {
        "capm": {
            "alpha_annual_pct": 3.45,
            "alpha_pvalue_hac": 0.032,
            "beta": 1.05,
            "r_squared": 0.88,
            "n_obs": 1200,
        },
        "bootstrap": {
            "bootstrap_pvalue": 0.041,
        },
        "factor": {
            "alpha_daily": 0.0001,
            "betas": {"Mkt-RF": 1.02, "SMB": 0.25, "HML": -0.15, "WML": 0.10},
            "tstats": {"Mkt-RF": 18.2, "SMB": 3.1, "HML": -2.2, "WML": 1.8},
            "pvalues": {"Mkt-RF": 0.0001, "SMB": 0.002, "HML": 0.028, "WML": 0.072},
        },
        "persistence": {
            "positive_periods": 3,
            "total_periods": 4,
        },
        "evaluation": {
            "evidence_class": "Moderate Evidence",
            "score": 62,
            "explanation": "Positive alpha across 3 periods with bootstrap significance.",
            "factor_alpha_survives": True,
        },
    }

    report_data = build_report_data("Test Growth Fund", "India", pipeline_result)
    assert report_data["fund_name"] == "Test Growth Fund"
    assert report_data["evidence_class"] == "Moderate Evidence"

    # PDF or HTML generation test
    content_bytes = generate_pdf_bytes(report_data)
    assert len(content_bytes) > 0
    assert isinstance(content_bytes, bytes)
