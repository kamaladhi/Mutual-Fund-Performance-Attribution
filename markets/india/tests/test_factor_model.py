"""
Unit Test Suite for Carhart 4-Factor Analysis Engine (src/factor_model.py)
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
import pytest

from src.factor_model import (
    fit_four_factor_ols,
    run_four_factor_analysis,
    FactorModelResult
)


@pytest.fixture
def synthetic_4factor_data():
    """Generate synthetic 4-factor return series with known true parameters."""
    np.random.seed(42)
    n = 2000
    mkt = np.random.normal(0, 0.01, n)
    smb = np.random.normal(0, 0.005, n)
    hml = np.random.normal(0, 0.005, n)
    mom = np.random.normal(0, 0.006, n)
    
    # Demean factors for exact independence
    mkt -= np.mean(mkt)
    smb -= np.mean(smb)
    hml -= np.mean(hml)
    mom -= np.mean(mom)

    true_alpha = 0.0003
    true_b_mkt = 1.05
    true_b_smb = 0.25
    true_b_hml = -0.30
    true_b_mom = 0.15
    noise = np.random.normal(0, 0.001, n)
    noise -= np.mean(noise)

    y = true_alpha + true_b_mkt * mkt + true_b_smb * smb + true_b_hml * hml + true_b_mom * mom + noise

    return y, mkt, smb, hml, mom, true_alpha, true_b_mkt, true_b_smb, true_b_hml, true_b_mom


def test_synthetic_coefficient_recovery(synthetic_4factor_data):
    """Test recovery of known alpha, betas, standard errors, t-stats, p-vals, and CIs."""
    y, mkt, smb, hml, mom, t_alpha, t_mkt, t_smb, t_hml, t_mom = synthetic_4factor_data
    res, residuals = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="SYNTH_FUND", capm_alpha_daily=0.0005)

    assert isinstance(res, FactorModelResult)
    assert res.n_observations == 2000

    # Coefficient recovery
    np.testing.assert_allclose(res.alpha_daily, t_alpha, atol=1e-4)
    np.testing.assert_allclose(res.beta_market, t_mkt, atol=1e-2)
    np.testing.assert_allclose(res.beta_smb, t_smb, atol=1e-2)
    np.testing.assert_allclose(res.beta_hml, t_hml, atol=1e-2)
    np.testing.assert_allclose(res.beta_momentum, t_mom, atol=1e-2)

    # Standard errors, t-stats, p-values, R-squared
    assert res.alpha_se > 0
    assert res.beta_market_se > 0
    assert abs(res.alpha_tstat) > 2.0
    assert res.alpha_pvalue < 0.05
    assert res.r_squared > 0.90
    assert res.adjusted_r_squared <= res.r_squared


def test_confidence_intervals_and_annualization(synthetic_4factor_data):
    """Test 95% confidence interval ordering and compounding annualization."""
    y, mkt, smb, hml, mom, t_alpha, t_mkt, _, _, _ = synthetic_4factor_data
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="SYNTH_FUND")

    # CI ordering
    assert res.alpha_ci_lower <= res.alpha_daily <= res.alpha_ci_upper
    assert res.beta_market_ci_lower <= res.beta_market <= res.beta_market_ci_upper
    assert res.beta_smb_ci_lower <= res.beta_smb <= res.beta_smb_ci_upper
    assert res.beta_hml_ci_lower <= res.beta_hml <= res.beta_hml_ci_upper
    assert res.beta_momentum_ci_lower <= res.beta_momentum <= res.beta_momentum_ci_upper

    # Annualization compounding formula test: ((1 + alpha)^252 - 1) * 100
    expected_annual = ((1.0 + res.alpha_daily) ** 252 - 1.0) * 100.0
    np.testing.assert_almost_equal(res.alpha_annual_pct, expected_annual)


def test_capm_vs_factor_alpha_comparison(synthetic_4factor_data):
    """Test CAPM vs four-factor alpha comparison and alpha reduction calculation."""
    y, mkt, smb, hml, mom, _, _, _, _, _ = synthetic_4factor_data
    capm_alpha = 0.0008  # Higher CAPM alpha
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="SYNTH_FUND", capm_alpha_daily=capm_alpha)

    assert res.capm_alpha_daily == capm_alpha
    expected_reduction = capm_alpha - res.alpha_daily
    expected_reduction_pct = (expected_reduction / capm_alpha) * 100.0

    np.testing.assert_almost_equal(res.alpha_reduction, expected_reduction)
    np.testing.assert_almost_equal(res.alpha_reduction_pct, expected_reduction_pct)
    assert res.factor_alpha_lower_than_capm is True


def test_zero_capm_alpha_handling(synthetic_4factor_data):
    """Test safe handling when CAPM alpha is zero (prevents zero division error)."""
    y, mkt, smb, hml, mom, _, _, _, _, _ = synthetic_4factor_data
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="ZERO_FUND", capm_alpha_daily=0.0)

    assert res.capm_alpha_daily == 0.0
    assert np.isnan(res.alpha_reduction_pct)


def test_missing_data_and_invalid_parameters():
    """Test missing data mask and invalid parameter error handling."""
    y = np.array([0.01, np.nan, 0.02, 0.01, 0.03, -0.01])
    mkt = np.array([0.005, 0.01, np.nan, 0.002, 0.015, -0.005])
    smb = np.array([0.001, 0.002, 0.003, 0.004, 0.001, 0.002])
    hml = np.array([-0.001, -0.002, 0.001, 0.002, 0.001, 0.002])
    mom = np.array([0.002, 0.001, -0.001, 0.002, 0.001, 0.003])

    # Insufficient valid rows (n <= 5)
    with pytest.raises(ValueError, match="Insufficient observations"):
        fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="SHORT_FUND")


def test_multi_fund_analysis_and_schema_validation():
    """Test multi-fund execution and schema completeness."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "Date": np.tile(dates, 2),
        "symbol": ["F1"] * n + ["F2"] * n,
        "excess_fund_return": np.random.normal(0.0002, 0.01, 2 * n),
        "Mkt-RF": np.random.normal(0.0004, 0.01, 2 * n),
        "SMB": np.random.normal(0, 0.004, 2 * n),
        "HML": np.random.normal(0, 0.004, 2 * n),
        "Mom": np.random.normal(0, 0.005, 2 * n)
    })

    summary_df, results_dict, resids_dict = run_four_factor_analysis(df)

    assert len(summary_df) == 2
    assert "F1" in results_dict
    assert "F2" in results_dict
    assert len(resids_dict["F1"]) == n
    assert "beta_market" in summary_df.columns
    assert "alpha_reduction_pct" in summary_df.columns


def test_source_data_immutability():
    """Test that source dataframes are not mutated in place."""
    df_orig = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09", "2023-01-10"],
        "symbol": ["F1"] * 6,
        "excess_fund_return": [0.01, 0.02, -0.01, 0.015, -0.005, 0.02],
        "Mkt-RF": [0.005, 0.01, -0.008, 0.012, -0.003, 0.015],
        "SMB": [0.001, -0.002, 0.003, -0.001, 0.002, -0.001],
        "HML": [-0.002, 0.001, -0.001, 0.003, -0.002, 0.001],
        "Mom": [0.003, -0.001, 0.002, 0.001, -0.002, 0.004]
    })
    copy_df = df_orig.copy(deep=True)

    run_four_factor_analysis(df_orig)

    pd.testing.assert_frame_equal(df_orig, copy_df)
