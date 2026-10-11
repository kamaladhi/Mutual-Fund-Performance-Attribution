"""
Unit Test Suite for CAPM Analysis Engine (src/capm.py)
"""

import numpy as np
import pandas as pd
import pytest

from src.capm import (
    fit_capm_ols,
    run_capm_analysis,
    CAPMResult,
    compute_durbin_watson
)


@pytest.fixture
def synthetic_capm_data():
    """Generate synthetic return series with exact known OLS relationship y = 0.0005 + 1.2 * x + noise."""
    np.random.seed(123)
    n = 2000
    market_excess = np.random.normal(loc=0.0, scale=0.01, size=n)
    # Demean market excess so mean is exactly 0
    market_excess = market_excess - np.mean(market_excess)
    
    true_alpha = 0.0005
    true_beta = 1.2
    noise = np.random.normal(loc=0.0, scale=0.001, size=n)
    # Demean noise so mean is exactly 0 and orthogonal to x
    noise = noise - np.mean(noise)
    
    fund_excess = true_alpha + true_beta * market_excess + noise
    return fund_excess, market_excess, true_alpha, true_beta


def test_known_parameter_recovery(synthetic_capm_data):
    """Test recovery of known alpha and beta under OLS regression."""
    y, x, true_alpha, true_beta = synthetic_capm_data
    result, residuals = fit_capm_ols(y, x, symbol="TEST_FUND")
    
    assert isinstance(result, CAPMResult)
    assert result.n_observations == 2000
    # Parameter estimates should recover true parameters with high precision
    np.testing.assert_allclose(result.alpha_daily, true_alpha, atol=1e-4)
    np.testing.assert_allclose(result.beta, true_beta, atol=1e-2)
    assert result.r_squared > 0.90
    assert result.alpha_pvalue < 0.05
    assert result.beta_pvalue < 0.001


def test_confidence_interval_generation(synthetic_capm_data):
    """Test 95% parametric confidence interval generation for alpha and beta."""
    y, x, true_alpha, true_beta = synthetic_capm_data
    result, _ = fit_capm_ols(y, x, symbol="TEST_FUND")
    
    # 95% CI bounds should bracket true parameters
    assert result.alpha_daily_ci95_lower <= true_alpha <= result.alpha_daily_ci95_upper
    assert result.beta_ci95_lower <= true_beta <= result.beta_ci95_upper
    assert result.alpha_daily_ci95_lower < result.alpha_daily < result.alpha_daily_ci95_upper
    assert result.beta_ci95_lower < result.beta < result.beta_ci95_upper


def test_annualization_and_units(synthetic_capm_data):
    """Test daily vs annualized alpha unit conversion."""
    y, x, true_alpha, _ = synthetic_capm_data
    result, _ = fit_capm_ols(y, x, symbol="TEST_FUND")
    
    # Check that annualized alpha is computed via compounding: ((1 + alpha_daily)^252 - 1) * 100
    expected_annual_pct = ((1.0 + result.alpha_daily) ** 252 - 1.0) * 100.0
    np.testing.assert_almost_equal(result.alpha_annual_pct, expected_annual_pct)
    assert result.alpha_annual_pct_ci95_lower < result.alpha_annual_pct < result.alpha_annual_pct_ci95_upper


def test_diagnostics(synthetic_capm_data):
    """Test computation of regression diagnostics (Durbin-Watson, Jarque-Bera, Kurtosis)."""
    y, x, _, _ = synthetic_capm_data
    result, residuals = fit_capm_ols(y, x, symbol="TEST_FUND")
    
    assert abs(result.residual_mean) < 1e-12  # OLS residual mean is zero by construction
    assert result.residual_std > 0
    assert 1.5 < result.durbin_watson < 2.5   # DW should be ~2 for independent i.i.d. noise
    assert result.jarque_bera_stat >= 0


def test_multiple_fund_analysis():
    """Test run_capm_analysis over multi-fund DataFrame."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    
    df1 = pd.DataFrame({
        "Date": dates,
        "symbol": "FUND_A",
        "excess_fund_return": 0.0001 + 1.1 * np.random.normal(0, 0.01, n) + np.random.normal(0, 0.001, n),
        "excess_market_return": np.random.normal(0, 0.01, n)
    })
    df2 = pd.DataFrame({
        "Date": dates,
        "symbol": "FUND_B",
        "excess_fund_return": -0.0002 + 0.9 * np.random.normal(0, 0.01, n) + np.random.normal(0, 0.001, n),
        "excess_market_return": np.random.normal(0, 0.01, n)
    })
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    summary_df, results_dict, resids_dict = run_capm_analysis(df_combined)
    
    assert len(summary_df) == 2
    assert "FUND_A" in results_dict
    assert "FUND_B" in results_dict
    assert len(resids_dict["FUND_A"]) == n
    assert len(resids_dict["FUND_B"]) == n


# ---------------------------------------------------------------------------
# HAC inference and independent verification (added in review v2.0.0)
# ---------------------------------------------------------------------------

def _reference_ols(y, x):
    """Independent reference estimate using numpy.linalg.lstsq (not the normal-equation code path)."""
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, resid


def test_capm_matches_independent_lstsq(synthetic_capm_data):
    y, x, _, _ = synthetic_capm_data
    result, residuals = fit_capm_ols(y, x, symbol="REF")
    coef, ref_resid = _reference_ols(y, x)
    np.testing.assert_allclose(result.alpha_daily, coef[0], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result.beta, coef[1], rtol=1e-10)
    np.testing.assert_allclose(residuals, ref_resid, atol=1e-14)


def test_hac_fields_populated_and_consistent(synthetic_capm_data):
    y, x, _, _ = synthetic_capm_data
    result, _ = fit_capm_ols(y, x, symbol="HAC")
    assert result.hac_lags == int(np.floor(4.0 * (2000 / 100.0) ** (2.0 / 9.0))) == 7   # Newey-West plug-in rule
    assert result.alpha_stderr_hac > 0
    assert 0.0 <= result.alpha_pvalue_hac <= 1.0
    assert result.alpha_tstat_hac == pytest.approx(result.alpha_daily / result.alpha_stderr_hac)


def test_hac_disabled_flag_skips_hac(synthetic_capm_data):
    y, x, _, _ = synthetic_capm_data
    result, _ = fit_capm_ols(y, x, symbol="NOHAC", compute_hac=False)
    assert result.hac_lags == 0
    assert np.isnan(result.alpha_pvalue_hac)


def test_hac_widens_standard_error_for_autocorrelated_residuals():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.normal(0, 0.01, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.003)
    y = 0.0 + 1.0 * x + e
    result, _ = fit_capm_ols(y, x, symbol="AR1")
    assert result.durbin_watson < 1.0                              # strong positive autocorrelation
    assert result.alpha_stderr_hac > 1.3 * result.alpha_stderr     # i.i.d. SE is badly understated
    assert result.alpha_pvalue_hac > result.alpha_pvalue


def test_alpha_annualization_helper_roundtrip():
    from src.capm import annualize_alpha_pct
    assert annualize_alpha_pct(0.0) == 0.0
    assert annualize_alpha_pct(0.0004) == pytest.approx(((1.0004) ** 252 - 1) * 100)
    assert annualize_alpha_pct(-0.0004) < 0


def test_run_capm_analysis_accepts_integer_symbols():
    """Symbols read back from CSV as integers must not break grouping or lookup."""
    n = 60
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": 122639,
        "excess_fund_return": rng.normal(0, 0.01, n),
        "excess_market_return": rng.normal(0, 0.01, n),
    })
    summary, results, _ = run_capm_analysis(df)
    assert "122639" in results and len(summary) == 1
