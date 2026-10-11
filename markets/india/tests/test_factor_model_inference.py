"""
HAC inference and independent verification for src/factor_model.py (added in review, version 2.0.0).
"""

import numpy as np
import pandas as pd
import pytest

from src.factor_model import fit_four_factor_ols, run_four_factor_analysis


@pytest.fixture
def factor_data():
    rng = np.random.default_rng(42)
    n = 1500
    mkt = rng.normal(0, 0.01, n)
    smb = rng.normal(0, 0.005, n)
    hml = rng.normal(0, 0.005, n)
    mom = rng.normal(0, 0.006, n)
    y = 0.0003 + 1.05 * mkt + 0.25 * smb - 0.30 * hml + 0.15 * mom + rng.normal(0, 0.001, n)
    return y, mkt, smb, hml, mom


def test_four_factor_matches_independent_lstsq(factor_data):
    y, mkt, smb, hml, mom = factor_data
    res, residuals = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="REF")
    X = np.column_stack([np.ones(len(y)), mkt, smb, hml, mom])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(
        [res.alpha_daily, res.beta_market, res.beta_smb, res.beta_hml, res.beta_momentum],
        coef, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(residuals, y - X @ coef, atol=1e-13)


def test_four_factor_hac_fields(factor_data):
    y, mkt, smb, hml, mom = factor_data
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="HAC", capm_alpha_daily=0.0006)
    assert res.hac_lags > 0 and res.alpha_se_hac > 0
    assert 0.0 <= res.alpha_pvalue_hac <= 1.0
    assert res.alpha_significant_hac == (res.alpha_pvalue_hac < 0.05)
    assert res.methodology_version == "2.0.0"


def test_alpha_reduction_in_annual_points(factor_data):
    y, mkt, smb, hml, mom = factor_data
    capm_daily = 0.0008
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="PTS", capm_alpha_daily=capm_daily)
    expected = ((1 + capm_daily) ** 252 - 1) * 100 - res.alpha_annual_pct
    assert res.alpha_reduction_pts_annual == pytest.approx(expected)


def test_negative_capm_alpha_reduction_is_undefined(factor_data):
    y, mkt, smb, hml, mom = factor_data
    res, _ = fit_four_factor_ols(y, mkt, smb, hml, mom, symbol="NEG", capm_alpha_daily=-0.0004)
    assert np.isnan(res.alpha_reduction_pct)  # percentage reduction is not meaningful for a negative baseline


def test_run_four_factor_analysis_handles_integer_symbol_keys():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": 122639,
        "excess_fund_return": rng.normal(0, 0.01, n),
        "Mkt-RF": rng.normal(0, 0.01, n), "SMB": rng.normal(0, 0.004, n),
        "HML": rng.normal(0, 0.004, n), "Mom": rng.normal(0, 0.005, n)})
    summary, results, _ = run_four_factor_analysis(df, {122639: {"alpha_daily": 0.0001, "alpha_annual_pct": 2.5}})
    assert summary.loc[0, "capm_alpha_daily"] == pytest.approx(0.0001)
