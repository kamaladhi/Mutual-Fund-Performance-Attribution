"""
Direction-aware inference, FDR adjustment and block-length sensitivity for src/bootstrap.py
(added in review, evaluation version 2.0.0).
"""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    _ols_alpha,
    run_block_length_sensitivity,
    run_bootstrap_analysis,
    run_circular_block_bootstrap,
)
from src.capm import fit_capm_ols


def _series(alpha, n=500, seed=11, noise=0.002):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n)
    y = alpha + 1.0 * x + rng.normal(0, noise, n)
    return y, x


def test_closed_form_alpha_matches_ols_engine():
    y, x = _series(0.0007)
    res, _ = fit_capm_ols(y, x, compute_hac=False)
    assert _ols_alpha(y, x) == pytest.approx(res.alpha_daily, rel=1e-10, abs=1e-15)


def test_degenerate_market_series_raises():
    with pytest.raises(ValueError, match="zero variance"):
        _ols_alpha(np.arange(5.0), np.ones(5))


def test_significantly_negative_alpha_is_not_flagged_as_positive():
    y, x = _series(-0.002)                       # strongly NEGATIVE alpha
    a_hat = _ols_alpha(y, x)
    res, _, _ = run_circular_block_bootstrap(y, x, a_hat, 1.0, "LOSER", block_length=20, n_bootstrap=300, seed=1)
    assert res.significant_at_5pct is True               # two-sided: differs from zero
    assert res.significantly_positive_5pct is False       # ...but it is under-performance, not skill
    assert res.significantly_positive_1pct is False
    assert res.p_value_one_sided > 0.95                   # H1: alpha > 0 is clearly not supported


def test_significantly_positive_alpha_flagged_with_small_one_sided_p():
    y, x = _series(0.002)
    a_hat = _ols_alpha(y, x)
    res, _, _ = run_circular_block_bootstrap(y, x, a_hat, 1.0, "WINNER", block_length=20, n_bootstrap=300, seed=1)
    assert res.significantly_positive_5pct is True
    assert res.significantly_positive_1pct is True
    assert res.p_value_one_sided <= res.p_value          # one-sided <= two-sided for a positive alpha


def test_smoothed_pvalue_never_zero_and_bounded():
    y, x = _series(0.01, noise=0.0005)
    a_hat = _ols_alpha(y, x)
    res, _, _ = run_circular_block_bootstrap(y, x, a_hat, 1.0, "EXTREME", block_length=10, n_bootstrap=200, seed=3)
    assert res.p_value == pytest.approx(1.0 / 201.0)      # (0 + 1) / (B + 1)
    assert 0 < res.p_value_one_sided <= 1


def test_null_distribution_is_centred_on_zero():
    y, x = _series(0.0005)
    a_hat = _ols_alpha(y, x)
    _, boot, null = run_circular_block_bootstrap(y, x, a_hat, 1.0, "C", block_length=20, n_bootstrap=400, seed=9)
    assert np.allclose(null, boot - a_hat)
    assert abs(np.mean(null)) < 4 * np.std(null, ddof=1) / np.sqrt(len(null))   # mean ~ 0 within MC error


def test_run_bootstrap_analysis_adds_bh_adjusted_pvalues():
    frames, capm_dict = [], {}
    for i, alpha in enumerate([0.0015, 0.0, -0.001]):
        y, x = _series(alpha, n=300, seed=20 + i)
        fund = f"F{i}"
        frames.append(pd.DataFrame({
            "Date": pd.date_range("2022-01-03", periods=300, freq="B").strftime("%Y-%m-%d"),
            "symbol": fund, "excess_fund_return": y, "excess_market_return": x}))
        capm_dict[fund], _ = fit_capm_ols(y, x, symbol=fund)
    summary, results, _ = run_bootstrap_analysis(pd.concat(frames), capm_dict, block_length=15, n_bootstrap=100, seed=5)
    assert {"p_value_one_sided", "p_value_bh", "significantly_positive_5pct"} <= set(summary.columns)
    assert (summary["p_value_bh"] >= summary["p_value"] - 1e-12).all()
    assert (summary["p_value_bh"] <= 1.0).all()


def test_block_length_sensitivity_table_shape():
    y, x = _series(0.001, n=240)
    df = pd.DataFrame({"Date": pd.date_range("2022-01-03", periods=240, freq="B").strftime("%Y-%m-%d"),
                       "symbol": "S", "excess_fund_return": y, "excess_market_return": x})
    sens = run_block_length_sensitivity(df, block_lengths=(1, 10, 30), n_bootstrap=60, seed=2)
    assert list(sens["block_length"]) == [1, 10, 30]
    assert (sens["p_value_two_sided"].between(0, 1)).all()
    assert sens["observed_alpha_annual_pct"].nunique() == 1   # same observed alpha for every block length
