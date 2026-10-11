"""
Unit tests for src/stats_utils.py (Newey-West HAC covariance, Benjamini-Hochberg FDR).
"""

import numpy as np
import pytest

from src.stats_utils import (
    benjamini_hochberg,
    default_hac_lags,
    newey_west_cov,
    two_sided_t_pvalue,
)


def _brute_force_newey_west(X, u, lags, small_sample=True):
    """Direct (slow, loop-based) transcription of the textbook formula, used as a reference."""
    n, k = X.shape
    S = np.zeros((k, k))
    for t in range(n):
        S += u[t] ** 2 * np.outer(X[t], X[t])
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        for t in range(l, n):
            term = u[t] * u[t - l] * (np.outer(X[t], X[t - l]) + np.outer(X[t - l], X[t]))
            S += w * term
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ S @ bread
    return cov * n / (n - k) if small_sample else cov


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 0.01, n)
    # AR(1) residuals so that HAC genuinely differs from the i.i.d. formula
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + rng.normal(0, 0.004)
    y = 0.0002 + 0.9 * x + e
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y - X @ beta


def test_newey_west_matches_brute_force(regression_data):
    X, u = regression_data
    for lags in (0, 1, 4, 9):
        fast = newey_west_cov(X, u, lags)
        slow = _brute_force_newey_west(X, u, lags)
        np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-18)


def test_newey_west_zero_lags_is_white_heteroskedastic(regression_data):
    X, u = regression_data
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T * u ** 2) @ X @ bread * n / (n - k)
    np.testing.assert_allclose(newey_west_cov(X, u, 0), white, rtol=1e-10)


def test_hac_exceeds_iid_standard_error_under_positive_autocorrelation(regression_data):
    X, u = regression_data
    n, k = X.shape
    iid = (u @ u / (n - k)) * np.linalg.inv(X.T @ X)
    hac = newey_west_cov(X, u, default_hac_lags(n))
    assert np.sqrt(hac[0, 0]) > np.sqrt(iid[0, 0])


def test_default_lag_rule():
    assert default_hac_lags(1227) == 6          # floor(4 * (12.27)^(2/9))
    assert default_hac_lags(306) == 5
    assert default_hac_lags(100) == 4
    assert default_hac_lags(1) == 0


def test_covariance_is_symmetric_positive_semidefinite(regression_data):
    X, u = regression_data
    cov = newey_west_cov(X, u, 6)
    np.testing.assert_allclose(cov, cov.T, atol=1e-18)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-15)


def test_two_sided_p_value_symmetry_and_range():
    assert two_sided_t_pvalue(0.0, 100) == pytest.approx(1.0)
    assert two_sided_t_pvalue(2.5, 50) == pytest.approx(two_sided_t_pvalue(-2.5, 50))
    assert 0.0 < two_sided_t_pvalue(40.0, 50) < 1e-30      # no cancellation to exactly zero


def test_benjamini_hochberg_hand_computed():
    # sorted p: .005 .01 .03 .04 -> scaled by m/rank: .02 .02 .04 .04 -> monotone q
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_properties():
    p = np.array([0.001, 0.2, 0.049, 0.5, 0.9, 0.03])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-15)                 # adjustment never lowers a p-value
    assert np.all(q <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-15)    # monotone in the original ordering
    assert benjamini_hochberg([]).size == 0
