"""
Shared statistical helpers
Indian Fund Manager Skill Detector

* Newey-West (Bartlett kernel) heteroskedasticity-and-autocorrelation-consistent
  (HAC) covariance estimator for OLS.
* Benjamini-Hochberg false-discovery-rate (FDR) adjustment for the cross-section
  of fund-level tests.

Why HAC: daily fund and market returns show volatility clustering and residual
autocorrelation (Durbin-Watson deviates from 2 for several funds), so the
classical OLS standard error sigma^2 (X'X)^{-1} is not reliable for inference.
"""

from typing import Optional, Sequence

import numpy as np
from scipy import stats


def default_hac_lags(n_obs: int) -> int:
    """Newey-West (1994) plug-in bandwidth: floor(4 * (n / 100)^(2/9))."""
    if n_obs < 2:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(
    X: np.ndarray,
    residuals: np.ndarray,
    lags: Optional[int] = None,
    small_sample_correction: bool = True,
) -> np.ndarray:
    """
    Newey-West HAC covariance matrix of the OLS coefficient vector.

        V = (X'X)^{-1} S (X'X)^{-1} * n / (n - k)

        S = sum_t u_t^2 x_t x_t'
            + sum_{l=1..L} w_l sum_{t>l} u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')

    with Bartlett weights w_l = 1 - l / (L + 1).

    Parameters
    ----------
    X : (n, k) design matrix (including the intercept column)
    residuals : (n,) OLS residuals
    lags : bandwidth L (default: ``default_hac_lags(n)``)
    small_sample_correction : multiply by n / (n - k)
    """
    X = np.asarray(X, dtype=np.float64)
    u = np.asarray(residuals, dtype=np.float64)
    n, k = X.shape
    if lags is None:
        lags = default_hac_lags(n)
    lags = int(max(0, min(lags, n - 1)))

    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = Xu[lag:].T @ Xu[:-lag]
        S += weight * (gamma + gamma.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    if small_sample_correction and n > k:
        cov = cov * (n / (n - k))
    return cov


def two_sided_t_pvalue(t_stat: float, df: float) -> float:
    """Two-sided p-value from Student's t, using the survival function (no cancellation)."""
    return float(2.0 * stats.t.sf(abs(t_stat), df=df))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) step-up FDR-adjusted p-values ("q-values").

    q_(i) = min_{j >= i} min(1, p_(j) * m / j)   for sorted p_(1) <= ... <= p_(m)
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
