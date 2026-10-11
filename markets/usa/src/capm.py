"""
Single-Factor CAPM Analysis Engine
Fund Manager Skill Detector

Fits single-factor OLS Capital Asset Pricing Model regressions:
Fund Excess Return_t = alpha + beta * Market Excess Return_t + error_t

Calculates parameter estimates, standard errors, t-statistics, p-values, 95% CIs,
daily/annualized alpha metrics, and regression diagnostics (skewness, kurtosis, Durbin-Watson, Jarque-Bera).
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.stats_utils import default_hac_lags, newey_west_cov, two_sided_t_pvalue


@dataclass
class CAPMResult:
    symbol: str
    n_observations: int
    alpha_daily: float
    beta: float
    alpha_stderr: float
    beta_stderr: float
    alpha_tstat: float
    beta_tstat: float
    alpha_pvalue: float
    beta_pvalue: float
    r_squared: float
    adj_r_squared: float
    residual_mean: float
    residual_std: float
    alpha_daily_ci95_lower: float
    alpha_daily_ci95_upper: float
    beta_ci95_lower: float
    beta_ci95_upper: float
    alpha_annual_pct: float
    alpha_annual_pct_ci95_lower: float
    alpha_annual_pct_ci95_upper: float
    residual_skewness: float
    residual_kurtosis: float
    durbin_watson: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    alpha_pvalue_hac: float = float("nan")
    alpha_stderr_hac: float = float("nan")
    alpha_tstat_hac: float = float("nan")
    hac_lags: int = 0

    def to_dict(self, include_residuals: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        return d


def compute_durbin_watson(residuals: np.ndarray) -> float:
    """Compute Durbin-Watson statistic for 1st-order residual autocorrelation."""
    diff = np.diff(residuals)
    denom = np.sum(residuals ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(diff ** 2) / denom)


def fit_capm_ols(
    excess_fund_return: np.ndarray,
    excess_market_return: np.ndarray,
    symbol: str = "FUND",
    trading_days_per_year: int = 252
) -> Tuple[CAPMResult, np.ndarray]:
    """
    Fit Ordinary Least Squares single-factor CAPM regression:
    Y_t = alpha + beta * X_t + error_t
    
    Parameters:
    -----------
    excess_fund_return: np.ndarray (Y)
    excess_market_return: np.ndarray (X)
    symbol: str (Fund ticker)
    trading_days_per_year: int (252)

    Returns:
    --------
    Tuple[CAPMResult, np.ndarray (residuals)]
    """
    y = np.asarray(excess_fund_return, dtype=np.float64)
    x = np.asarray(excess_market_return, dtype=np.float64)

    # Drop any missing observations
    valid_mask = ~(np.isnan(y) | np.isnan(x))
    y = y[valid_mask]
    x = x[valid_mask]

    n = len(y)
    if n < 3:
        raise ValueError(f"Insufficient observations (n={n}) to fit CAPM regression for {symbol}.")

    # Design matrix X = [1, x]
    X = np.column_stack([np.ones(n), x])
    
    # OLS Estimator beta_hat = (X^T X)^{-1} X^T Y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y

    alpha_daily = float(beta_hat[0])
    beta_val = float(beta_hat[1])

    # Residuals and Sum of Squared Errors
    y_pred = X @ beta_hat
    residuals = y - y_pred
    sse = float(np.sum(residuals ** 2))
    df_e = n - 2  # degrees of freedom error

    # Variance-covariance matrix of parameter estimates
    sigma2 = sse / df_e
    var_cov = sigma2 * XtX_inv

    alpha_se = float(np.sqrt(var_cov[0, 0]))
    beta_se = float(np.sqrt(var_cov[1, 1]))

    # t-statistics and two-tailed p-values under OLS normality assumptions
    alpha_tstat = alpha_daily / alpha_se if alpha_se > 0 else 0.0
    beta_tstat = beta_val / beta_se if beta_se > 0 else 0.0

    alpha_pval = float(2 * (1 - stats.t.cdf(abs(alpha_tstat), df=df_e)))
    beta_pval = float(2 * (1 - stats.t.cdf(abs(beta_tstat), df=df_e)))

    # R-squared and Adjusted R-squared
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - (sse / sst)) if sst > 0 else 0.0
    adj_r_squared = float(1.0 - ((1.0 - r_squared) * (n - 1) / (n - 2)))

    # 95% Parametric OLS Confidence Intervals (t_{n-2, 0.975})
    t_crit = float(stats.t.ppf(0.975, df=df_e))
    alpha_ci_lower = alpha_daily - t_crit * alpha_se
    alpha_ci_upper = alpha_daily + t_crit * alpha_se

    beta_ci_lower = beta_val - t_crit * beta_se
    beta_ci_upper = beta_val + t_crit * beta_se

    # Annualization of Alpha
    # Compounded Annualized Alpha Percentage: ((1 + alpha_daily)^252 - 1) * 100
    alpha_annual_pct = float(((1.0 + alpha_daily) ** trading_days_per_year - 1.0) * 100.0)
    alpha_annual_ci_lower = float(((1.0 + alpha_ci_lower) ** trading_days_per_year - 1.0) * 100.0)
    alpha_annual_ci_upper = float(((1.0 + alpha_ci_upper) ** trading_days_per_year - 1.0) * 100.0)

    # Newey-West HAC inference for alpha, used for classification decisions.
    hac_lags = default_hac_lags(n)
    hac_cov = newey_west_cov(X, residuals, hac_lags)
    alpha_se_hac = float(np.sqrt(max(hac_cov[0, 0], 0.0)))
    alpha_t_hac = alpha_daily / alpha_se_hac if alpha_se_hac > 0 else 0.0
    alpha_p_hac = two_sided_t_pvalue(alpha_t_hac, df_e)

    # Diagnostics
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals, ddof=1))
    res_skew = float(stats.skew(residuals))
    res_kurt = float(stats.kurtosis(residuals))  # excess kurtosis
    dw_stat = compute_durbin_watson(residuals)

    jb_res = stats.jarque_bera(residuals)
    jb_stat = float(jb_res.statistic)
    jb_pval = float(jb_res.pvalue)

    result = CAPMResult(
        symbol=symbol,
        n_observations=n,
        alpha_daily=alpha_daily,
        beta=beta_val,
        alpha_stderr=alpha_se,
        beta_stderr=beta_se,
        alpha_tstat=alpha_tstat,
        beta_tstat=beta_tstat,
        alpha_pvalue=alpha_pval,
        beta_pvalue=beta_pval,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        residual_mean=res_mean,
        residual_std=res_std,
        alpha_daily_ci95_lower=alpha_ci_lower,
        alpha_daily_ci95_upper=alpha_ci_upper,
        beta_ci95_lower=beta_ci_lower,
        beta_ci95_upper=beta_ci_upper,
        alpha_annual_pct=alpha_annual_pct,
        alpha_annual_pct_ci95_lower=alpha_annual_ci_lower,
        alpha_annual_pct_ci95_upper=alpha_annual_ci_upper,
        residual_skewness=res_skew,
        residual_kurtosis=res_kurt,
        durbin_watson=dw_stat,
        jarque_bera_stat=jb_stat,
        jarque_bera_pvalue=jb_pval,
        alpha_pvalue_hac=alpha_p_hac,
        alpha_stderr_hac=alpha_se_hac,
        alpha_tstat_hac=alpha_t_hac,
        hac_lags=hac_lags,
    )

    return result, residuals


def run_capm_analysis(aligned_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, CAPMResult], Dict[str, np.ndarray]]:
    """
    Run OLS CAPM regressions across all mutual funds present in aligned dataset.
    
    Returns:
    --------
    Tuple[summary_df, results_dict, residuals_dict]
    """
    if "excess_fund_return" not in aligned_df.columns or "excess_market_return" not in aligned_df.columns:
        raise KeyError("Dataset must contain 'excess_fund_return' and 'excess_market_return' columns.")

    results_list = []
    results_dict = {}
    residuals_dict = {}

    for symbol, group in aligned_df.groupby("symbol"):
        group = group.sort_values("Date")
        y = group["excess_fund_return"].values
        x = group["excess_market_return"].values

        capm_res, residuals = fit_capm_ols(y, x, symbol=symbol)
        results_list.append(capm_res.to_dict())
        results_dict[symbol] = capm_res
        residuals_dict[symbol] = residuals

    summary_df = pd.DataFrame(results_list)
    return summary_df, results_dict, residuals_dict


def save_capm_results(summary_df: pd.DataFrame, results_dict: Dict[str, CAPMResult], output_dir: str) -> Tuple[str, str]:
    """
    Save CAPM results as CSV and machine-readable JSON in data/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "capm_results.csv")
    json_path = os.path.join(output_dir, "capm_results.json")

    # Format numeric columns for CSV export
    summary_df.to_csv(csv_path, index=False)

    json_data = {sym: res.to_dict() for sym, res in results_dict.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"[CAPM ENGINE] Exported CAPM results for {len(summary_df)} funds to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    return csv_path, json_path


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    out_dir = os.path.join(root, "data", "processed")

    if os.path.exists(aligned_path):
        df_aligned = pd.read_csv(aligned_path)
        summary, res_dict, resids = run_capm_analysis(df_aligned)
        save_capm_results(summary, res_dict, out_dir)
    else:
        print(f"Aligned dataset not found at {aligned_path}. Run data pipeline first.")
