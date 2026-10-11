"""
Four-Factor APT / Multi-Factor Analysis Engine
Fund Manager Skill Detector

Fits Carhart 4-Factor OLS linear regressions:
R_fund,t - R_f,t = alpha + beta_mkt * (Mkt-RF)_t + beta_smb * SMB_t + beta_hml * HML_t + beta_mom * Mom_t + error_t

Calculates factor loadings (market, SMB, HML, Momentum), standard errors, t-statistics,
p-values, 95% confidence intervals, diagnostics, and compares four-factor alpha against
single-factor CAPM alpha to measure factor attribution and alpha reduction.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.stats_utils import default_hac_lags, newey_west_cov, two_sided_t_pvalue

from src.capm import compute_durbin_watson


@dataclass
class FactorModelResult:
    symbol: str
    n_observations: int
    alpha_daily: float
    alpha_annual_pct: float
    alpha_se: float
    alpha_tstat: float
    alpha_pvalue: float
    alpha_ci_lower: float
    alpha_ci_upper: float
    
    # Market Factor (Mkt-RF)
    beta_market: float
    beta_market_se: float
    beta_market_tstat: float
    beta_market_pvalue: float
    beta_market_ci_lower: float
    beta_market_ci_upper: float

    # Size Factor (SMB)
    beta_smb: float
    beta_smb_se: float
    beta_smb_tstat: float
    beta_smb_pvalue: float
    beta_smb_ci_lower: float
    beta_smb_ci_upper: float

    # Value Factor (HML)
    beta_hml: float
    beta_hml_se: float
    beta_hml_tstat: float
    beta_hml_pvalue: float
    beta_hml_ci_lower: float
    beta_hml_ci_upper: float

    # Momentum Factor (Mom/UMD)
    beta_momentum: float
    beta_momentum_se: float
    beta_momentum_tstat: float
    beta_momentum_pvalue: float
    beta_momentum_ci_lower: float
    beta_momentum_ci_upper: float

    # Diagnostics
    r_squared: float
    adjusted_r_squared: float
    residual_std: float
    durbin_watson: float
    jarque_bera: float
    jarque_bera_pvalue: float

    # CAPM Comparison
    capm_alpha_daily: float
    capm_alpha_annual_pct: float
    factor_alpha_daily: float
    factor_alpha_annual_pct: float
    alpha_reduction: float
    alpha_reduction_pct: float
    alpha_remains_positive: bool
    alpha_significant_ols: bool
    factor_alpha_lower_than_capm: bool

    # Metadata
    methodology: str
    methodology_version: str
    alpha_pvalue_hac: float = float("nan")
    alpha_se_hac: float = float("nan")
    alpha_tstat_hac: float = float("nan")
    hac_lags: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fit_four_factor_ols(
    excess_fund_return: np.ndarray,
    mkt_rf: np.ndarray,
    smb: np.ndarray,
    hml: np.ndarray,
    mom: np.ndarray,
    symbol: str = "FUND",
    capm_alpha_daily: Optional[float] = None,
    capm_alpha_annual_pct: Optional[float] = None,
    trading_days_per_year: int = 252
) -> Tuple[FactorModelResult, np.ndarray]:
    """
    Fit Carhart 4-Factor Ordinary Least Squares multi-factor regression:
    Y_t = alpha + beta_mkt * Mkt_RF_t + beta_smb * SMB_t + beta_hml * HML_t + beta_mom * Mom_t + error_t
    
    Parameters:
    -----------
    excess_fund_return: np.ndarray (Y)
    mkt_rf: np.ndarray (Market Risk Premium)
    smb: np.ndarray (Size Factor)
    hml: np.ndarray (Value Factor)
    mom: np.ndarray (Momentum Factor)
    symbol: str (Fund ticker)
    capm_alpha_daily: Optional[float]
    capm_alpha_annual_pct: Optional[float]
    trading_days_per_year: int (252)

    Returns:
    --------
    Tuple[FactorModelResult, np.ndarray (residuals)]
    """
    y = np.asarray(excess_fund_return, dtype=np.float64)
    x1 = np.asarray(mkt_rf, dtype=np.float64)
    x2 = np.asarray(smb, dtype=np.float64)
    x3 = np.asarray(hml, dtype=np.float64)
    x4 = np.asarray(mom, dtype=np.float64)

    # Valid non-missing mask across all 5 variables
    mask = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2) | np.isnan(x3) | np.isnan(x4))
    y = y[mask]
    x1 = x1[mask]
    x2 = x2[mask]
    x3 = x3[mask]
    x4 = x4[mask]

    n = len(y)
    k = 4  # 4 factors
    if n <= k + 1:
        raise ValueError(f"Insufficient observations (n={n}) for 4-factor OLS regression for {symbol}.")

    # Design matrix X = [1, Mkt-RF, SMB, HML, Mom]
    X = np.column_stack([np.ones(n), x1, x2, x3, x4])

    # OLS Estimator beta_hat = (X^T X)^{-1} X^T Y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y

    alpha_daily = float(beta_hat[0])
    b_mkt = float(beta_hat[1])
    b_smb = float(beta_hat[2])
    b_hml = float(beta_hat[3])
    b_mom = float(beta_hat[4])

    # Residuals & Degrees of Freedom Error
    y_pred = X @ beta_hat
    residuals = y - y_pred
    sse = float(np.sum(residuals ** 2))
    df_e = n - k - 1  # n - 5

    sigma2 = sse / df_e
    var_cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_cov))

    alpha_se = float(se[0])
    b_mkt_se = float(se[1])
    b_smb_se = float(se[2])
    b_hml_se = float(se[3])
    b_mom_se = float(se[4])

    # t-statistics and two-tailed p-values
    t_stats = [beta_hat[j] / se[j] if se[j] > 0 else 0.0 for j in range(5)]
    p_vals = [float(2 * (1 - stats.t.cdf(abs(t_stats[j]), df=df_e))) for j in range(5)]

    # 95% Parametric Confidence Intervals (t_{df_e, 0.975})
    t_crit = float(stats.t.ppf(0.975, df=df_e))
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se

    # R-squared & Adjusted R-squared
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0
    adj_r2 = float(1.0 - ((1.0 - r2) * (n - 1) / df_e))

    # Newey-West HAC inference for alpha. Classical OLS values remain available
    # for diagnostics, but HAC p-values drive evidence classifications.
    hac_lags = default_hac_lags(n)
    hac_cov = newey_west_cov(X, residuals, hac_lags)
    alpha_se_hac = float(np.sqrt(max(hac_cov[0, 0], 0.0)))
    alpha_t_hac = alpha_daily / alpha_se_hac if alpha_se_hac > 0 else 0.0
    alpha_p_hac = two_sided_t_pvalue(alpha_t_hac, df_e)

    # Annualization of Alpha
    alpha_annual_pct = float(((1.0 + alpha_daily) ** trading_days_per_year - 1.0) * 100.0)

    # Diagnostics
    res_std = float(np.std(residuals, ddof=1))
    dw_stat = compute_durbin_watson(residuals)
    jb_res = stats.jarque_bera(residuals)
    jb_stat = float(jb_res.statistic)
    jb_pval = float(jb_res.pvalue)

    # CAPM Comparison calculations
    if capm_alpha_daily is None:
        capm_alpha_daily = alpha_daily  # Default fallback if unprovided
    if capm_alpha_annual_pct is None:
        capm_alpha_annual_pct = float(((1.0 + capm_alpha_daily) ** trading_days_per_year - 1.0) * 100.0)

    alpha_reduction = float(capm_alpha_daily - alpha_daily)
    
    # Percentage reduction is interpretable only when CAPM alpha is positive
    # and not numerically close to zero. Otherwise report NaN, not a misleading
    # huge percentage caused by dividing by a tiny/negative baseline.
    if capm_alpha_daily > 1e-5:
        alpha_reduction_pct = float(((capm_alpha_daily - alpha_daily) / capm_alpha_daily) * 100.0)
    else:
        alpha_reduction_pct = float("nan")

    alpha_remains_pos = bool(alpha_daily > 0)
    alpha_sig_ols = bool(p_vals[0] < 0.05)
    factor_alpha_lower = bool(alpha_daily < capm_alpha_daily)

    result = FactorModelResult(
        symbol=symbol,
        n_observations=n,
        alpha_daily=alpha_daily,
        alpha_annual_pct=alpha_annual_pct,
        alpha_se=alpha_se,
        alpha_tstat=float(t_stats[0]),
        alpha_pvalue=p_vals[0],
        alpha_ci_lower=float(ci_lower[0]),
        alpha_ci_upper=float(ci_upper[0]),
        
        beta_market=b_mkt,
        beta_market_se=b_mkt_se,
        beta_market_tstat=float(t_stats[1]),
        beta_market_pvalue=p_vals[1],
        beta_market_ci_lower=float(ci_lower[1]),
        beta_market_ci_upper=float(ci_upper[1]),

        beta_smb=b_smb,
        beta_smb_se=b_smb_se,
        beta_smb_tstat=float(t_stats[2]),
        beta_smb_pvalue=p_vals[2],
        beta_smb_ci_lower=float(ci_lower[2]),
        beta_smb_ci_upper=float(ci_upper[2]),

        beta_hml=b_hml,
        beta_hml_se=b_hml_se,
        beta_hml_tstat=float(t_stats[3]),
        beta_hml_pvalue=p_vals[3],
        beta_hml_ci_lower=float(ci_lower[3]),
        beta_hml_ci_upper=float(ci_upper[3]),

        beta_momentum=b_mom,
        beta_momentum_se=b_mom_se,
        beta_momentum_tstat=float(t_stats[4]),
        beta_momentum_pvalue=p_vals[4],
        beta_momentum_ci_lower=float(ci_lower[4]),
        beta_momentum_ci_upper=float(ci_upper[4]),

        r_squared=r2,
        adjusted_r_squared=adj_r2,
        residual_std=res_std,
        durbin_watson=dw_stat,
        jarque_bera=jb_stat,
        jarque_bera_pvalue=jb_pval,

        capm_alpha_daily=capm_alpha_daily,
        capm_alpha_annual_pct=capm_alpha_annual_pct,
        factor_alpha_daily=alpha_daily,
        factor_alpha_annual_pct=alpha_annual_pct,
        alpha_reduction=alpha_reduction,
        alpha_reduction_pct=alpha_reduction_pct,
        alpha_remains_positive=alpha_remains_pos,
        alpha_significant_ols=alpha_sig_ols,
        factor_alpha_lower_than_capm=factor_alpha_lower,

        methodology="Carhart 4-Factor OLS Linear Regression (Mkt-RF, SMB, HML, Mom)",
        methodology_version="1.1.0 (HAC alpha inference)",
        alpha_pvalue_hac=alpha_p_hac,
        alpha_se_hac=alpha_se_hac,
        alpha_tstat_hac=alpha_t_hac,
        hac_lags=hac_lags,
    )

    return result, residuals


def run_four_factor_analysis(
    aligned_df: pd.DataFrame,
    capm_results_dict: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, FactorModelResult], Dict[str, np.ndarray]]:
    """
    Run Carhart 4-Factor OLS regression analysis across all mutual funds present in aligned dataset.
    
    Returns:
    --------
    Tuple[summary_df, results_dict, residuals_dict]
    """
    required_cols = ["excess_fund_return", "Mkt-RF", "SMB", "HML", "Mom"]
    missing = [c for c in required_cols if c not in aligned_df.columns]
    if missing:
        raise KeyError(f"Aligned dataset missing required factor columns: {missing}")

    results_list = []
    results_dict = {}
    residuals_dict = {}

    for symbol, group in aligned_df.groupby("symbol"):
        group = group.sort_values("Date")
        y = group["excess_fund_return"].values
        x1 = group["Mkt-RF"].values
        x2 = group["SMB"].values
        x3 = group["HML"].values
        x4 = group["Mom"].values

        capm_alpha_daily = None
        capm_alpha_annual = None
        if capm_results_dict and symbol in capm_results_dict:
            capm_obj = capm_results_dict[symbol]
            if isinstance(capm_obj, dict):
                capm_alpha_daily = capm_obj.get("alpha_daily")
                capm_alpha_annual = capm_obj.get("alpha_annual_pct")
            else:
                capm_alpha_daily = getattr(capm_obj, "alpha_daily", None)
                capm_alpha_annual = getattr(capm_obj, "alpha_annual_pct", None)

        factor_res, residuals = fit_four_factor_ols(
            excess_fund_return=y,
            mkt_rf=x1,
            smb=x2,
            hml=x3,
            mom=x4,
            symbol=symbol,
            capm_alpha_daily=capm_alpha_daily,
            capm_alpha_annual_pct=capm_alpha_annual
        )

        results_list.append(factor_res.to_dict())
        results_dict[symbol] = factor_res
        residuals_dict[symbol] = residuals

    summary_df = pd.DataFrame(results_list)
    return summary_df, results_dict, residuals_dict


def save_factor_model_results(
    summary_df: pd.DataFrame,
    results_dict: Dict[str, FactorModelResult],
    output_dir: str
) -> Tuple[str, str]:
    """
    Save factor model results to data/processed/factor_model_results.csv and factor_model_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "factor_model_results.csv")
    json_path = os.path.join(output_dir, "factor_model_results.json")

    summary_df.to_csv(csv_path, index=False)

    json_data = {sym: res.to_dict() for sym, res in results_dict.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"[FACTOR MODEL ENGINE] Exported four-factor results for {len(summary_df)} funds to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    return csv_path, json_path


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    capm_json_path = os.path.join(root, "data", "processed", "capm_results.json")
    out_dir = os.path.join(root, "data", "processed")

    if os.path.exists(aligned_path):
        df_aligned = pd.read_csv(aligned_path)
        capm_dict = None
        if os.path.exists(capm_json_path):
            with open(capm_json_path, "r", encoding="utf-8") as f:
                capm_dict = json.load(f)

        summary, factor_dict, factor_resids = run_four_factor_analysis(df_aligned, capm_dict)
        save_factor_model_results(summary, factor_dict, out_dir)
    else:
        print("Aligned dataset not found. Run Task 3 pipeline first.")
