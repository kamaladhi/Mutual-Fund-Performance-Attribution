"""
Circular Block Bootstrap (CBB) Significance Engine
Fund Manager Skill Detector

Performs non-parametric Circular Block Bootstrap resampling to evaluate the statistical
significance of CAPM Jensen's Alpha under H0: alpha = 0 while preserving time-series
autocorrelation and volatility clustering.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.capm import fit_capm_ols, CAPMResult


@dataclass
class BootstrapResult:
    symbol: str
    n_observations: int
    observed_alpha_daily: float
    observed_beta: float
    block_length: int
    n_bootstrap: int
    seed: Optional[int]
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_median: float
    ci_upper: float
    p_value: float
    significant_at_5pct: bool
    significant_at_1pct: bool
    observed_alpha_annual_pct: float
    ci_annual_lower_pct: float
    ci_annual_upper_pct: float
    methodology: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_circular_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate circular block resample indices of exact length N.
    
    Parameters:
    -----------
    n_observations: int (N)
    block_length: int (b)
    rng: np.random.Generator

    Returns:
    --------
    np.ndarray of shape (N,) containing 0-indexed resample positions.
    """
    if block_length <= 0 or block_length > n_observations:
        raise ValueError(f"Block length b={block_length} must satisfy 1 <= b <= N={n_observations}.")

    num_blocks = int(np.ceil(n_observations / block_length))
    # Random starting positions in 0..N-1
    start_positions = rng.integers(0, n_observations, size=num_blocks)
    
    indices = []
    for start in start_positions:
        # Generate block indices with circular wrapping
        block_idx = (start + np.arange(block_length)) % n_observations
        indices.extend(block_idx)

    # Trim to exact length N
    final_indices = np.array(indices[:n_observations], dtype=int)
    return final_indices


def run_circular_block_bootstrap(
    excess_fund_return: np.ndarray,
    excess_market_return: np.ndarray,
    observed_alpha_daily: float,
    observed_beta: float,
    symbol: str = "FUND",
    block_length: int = 30,
    n_bootstrap: int = 1000,
    seed: Optional[int] = 42,
    trading_days_per_year: int = 252
) -> Tuple[BootstrapResult, np.ndarray, np.ndarray]:
    """
    Perform Circular Block Bootstrap resampling for a single fund.

    Parameters:
    -----------
    excess_fund_return: np.ndarray (Y)
    excess_market_return: np.ndarray (X)
    observed_alpha_daily: float
    observed_beta: float
    symbol: str
    block_length: int (default 30)
    n_bootstrap: int (default 1000)
    seed: Optional[int] (default 42)
    trading_days_per_year: int (252)

    Returns:
    --------
    Tuple[BootstrapResult, bootstrap_alphas_uncentered, bootstrap_alphas_null_centered]
    """
    y = np.asarray(excess_fund_return, dtype=np.float64)
    x = np.asarray(excess_market_return, dtype=np.float64)

    # Filter invalid entries
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    n = len(y)

    if n < block_length:
        raise ValueError(f"Sample size N={n} is smaller than block length b={block_length} for {symbol}.")

    rng = np.random.default_rng(seed)
    boot_alphas = np.zeros(n_bootstrap, dtype=np.float64)

    # Execute B bootstrap replications
    for b in range(n_bootstrap):
        indices = generate_circular_block_indices(n, block_length, rng)
        y_boot = y[indices]
        x_boot = x[indices]

        # Re-estimate CAPM alpha on paired resample
        capm_res, _ = fit_capm_ols(y_boot, x_boot, symbol=f"{symbol}_boot_{b}")
        boot_alphas[b] = capm_res.alpha_daily

    # 1. Uncentered sampling distribution around observed alpha
    boot_mean = float(np.mean(boot_alphas))
    boot_std = float(np.std(boot_alphas, ddof=1))
    
    ci_lower = float(np.percentile(boot_alphas, 2.5))
    ci_median = float(np.percentile(boot_alphas, 50.0))
    ci_upper = float(np.percentile(boot_alphas, 97.5))

    # Annualization of CIs
    obs_annual_pct = float(((1.0 + observed_alpha_daily) ** trading_days_per_year - 1.0) * 100.0)
    ci_ann_lower = float(((1.0 + ci_lower) ** trading_days_per_year - 1.0) * 100.0)
    ci_ann_upper = float(((1.0 + ci_upper) ** trading_days_per_year - 1.0) * 100.0)

    # 2. Null-Centered Hypothesis Testing Distribution (H0: alpha = 0)
    # Centering transform: alpha_null = alpha_boot - observed_alpha
    boot_alphas_null = boot_alphas - observed_alpha_daily

    # Two-sided empirical p-value for H0: alpha = 0
    # Proportion of null-centered resamples whose absolute magnitude >= observed absolute alpha
    abs_obs = abs(observed_alpha_daily)
    count_exceed = np.sum(np.abs(boot_alphas_null) >= abs_obs)
    p_value = float((count_exceed + 1) / (n_bootstrap + 1))  # Smoothed empirical p-value

    sig_5pct = bool(p_value < 0.05)
    sig_1pct = bool(p_value < 0.01)

    result = BootstrapResult(
        symbol=symbol,
        n_observations=n,
        observed_alpha_daily=observed_alpha_daily,
        observed_beta=observed_beta,
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        seed=seed,
        bootstrap_mean=boot_mean,
        bootstrap_std=boot_std,
        ci_lower=ci_lower,
        ci_median=ci_median,
        ci_upper=ci_upper,
        p_value=p_value,
        significant_at_5pct=sig_5pct,
        significant_at_1pct=sig_1pct,
        observed_alpha_annual_pct=obs_annual_pct,
        ci_annual_lower_pct=ci_ann_lower,
        ci_annual_upper_pct=ci_ann_upper,
        methodology="Circular Block Bootstrap (CBB) Null-Centered Paired Resampling"
    )

    return result, boot_alphas, boot_alphas_null


def run_bootstrap_analysis(
    aligned_df: pd.DataFrame,
    capm_results_dict: Dict[str, CAPMResult],
    block_length: int = 30,
    n_bootstrap: int = 1000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict[str, BootstrapResult], Dict[str, np.ndarray]]:
    """
    Run Circular Block Bootstrap significance testing across all funds in aligned dataset.
    
    Returns:
    --------
    Tuple[summary_df, results_dict, boot_distributions_dict]
    """
    results_list = []
    results_dict = {}
    distributions_dict = {}

    for symbol, group in aligned_df.groupby("symbol"):
        group = group.sort_values("Date")
        y = group["excess_fund_return"].values
        x = group["excess_market_return"].values

        capm_res = capm_results_dict.get(symbol)
        if not capm_res:
            raise KeyError(f"CAPM result for fund '{symbol}' not found in provided dictionary.")

        boot_res, boot_alphas, boot_null = run_circular_block_bootstrap(
            excess_fund_return=y,
            excess_market_return=x,
            observed_alpha_daily=capm_res.alpha_daily,
            observed_beta=capm_res.beta,
            symbol=symbol,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            seed=seed
        )

        results_list.append(boot_res.to_dict())
        results_dict[symbol] = boot_res
        distributions_dict[symbol] = boot_alphas

    summary_df = pd.DataFrame(results_list)
    return summary_df, results_dict, distributions_dict


def save_bootstrap_results(
    summary_df: pd.DataFrame,
    results_dict: Dict[str, BootstrapResult],
    output_dir: str
) -> Tuple[str, str]:
    """
    Save bootstrap results to data/processed/bootstrap_results.csv and bootstrap_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "bootstrap_results.csv")
    json_path = os.path.join(output_dir, "bootstrap_results.json")

    summary_df.to_csv(csv_path, index=False)

    json_data = {sym: res.to_dict() for sym, res in results_dict.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"[BOOTSTRAP ENGINE] Exported bootstrap results for {len(summary_df)} funds to:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")

    return csv_path, json_path


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    capm_json_path = os.path.join(root, "data", "processed", "capm_results.json")
    out_dir = os.path.join(root, "data", "processed")

    if os.path.exists(aligned_path) and os.path.exists(capm_json_path):
        df_aligned = pd.read_csv(aligned_path)
        with open(capm_json_path, "r", encoding="utf-8") as f:
            capm_dict_raw = json.load(f)
        
        capm_results = {k: CAPMResult(**v) for k, v in capm_dict_raw.items()}
        summary, boot_dict, boot_dists = run_bootstrap_analysis(
            df_aligned, capm_results, block_length=30, n_bootstrap=1000, seed=42
        )
        save_bootstrap_results(summary, boot_dict, out_dir)
    else:
        print("Required input files not found. Run Task 3 data pipeline and Task 4 CAPM engine first.")
