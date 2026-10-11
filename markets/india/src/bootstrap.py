"""
Circular Block Bootstrap (CBB) Significance Engine
Indian Fund Manager Skill Detector

Performs non-parametric Circular Block Bootstrap resampling of paired
(fund excess return, market excess return) observations to evaluate the statistical
significance of CAPM Jensen's alpha while preserving time-series autocorrelation
and volatility clustering.

Hypothesis framework
--------------------
The bootstrap distribution of alpha is re-centred on zero (alpha_null = alpha* - alpha_hat),
which imposes H0: alpha = 0.

  * ``p_value``            two-sided: P(|alpha_null| >= |alpha_hat|)   (smoothed, +1 / B+1)
  * ``p_value_one_sided``  one-sided (H1: alpha > 0): P(alpha_null >= alpha_hat)
  * ``significantly_positive_5pct`` / ``_1pct``: two-sided p below the level AND alpha_hat > 0.
    This is the flag used as evidence of skill: a significantly NEGATIVE alpha is
    significant in the two-sided sense but is evidence of under-performance, not skill.

Cross-sectional multiplicity: ``run_bootstrap_analysis`` additionally reports
Benjamini-Hochberg FDR-adjusted p-values (``p_value_bh``) across the fund universe.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional, Sequence
import numpy as np
import pandas as pd

from src.capm import CAPMResult, annualize_alpha_pct
from src.stats_utils import benjamini_hochberg


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
    p_value_one_sided: float = float("nan")
    significantly_positive_5pct: bool = False
    significantly_positive_1pct: bool = False
    p_value_bh: float = float("nan")

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


def _ols_alpha(y: np.ndarray, x: np.ndarray) -> float:
    """Closed-form OLS intercept of y on [1, x] (identical to the normal-equation solution)."""
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0.0:
        raise ValueError("Degenerate bootstrap resample: market returns have zero variance.")
    beta = np.sum((x - x_mean) * (y - y.mean())) / sxx
    return float(y.mean() - beta * x_mean)


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

    # Execute B bootstrap replications (paired resampling keeps fund/market observations together)
    for b in range(n_bootstrap):
        indices = generate_circular_block_indices(n, block_length, rng)
        boot_alphas[b] = _ols_alpha(y[indices], x[indices])

    # 1. Uncentered sampling distribution around observed alpha
    boot_mean = float(np.mean(boot_alphas))
    boot_std = float(np.std(boot_alphas, ddof=1))

    ci_lower = float(np.percentile(boot_alphas, 2.5))
    ci_median = float(np.percentile(boot_alphas, 50.0))
    ci_upper = float(np.percentile(boot_alphas, 97.5))

    # Annualization of CIs
    obs_annual_pct = annualize_alpha_pct(observed_alpha_daily, trading_days_per_year)
    ci_ann_lower = annualize_alpha_pct(ci_lower, trading_days_per_year)
    ci_ann_upper = annualize_alpha_pct(ci_upper, trading_days_per_year)

    # 2. Null-centered hypothesis-testing distribution (imposes H0: alpha = 0)
    # Centering transform: alpha_null = alpha_boot - observed_alpha
    boot_alphas_null = boot_alphas - observed_alpha_daily

    # Two-sided empirical p-value for H0: alpha = 0 (smoothed: +1 in numerator and denominator)
    abs_obs = abs(observed_alpha_daily)
    count_exceed = np.sum(np.abs(boot_alphas_null) >= abs_obs)
    p_value = float((count_exceed + 1) / (n_bootstrap + 1))

    # One-sided empirical p-value for H1: alpha > 0
    count_upper = np.sum(boot_alphas_null >= observed_alpha_daily)
    p_value_one_sided = float((count_upper + 1) / (n_bootstrap + 1))

    sig_5pct = bool(p_value < 0.05)
    sig_1pct = bool(p_value < 0.01)
    alpha_positive = bool(observed_alpha_daily > 0)

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
        methodology="Circular Block Bootstrap (CBB) Null-Centered Paired Resampling",
        p_value_one_sided=p_value_one_sided,
        significantly_positive_5pct=bool(sig_5pct and alpha_positive),
        significantly_positive_1pct=bool(sig_1pct and alpha_positive),
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
    Adds Benjamini-Hochberg FDR-adjusted two-sided p-values across the fund universe.

    Returns:
    --------
    Tuple[summary_df, results_dict, boot_distributions_dict]
    """
    results_dict: Dict[str, BootstrapResult] = {}
    distributions_dict = {}

    df_proc = aligned_df.copy()
    df_proc["symbol"] = df_proc["symbol"].astype(str)

    for symbol, group in df_proc.groupby("symbol"):
        group = group.sort_values("Date")
        y = group["excess_fund_return"].values
        x = group["excess_market_return"].values

        capm_res = capm_results_dict.get(str(symbol))
        if capm_res is None:
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

        results_dict[symbol] = boot_res
        distributions_dict[symbol] = boot_alphas

    # Cross-sectional multiple-testing adjustment (Benjamini-Hochberg FDR)
    symbols = list(results_dict.keys())
    q_values = benjamini_hochberg([results_dict[s].p_value for s in symbols])
    for sym, q in zip(symbols, q_values):
        results_dict[sym].p_value_bh = float(q)

    summary_df = pd.DataFrame([results_dict[s].to_dict() for s in symbols])
    return summary_df, results_dict, distributions_dict


def run_block_length_sensitivity(
    aligned_df: pd.DataFrame,
    block_lengths: Sequence[int] = (1, 5, 10, 20, 30, 60),
    n_bootstrap: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Re-run the CBB test for several block lengths to expose how sensitive each fund's
    significance conclusion is to the (arbitrary) block length. b = 1 is the i.i.d. bootstrap.

    Returns a long DataFrame: symbol, block_length, observed alpha, two-sided p, one-sided p,
    annualised 95% CI, and significantly_positive_5pct.
    """
    df_proc = aligned_df.copy()
    df_proc["symbol"] = df_proc["symbol"].astype(str)
    rows: List[Dict[str, Any]] = []
    for symbol, group in df_proc.groupby("symbol"):
        group = group.sort_values("Date")
        y = group["excess_fund_return"].values
        x = group["excess_market_return"].values
        a_hat = _ols_alpha(y, x)
        beta_hat = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        for b in block_lengths:
            res, _, _ = run_circular_block_bootstrap(
                y, x, a_hat, beta_hat, symbol=str(symbol),
                block_length=int(b), n_bootstrap=n_bootstrap, seed=seed,
            )
            rows.append({
                "symbol": str(symbol),
                "block_length": int(b),
                "observed_alpha_annual_pct": res.observed_alpha_annual_pct,
                "p_value_two_sided": res.p_value,
                "p_value_one_sided": res.p_value_one_sided,
                "ci_annual_lower_pct": res.ci_annual_lower_pct,
                "ci_annual_upper_pct": res.ci_annual_upper_pct,
                "significantly_positive_5pct": res.significantly_positive_5pct,
            })
    return pd.DataFrame(rows)


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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    capm_json_path = os.path.join(root, "data", "processed", "capm_results.json")
    out_dir = os.path.join(root, "data", "processed")

    if os.path.exists(aligned_path) and os.path.exists(capm_json_path):
        df_aligned = pd.read_csv(aligned_path, dtype={"symbol": str})
        with open(capm_json_path, "r", encoding="utf-8") as f:
            capm_dict_raw = json.load(f)

        capm_results = {k: CAPMResult(**v) for k, v in capm_dict_raw.items()}
        summary, boot_dict, boot_dists = run_bootstrap_analysis(
            df_aligned, capm_results, block_length=30, n_bootstrap=1000, seed=42
        )
        save_bootstrap_results(summary, boot_dict, out_dir)

        sens = run_block_length_sensitivity(df_aligned, n_bootstrap=1000, seed=42)
        sens_path = os.path.join(out_dir, "bootstrap_block_sensitivity.csv")
        sens.to_csv(sens_path, index=False)
        print(f"  - Block-length sensitivity: {sens_path}")
    else:
        print("Required input files not found. Run the data pipeline and CAPM engine first.")
