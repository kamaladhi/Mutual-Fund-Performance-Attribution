"""
Performance Persistence Analysis Engine
Fund Manager Skill Detector

Splits historical returns into non-overlapping chronological subperiods (4 periods),
estimates subperiod CAPM alphas, computes alpha persistence rates, stability metrics,
and sign consistency indicators across independent time windows.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.capm import fit_capm_ols, CAPMResult


@dataclass
class PeriodCAPMResult:
    symbol: str
    period_id: str
    start_date: str
    end_date: str
    n_observations: int
    alpha_daily: float
    alpha_annual_pct: float
    beta: float
    beta_stderr: float
    beta_tstat: float
    beta_pvalue: float
    beta_ci_lower: float
    beta_ci_upper: float
    alpha_stderr: float
    alpha_tstat: float
    alpha_pvalue: float
    alpha_ci_lower: float
    alpha_ci_upper: float
    r_squared: float
    adj_r_squared: float
    residual_std: float
    methodology: str
    methodology_version: str
    alpha_stderr_hac: float = float("nan")
    alpha_pvalue_hac: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersistenceSummaryResult:
    symbol: str
    total_periods: int
    positive_alpha_periods: int
    significant_positive_alpha_periods: int
    positive_alpha_persistence_rate: float
    significant_positive_alpha_persistence_rate: float
    alpha_mean_annual_pct: float
    alpha_median_annual_pct: float
    alpha_std_annual_pct: float
    alpha_min_annual_pct: float
    alpha_max_annual_pct: float
    alpha_positive_all_periods: bool
    alpha_positive_majority: bool
    significant_positive_majority: bool
    capm_full_sample_alpha_daily: float
    capm_full_sample_alpha_annual_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_chronological_period_splits(unique_dates: List[str], num_periods: int = 4) -> List[Dict[str, Any]]:
    """
    Split sorted unique trading dates into non-overlapping chronological subperiods of equal length.
    
    Parameters:
    -----------
    unique_dates: List[str] (Sorted unique YYYY-MM-DD date strings)
    num_periods: int (Default 4)

    Returns:
    --------
    List of dicts: [{'period_id': 'P1', 'start_date': ..., 'end_date': ..., 'dates': [...]}, ...]
    """
    sorted_dates = sorted(list(set(unique_dates)))
    total_n = len(sorted_dates)

    if total_n < num_periods:
        raise ValueError(f"Total dates ({total_n}) is less than requested periods ({num_periods}).")

    base_len = total_n // num_periods
    remainder = total_n % num_periods

    splits = []
    current_idx = 0

    for p in range(num_periods):
        # Distribute remaining observations to the last period(s) transparently
        p_len = base_len + (1 if p >= (num_periods - remainder) else 0)
        p_dates = sorted_dates[current_idx : current_idx + p_len]
        
        splits.append({
            "period_id": f"P{p + 1}",
            "start_date": p_dates[0],
            "end_date": p_dates[-1],
            "n_dates": len(p_dates),
            "dates": set(p_dates)
        })
        current_idx += p_len

    return splits


def run_period_capm_analysis(
    aligned_df: pd.DataFrame,
    num_periods: int = 4
) -> Tuple[pd.DataFrame, Dict[str, List[PeriodCAPMResult]]]:
    """
    Run OLS CAPM regressions across all funds for each chronological subperiod.
    
    Returns:
    --------
    Tuple[period_summary_df, results_by_fund_dict]
    """
    unique_dates = sorted(aligned_df["Date"].unique().tolist())
    splits = get_chronological_period_splits(unique_dates, num_periods)

    results_list = []
    results_by_fund = {}

    for symbol, group in aligned_df.groupby("symbol"):
        group = group.sort_values("Date")
        fund_period_results = []

        for sp in splits:
            p_id = sp["period_id"]
            p_dates = sp["dates"]
            sub_df = group[group["Date"].isin(p_dates)].reset_index(drop=True)

            if len(sub_df) < 3:
                raise ValueError(f"Insufficient observations (n={len(sub_df)}) in {p_id} for {symbol}.")

            y = sub_df["excess_fund_return"].values
            x = sub_df["excess_market_return"].values

            capm_res, residuals = fit_capm_ols(y, x, symbol=f"{symbol}_{p_id}")

            p_res = PeriodCAPMResult(
                symbol=symbol,
                period_id=p_id,
                start_date=sp["start_date"],
                end_date=sp["end_date"],
                n_observations=len(sub_df),
                alpha_daily=capm_res.alpha_daily,
                alpha_annual_pct=capm_res.alpha_annual_pct,
                beta=capm_res.beta,
                beta_stderr=capm_res.beta_stderr,
                beta_tstat=capm_res.beta_tstat,
                beta_pvalue=capm_res.beta_pvalue,
                beta_ci_lower=capm_res.beta_ci95_lower,
                beta_ci_upper=capm_res.beta_ci95_upper,
                alpha_stderr=capm_res.alpha_stderr,
                alpha_tstat=capm_res.alpha_tstat,
                alpha_pvalue=capm_res.alpha_pvalue_hac,
                alpha_ci_lower=capm_res.alpha_daily_ci95_lower,
                alpha_ci_upper=capm_res.alpha_daily_ci95_upper,
                r_squared=capm_res.r_squared,
                adj_r_squared=capm_res.adj_r_squared,
                residual_std=capm_res.residual_std,
                methodology="Chronological Subperiod CAPM OLS Regression (Newey-West HAC alpha inference)",
                methodology_version="2.0.0",
                alpha_stderr_hac=capm_res.alpha_stderr_hac,
                alpha_pvalue_hac=capm_res.alpha_pvalue_hac,
            )

            results_list.append(p_res.to_dict())
            fund_period_results.append(p_res)

        results_by_fund[symbol] = fund_period_results

    period_df = pd.DataFrame(results_list)
    return period_df, results_by_fund


def calculate_persistence_metrics(
    period_df: pd.DataFrame,
    capm_full_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute fund-level persistence metrics across chronological subperiods.
    
    Returns:
    --------
    summary_df (1 row per fund)
    """
    summary_list = []

    capm_map = {}
    if capm_full_df is not None and not capm_full_df.empty:
        for _, row in capm_full_df.iterrows():
            capm_map[str(row["symbol"])] = {
                "alpha_daily": row["alpha_daily"],
                "alpha_annual_pct": row["alpha_annual_pct"]
            }

    period_df_copy = period_df.copy()
    period_df_copy["symbol"] = period_df_copy["symbol"].astype(str)

    for symbol, group in period_df_copy.groupby("symbol"):
        group = group.sort_values("period_id")
        total_p = len(group)

        daily_alphas = group["alpha_daily"].values
        annual_alphas = group["alpha_annual_pct"].values
        # Significance inside each subperiod uses the HAC p-value when available
        p_col = "alpha_pvalue_hac" if "alpha_pvalue_hac" in group.columns else "alpha_pvalue"
        p_values = group[p_col].values

        pos_count = int(np.sum(daily_alphas > 0))
        sig_pos_count = int(np.sum((daily_alphas > 0) & (p_values < 0.05)))

        pos_rate = float(pos_count / total_p) if total_p > 0 else 0.0
        sig_pos_rate = float(sig_pos_count / total_p) if total_p > 0 else 0.0

        mean_ann = float(np.mean(annual_alphas))
        median_ann = float(np.median(annual_alphas))
        std_ann = float(np.std(annual_alphas, ddof=1)) if total_p > 1 else 0.0
        min_ann = float(np.min(annual_alphas))
        max_ann = float(np.max(annual_alphas))

        # "Majority" means STRICTLY more than half of the subperiods; a 2-of-4 tie is a
        # coin-flip outcome and is not treated as persistence.
        pos_all = bool(pos_count == total_p)
        pos_maj = bool(pos_count > (total_p / 2.0))
        sig_pos_maj = bool(sig_pos_count > (total_p / 2.0))

        full_daily = capm_map.get(symbol, {}).get("alpha_daily", np.nan)
        full_annual = capm_map.get(symbol, {}).get("alpha_annual_pct", np.nan)

        res = PersistenceSummaryResult(
            symbol=symbol,
            total_periods=total_p,
            positive_alpha_periods=pos_count,
            significant_positive_alpha_periods=sig_pos_count,
            positive_alpha_persistence_rate=pos_rate,
            significant_positive_alpha_persistence_rate=sig_pos_rate,
            alpha_mean_annual_pct=mean_ann,
            alpha_median_annual_pct=median_ann,
            alpha_std_annual_pct=std_ann,
            alpha_min_annual_pct=min_ann,
            alpha_max_annual_pct=max_ann,
            alpha_positive_all_periods=pos_all,
            alpha_positive_majority=pos_maj,
            significant_positive_majority=sig_pos_maj,
            capm_full_sample_alpha_daily=full_daily,
            capm_full_sample_alpha_annual_pct=full_annual
        )

        summary_list.append(res.to_dict())

    return pd.DataFrame(summary_list)


def save_persistence_results(
    period_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str
) -> Tuple[str, str, str, str]:
    """
    Save persistence period-level and summary results to data/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    p_csv = os.path.join(output_dir, "persistence_period_results.csv")
    p_json = os.path.join(output_dir, "persistence_period_results.json")
    s_csv = os.path.join(output_dir, "persistence_summary.csv")
    s_json = os.path.join(output_dir, "persistence_summary.json")

    period_df.to_csv(p_csv, index=False)
    summary_df.to_csv(s_csv, index=False)

    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(period_df.to_dict(orient="records"), f, indent=2)

    with open(s_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print(f"[PERSISTENCE ENGINE] Exported persistence results:")
    print(f"  - Period CSV:    {p_csv}")
    print(f"  - Period JSON:   {p_json}")
    print(f"  - Summary CSV:   {s_csv}")
    print(f"  - Summary JSON:  {s_json}")

    return p_csv, p_json, s_csv, s_json


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    capm_csv_path = os.path.join(root, "data", "processed", "capm_results.csv")
    out_dir = os.path.join(root, "data", "processed")

    if os.path.exists(aligned_path):
        df_aligned = pd.read_csv(aligned_path, dtype={"symbol": str})
        capm_df = pd.read_csv(capm_csv_path, dtype={"symbol": str}) if os.path.exists(capm_csv_path) else None

        period_df, _ = run_period_capm_analysis(df_aligned, num_periods=4)
        summary_df = calculate_persistence_metrics(period_df, capm_df)
        save_persistence_results(period_df, summary_df, out_dir)
    else:
        print("Aligned dataset not found. Run Task 3 data pipeline first.")
