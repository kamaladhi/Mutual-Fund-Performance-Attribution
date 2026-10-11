"""
Benchmark (market-proxy) sensitivity analysis
Indian Fund Manager Skill Detector

The skill classification depends materially on which market portfolio the CAPM alpha is measured
against.  This module re-runs the complete analytical stack (CAPM -> bootstrap -> 4-factor ->
persistence -> evaluator) under several market proxies and tabulates the outcome, so that readers
can see which conclusions are robust and which are artefacts of the benchmark choice.

Proxies
-------
  iima_market      IIMA value-weighted market portfolio, total return (Mkt-RF + RF)   [primary]
  nifty500_price   NIFTY 500 PRICE index (^CRSLDX; excludes dividends -> alpha biased UP)
  nifty500_price_plus_assumed_dividend_yield
                   price index + ASSUMED_NIFTY500_DIVIDEND_YIELD per annum: a rough stand-in for the
                   NIFTY 500 TRI. The yield is an ASSUMPTION (not measured from data); replace by a
                   genuine TRI file (data/raw/raw_benchmark_nifty500_tri.csv) when available.
  nifty500_tri     genuine TRI, only if supplied.
"""

import os
import tempfile
from typing import Callable, Dict, Optional

import pandas as pd

from src import capm, bootstrap, factor_model, persistence, evaluator

TRADING_DAYS = 252
ASSUMED_NIFTY500_DIVIDEND_YIELD = 0.013   # ~1.3% p.a. -- assumption, flagged as such everywhere it is used

OUTPUT_COLUMNS = [
    "market_proxy", "symbol", "capm_alpha_annual_pct", "capm_alpha_pvalue", "bootstrap_p_value",
    "factor_alpha_annual_pct", "factor_alpha_pvalue", "positive_alpha_persistence_rate",
    "total_score", "final_category",
]


def _proxies(aligned: pd.DataFrame) -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    proxies: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "iima_market": lambda d: d["iima_market_return"],
        "nifty500_price": lambda d: d["nifty500_price_return"],
        "nifty500_price_plus_assumed_dividend_yield":
            lambda d: d["nifty500_price_return"] + ASSUMED_NIFTY500_DIVIDEND_YIELD / TRADING_DAYS,
    }
    if "nifty500_tri_return" in aligned.columns:
        proxies["nifty500_tri"] = lambda d: d["nifty500_tri_return"]
    return proxies


def _evaluate_with_proxy(aligned: pd.DataFrame, market_return: pd.Series, n_bootstrap: int,
                         block_length: int, seed: Optional[int]) -> pd.DataFrame:
    df = aligned.copy()
    df["market_return"] = market_return
    df["excess_market_return"] = df["market_return"] - df["RF"]

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data", "processed")
        os.makedirs(out)
        capm_df, capm_dict, _ = capm.run_capm_analysis(df)
        capm.save_capm_results(capm_df, capm_dict, out)
        boot_df, boot_dict, _ = bootstrap.run_bootstrap_analysis(
            df, capm_dict, block_length=block_length, n_bootstrap=n_bootstrap, seed=seed)
        bootstrap.save_bootstrap_results(boot_df, boot_dict, out)
        fac_df, fac_dict, _ = factor_model.run_four_factor_analysis(df, capm_dict)
        factor_model.save_factor_model_results(fac_df, fac_dict, out)
        period_df, _ = persistence.run_period_capm_analysis(df, num_periods=4)
        summary = persistence.calculate_persistence_metrics(period_df, capm_df)
        persistence.save_persistence_results(period_df, summary, out)
        return evaluator.run_evaluation(tmp)


def run_benchmark_sensitivity(aligned: pd.DataFrame, n_bootstrap: int = 1000, block_length: int = 30,
                              seed: Optional[int] = 42) -> pd.DataFrame:
    """Long table: one row per (market proxy, fund) with the headline statistics and category."""
    aligned = aligned.copy()
    aligned["symbol"] = aligned["symbol"].astype(str)
    frames = []
    import contextlib
    import io
    for name, make in _proxies(aligned).items():
        with contextlib.redirect_stdout(io.StringIO()):          # silence the engines' export banners
            res = _evaluate_with_proxy(aligned, make(aligned), n_bootstrap, block_length, seed)
        res = res.copy()
        res["market_proxy"] = name
        frames.append(res[OUTPUT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def save_benchmark_sensitivity(table: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_sensitivity.csv")
    table.to_csv(path, index=False)
    print(f"[SENSITIVITY] Exported benchmark-proxy sensitivity ({table['market_proxy'].nunique()} proxies) to {path}")
    return path


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    if os.path.exists(aligned_path):
        data = pd.read_csv(aligned_path, dtype={"symbol": str})
        save_benchmark_sensitivity(run_benchmark_sensitivity(data), os.path.join(root, "data", "processed"))
    else:
        print("Aligned dataset not found. Run the data pipeline first.")
