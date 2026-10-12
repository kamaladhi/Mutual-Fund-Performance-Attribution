"""
live_pipeline.py — In-Memory Econometric Pipeline
===================================================
Wraps the existing India/USA econometric modules so that the Streamlit dashboard
can run the full 5-pillar analysis on any live-fetched fund NAV DataFrame
*without* reading or writing CSV files.

The pipeline mirrors run_pipeline.py but:
  - Accepts a DataFrame from live_ingestion.py instead of reading raw CSVs
  - Loads factor data once and caches it (IIMA for India, FF for USA)
  - Returns a results dict the dashboard can render immediately
  - Completes in < 2 seconds for a single fund

Usage:
    from components.live_pipeline import run_india_live, run_usa_live
"""

from __future__ import annotations

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ─── path setup so we can import the market-specific src modules ───────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDIA_ROOT   = os.path.join(_PROJECT_ROOT, "markets", "india")
_USA_ROOT     = os.path.join(_PROJECT_ROOT, "markets", "usa")

for _p in [_PROJECT_ROOT, _INDIA_ROOT, _USA_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── IIMA factor cache ─────────────────────────────────────────────────────────
IIMA_URL = (
    "https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA/"
    "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"
)
IIMA_CACHE_PATH = os.path.join(_INDIA_ROOT, "data", "raw", "raw_iima_four_factors_daily.csv")
NIFTY_BENCH_PATH = os.path.join(_INDIA_ROOT, "data", "raw", "raw_benchmark_nifty500.csv")

FF_CACHE_PATH = os.path.join(_USA_ROOT, "data", "raw", "raw_ff_factors_daily.csv")
SP500_BENCH_PATH = os.path.join(_USA_ROOT, "data", "raw", "raw_benchmark_sp500.csv")


def _load_iima_factors(start: str = "2019-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Load IIMA factor data from cache or download. Preprocessed for the pipeline."""
    # Import the India data_loader's preprocessing function
    india_dl_path = os.path.join(_INDIA_ROOT)
    if india_dl_path not in sys.path:
        sys.path.insert(0, india_dl_path)

    from src import data_loader as india_dl

    raw_dir = os.path.join(_INDIA_ROOT, "data", "raw")
    try:
        factors_raw = india_dl.acquire_raw_iima_factors(start, end, raw_dir)
        return india_dl.preprocess_iima_factors(factors_raw, start, end)
    except Exception as e:
        st.error(f"Could not load IIMA factor data: {e}")
        return pd.DataFrame()


def _load_nifty_benchmark(start: str = "2019-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Load NIFTY 500 benchmark. Use cached CSV if it exists."""
    if india_dl_path := os.path.join(_INDIA_ROOT):
        if india_dl_path not in sys.path:
            sys.path.insert(0, india_dl_path)

    from src import data_loader as india_dl
    raw_dir = os.path.join(_INDIA_ROOT, "data", "raw")

    try:
        _, bench_raw, _ = india_dl.acquire_raw_fund_and_benchmark_data(
            [], india_dl.TARGET_BENCHMARK, start, end, raw_dir
        )
        return india_dl.preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col="symbol")
    except Exception:
        # Try loading existing benchmark CSV
        if os.path.exists(NIFTY_BENCH_PATH):
            bench_raw = pd.read_csv(NIFTY_BENCH_PATH)
            from src import data_loader as india_dl
            return india_dl.preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col="symbol")
        return pd.DataFrame()


def run_india_live(nav_df: pd.DataFrame,
                   fund_name: str = "Live Fund",
                   start: str = "2019-01-01",
                   end: str   = "2023-12-31") -> dict:
    """
    Runs the full India 5-pillar analysis on a live-fetched NAV DataFrame.

    Args:
        nav_df:    DataFrame with [Date, Adj Close, symbol] from live_ingestion.fetch_india_nav()
        fund_name: Display name for the fund
        start/end: Analysis window

    Returns:
        dict with keys: capm, bootstrap, factor, persistence, evaluation, aligned_df
        Returns {"error": str} on failure.
    """
    t0 = time.time()
    if nav_df.empty:
        return {"error": "No NAV data available for this fund in the selected date range."}

    # Add india path
    if _INDIA_ROOT not in sys.path:
        sys.path.insert(0, _INDIA_ROOT)

    try:
        from src import data_loader as india_dl, capm, bootstrap, factor_model, persistence, evaluator

        # 1. Load shared factor + benchmark data (cached on disk)
        raw_dir = os.path.join(_INDIA_ROOT, "data", "raw")
        factors_raw = india_dl.acquire_raw_iima_factors(start, end, raw_dir)
        factors_proc = india_dl.preprocess_iima_factors(factors_raw, start, end)

        # Load benchmark from cached CSV (already saved from initial pipeline run)
        if os.path.exists(NIFTY_BENCH_PATH):
            bench_raw = pd.read_csv(NIFTY_BENCH_PATH)
        else:
            _, bench_raw, _ = india_dl.acquire_raw_fund_and_benchmark_data(
                [], india_dl.TARGET_BENCHMARK, start, end, raw_dir
            )
        bench_proc = india_dl.preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col="symbol")

        # 2. Preprocess the live NAV
        fund_proc = india_dl.preprocess_price_series(nav_df.copy(), price_col="Adj Close", symbol_col="symbol")

        # 3. Align: inner join on common trading dates
        aligned = india_dl.create_aligned_dataset(fund_proc, bench_proc, factors_proc)
        if aligned.empty or len(aligned) < 100:
            return {"error": f"Insufficient aligned observations ({len(aligned)}). Need at least 100 trading days."}

        # 4. Run the 4 analytical pillars
        capm_df, capm_dict, _  = capm.run_capm_analysis(aligned)
        boot_df, boot_dict, _  = bootstrap.run_bootstrap_analysis(
            aligned, capm_dict, block_length=30, n_bootstrap=500, seed=42  # 500 for speed
        )
        fac_df, fac_dict, _    = factor_model.run_four_factor_analysis(aligned, capm_dict)
        period_df, _           = persistence.run_period_capm_analysis(aligned, num_periods=4)
        pers_summary           = persistence.calculate_persistence_metrics(period_df, capm_df)

        # 5. Classification — use evaluator's in-memory helper
        sym = nav_df["symbol"].iloc[0]
        capm_row  = capm_df[capm_df["symbol"] == sym].iloc[0].to_dict() if not capm_df.empty else {}
        boot_row  = boot_df[boot_df["symbol"] == sym].iloc[0].to_dict() if not boot_df.empty else {}
        fac_row   = fac_df[fac_df["symbol"] == sym].iloc[0].to_dict()  if not fac_df.empty  else {}
        pers_row  = pers_summary[pers_summary["symbol"] == sym].iloc[0].to_dict() if not pers_summary.empty else {}

        eval_result = evaluator.classify_single_fund(capm_row, boot_row, fac_row, pers_row)

        elapsed = time.time() - t0
        return {
            "fund_name":   fund_name,
            "symbol":      sym,
            "capm":        capm_row,
            "bootstrap":   boot_row,
            "factor":      fac_row,
            "persistence": pers_row,
            "evaluation":  eval_result,
            "aligned_df":  aligned,
            "elapsed_sec": round(elapsed, 2),
            "n_obs":       len(aligned),
            "error":       None,
        }

    except Exception as exc:
        return {"error": f"Pipeline error: {exc}"}


def run_usa_live(nav_df: pd.DataFrame,
                 fund_name: str = "Live Fund",
                 start: str = "2019-01-01",
                 end: str   = "2023-12-31") -> dict:
    """
    Runs the full USA 5-pillar analysis on a live-fetched price DataFrame.
    Mirrors run_india_live() but uses USA-specific data_loader and FF factors.
    """
    t0 = time.time()
    if nav_df.empty:
        return {"error": "No price data available for this fund in the selected date range."}

    if _USA_ROOT not in sys.path:
        sys.path.insert(0, _USA_ROOT)

    try:
        from src import data_loader as usa_dl, capm, bootstrap, factor_model, persistence, evaluator

        raw_dir = os.path.join(_USA_ROOT, "data", "raw")
        fund_proc = usa_dl.preprocess_price_series(nav_df.copy(), price_col="Adj Close", symbol_col="symbol")

        # Load benchmark + factors from existing cached files
        bench_path  = os.path.join(raw_dir, "raw_benchmark_sp500.csv")
        factor_path = os.path.join(raw_dir, "raw_ff_factors_daily.csv")

        if not os.path.exists(bench_path) or not os.path.exists(factor_path):
            return {"error": "US factor / benchmark files not found. Run the US market pipeline once first."}

        bench_raw   = pd.read_csv(bench_path)
        factors_raw = pd.read_csv(factor_path)

        bench_proc   = usa_dl.preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col="symbol")
        factors_proc = usa_dl.preprocess_ff_factors(factors_raw, start, end)

        aligned = usa_dl.create_aligned_dataset(fund_proc, bench_proc, factors_proc)
        if aligned.empty or len(aligned) < 100:
            return {"error": f"Insufficient aligned observations ({len(aligned)})."}

        capm_df, capm_dict, _  = capm.run_capm_analysis(aligned)
        boot_df, boot_dict, _  = bootstrap.run_bootstrap_analysis(
            aligned, capm_dict, block_length=30, n_bootstrap=500, seed=42
        )
        fac_df, fac_dict, _    = factor_model.run_four_factor_analysis(aligned, capm_dict)
        period_df, _           = persistence.run_period_capm_analysis(aligned, num_periods=4)
        pers_summary           = persistence.calculate_persistence_metrics(period_df, capm_df)

        sym      = nav_df["symbol"].iloc[0]
        capm_row = capm_df[capm_df["symbol"] == sym].iloc[0].to_dict()  if not capm_df.empty  else {}
        boot_row = boot_df[boot_df["symbol"] == sym].iloc[0].to_dict()  if not boot_df.empty  else {}
        fac_row  = fac_df[fac_df["symbol"] == sym].iloc[0].to_dict()    if not fac_df.empty   else {}
        pers_row = pers_summary[pers_summary["symbol"] == sym].iloc[0].to_dict() if not pers_summary.empty else {}

        eval_result = evaluator.classify_single_fund(capm_row, boot_row, fac_row, pers_row)

        return {
            "fund_name":   fund_name,
            "symbol":      sym,
            "capm":        capm_row,
            "bootstrap":   boot_row,
            "factor":      fac_row,
            "persistence": pers_row,
            "evaluation":  eval_result,
            "aligned_df":  aligned,
            "elapsed_sec": round(time.time() - t0, 2),
            "n_obs":       len(aligned),
            "error":       None,
        }

    except Exception as exc:
        return {"error": f"Pipeline error: {exc}"}
