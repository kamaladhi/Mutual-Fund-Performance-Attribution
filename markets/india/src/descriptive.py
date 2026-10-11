"""
Descriptive Return & Risk Statistics (FR-STA-01)
Indian Fund Manager Skill Detector

Annualised return (CAGR and arithmetic), annualised volatility, Sharpe ratio, maximum
drawdown and distribution moments for every fund and for the market proxies present in
the aligned dataset. Pure descriptive statistics: nothing here feeds the skill classifier.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

# Market-proxy columns that may exist in the aligned dataset: column -> label
MARKET_SERIES = {
    "market_return": "Primary market proxy (used in CAPM)",
    "iima_market_return": "IIMA market portfolio (total return)",
    "nifty500_price_return": "NIFTY 500 price index (excludes dividends)",
}


def summarize_return_series(
    returns: np.ndarray,
    rf: Optional[np.ndarray] = None,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Return/risk summary of one daily simple-return series (aligned optional daily RF)."""
    r = np.asarray(returns, dtype=np.float64)
    mask = ~np.isnan(r)
    r = r[mask]
    n = len(r)
    if n < 2:
        raise ValueError("Need at least two observations for descriptive statistics.")

    growth = np.prod(1.0 + r)
    cagr = growth ** (trading_days / n) - 1.0
    arith = float(np.mean(r) * trading_days)
    vol = float(np.std(r, ddof=1) * np.sqrt(trading_days))

    if rf is not None:
        excess = r - np.asarray(rf, dtype=np.float64)[mask]
    else:
        excess = r
    ex_vol = float(np.std(excess, ddof=1) * np.sqrt(trading_days))
    # A constant series has float-noise (not exactly zero) dispersion: treat it as degenerate.
    degenerate_excess = float(np.ptp(excess)) < 1e-14
    sharpe = float("nan") if degenerate_excess or ex_vol <= 0 else float(np.mean(excess) * trading_days / ex_vol)
    degenerate_returns = float(np.ptp(r)) < 1e-14

    wealth = np.cumprod(1.0 + r)
    running_peak = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / running_peak - 1.0))

    return {
        "n_observations": int(n),
        "cumulative_return_pct": float((growth - 1.0) * 100.0),
        "annualized_return_cagr_pct": float(cagr * 100.0),
        "annualized_return_arithmetic_pct": arith * 100.0,
        "annualized_volatility_pct": vol * 100.0,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "skewness": float("nan") if degenerate_returns else float(stats.skew(r)),
        "excess_kurtosis": float("nan") if degenerate_returns else float(stats.kurtosis(r)),
    }


def _level_returns(levels: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Daily simple returns of a price/NAV level series on its own trading calendar."""
    df = levels.sort_values("Date").drop_duplicates("Date").copy()
    df["ret"] = df[price_col].pct_change()
    return df.dropna(subset=["ret"])[["Date", "ret"]]


def run_buy_and_hold_statistics(raw_dir: str) -> pd.DataFrame:
    """
    Return/risk statistics from the FULL raw series (every trading day of each series), i.e. the
    actual buy-and-hold record: cumulative return equals last NAV / first NAV - 1 exactly.

    The regression sample drops ~1% of days to keep fund, benchmark and factor returns on the same
    interval; statistics computed on it would understate performance, so reporting uses this basis.
    The daily risk-free rate is the IIMA series matched on date (missing dates take the window mean).
    """
    funds = pd.read_csv(os.path.join(raw_dir, "raw_fund_prices.csv"), dtype={"symbol": str})
    bench = pd.read_csv(os.path.join(raw_dir, "raw_benchmark_nifty500.csv"))
    iima = pd.read_csv(os.path.join(raw_dir, "raw_iima_four_factors_daily.csv"))
    iima["Date"] = pd.to_datetime(iima["Date"]).dt.strftime("%Y-%m-%d")
    for col in ("MF", "RF"):
        iima[col] = pd.to_numeric(iima[col], errors="coerce") / 100.0

    base_date, last_date = funds["Date"].min(), funds["Date"].max()   # first NAV is the base, not a return
    window = iima[(iima["Date"] > base_date) & (iima["Date"] <= last_date)].dropna(subset=["MF", "RF"]).copy()
    rf_by_date = window.set_index("Date")["RF"]
    mean_rf = float(rf_by_date.mean())

    def rf_for(dates: pd.Series) -> np.ndarray:
        return dates.map(rf_by_date).fillna(mean_rf).values

    rows: List[Dict[str, object]] = []
    for sym, grp in funds.groupby("symbol"):
        rets = _level_returns(grp)
        rows.append({"series": str(sym), "kind": "fund", "basis": "buy_and_hold_all_trading_days",
                     **summarize_return_series(rets["ret"].values, rf_for(rets["Date"]))})

    rows.append({"series": "iima_market_return", "kind": "market", "basis": "buy_and_hold_all_trading_days",
                 "description": MARKET_SERIES["iima_market_return"],
                 **summarize_return_series((window["MF"] + window["RF"]).values, window["RF"].values)})
    bench_rets = _level_returns(bench)
    bench_rets = bench_rets[(bench_rets["Date"] > base_date) & (bench_rets["Date"] <= last_date)]
    rows.append({"series": "nifty500_price_return", "kind": "market", "basis": "buy_and_hold_all_trading_days",
                 "description": MARKET_SERIES["nifty500_price_return"],
                 **summarize_return_series(bench_rets["ret"].values, rf_for(bench_rets["Date"]))})
    return pd.DataFrame(rows)


def run_descriptive_statistics(aligned_df: pd.DataFrame, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """
    One row per fund plus one row per market proxy.

    With ``raw_dir`` the statistics describe the full buy-and-hold record (recommended for reporting).
    Without it they are computed on the aligned regression sample (``basis`` = regression_sample).
    """
    if raw_dir is not None:
        return run_buy_and_hold_statistics(raw_dir)

    required = {"symbol", "Date", "daily_return", "RF"}
    missing = required - set(aligned_df.columns)
    if missing:
        raise KeyError(f"Aligned dataset missing required columns: {sorted(missing)}")

    df = aligned_df.copy()
    df["symbol"] = df["symbol"].astype(str)
    rows: List[Dict[str, object]] = []

    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("Date")
        stats_row = summarize_return_series(grp["daily_return"].values, grp["RF"].values)
        rows.append({"series": sym, "kind": "fund", "basis": "regression_sample", **stats_row})

    # Market proxies: one observation per date (identical across funds)
    by_date = df.sort_values("Date").drop_duplicates("Date")
    for col, label in MARKET_SERIES.items():
        if col in by_date.columns and by_date[col].notna().any():
            stats_row = summarize_return_series(by_date[col].values, by_date["RF"].values)
            rows.append({"series": col, "kind": "market", "basis": "regression_sample", "description": label, **stats_row})

    return pd.DataFrame(rows)


def save_descriptive_statistics(summary_df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "descriptive_statistics.csv")
    summary_df.to_csv(path, index=False)
    print(f"[DESCRIPTIVE] Exported return/risk statistics ({len(summary_df)} series) to {path}")
    return path


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = os.path.join(root, "data", "processed", "aligned_analysis_dataset.csv")
    if os.path.exists(aligned_path):
        aligned = pd.read_csv(aligned_path, dtype={"symbol": str})
        save_descriptive_statistics(run_descriptive_statistics(aligned, raw_dir=os.path.join(root, "data", "raw")),
                                    os.path.join(root, "data", "processed"))
    else:
        print("Aligned dataset not found. Run the data pipeline first.")
