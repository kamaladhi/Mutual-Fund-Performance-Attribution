"""
Data Loader & Preprocessing Pipeline Module
Fund Manager Skill Detector

Handles raw data acquisition, raw storage protection, metadata tracking,
data validation, return calculation, factor scaling, and common-date alignment.
"""

import os
import json
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import requests
try:
    import yfinance as yf
except ImportError:  # allows offline analysis/tests using saved CSVs
    class _YFinanceUnavailable:
        @staticmethod
        def download(*args, **kwargs):
            raise ImportError("yfinance is required for live data downloads. Install dependencies from requirements.txt.")
    yf = _YFinanceUnavailable()

# Target candidate funds and benchmark symbols
TARGET_FUNDS = [
    "AGTHX", "FCNTX", "DODGX", "FDGRX", "TRBCX", "AIVSX",
    "FLPSX", "AMECX", "PRHSX", "PAGRX", "FMAGX", "MIGFX"
]
TARGET_BENCHMARK = "^GSPC"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2023-12-31"

FRENCH_FF3_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FRENCH_MOM_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


# =====================================================================
# 1. RAW DATA ACQUISITION & METADATA MANAGEMENT
# =====================================================================

def acquire_raw_fund_and_benchmark_data(
    symbols: List[str],
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
    raw_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Acquire raw daily price series for funds and benchmark from Yahoo Finance.
    Preserves existing files under data/raw/ without overwriting.
    """
    os.makedirs(raw_dir, exist_ok=True)
    funds_filepath = os.path.join(raw_dir, "raw_fund_prices.csv")
    bench_filepath = os.path.join(raw_dir, "raw_benchmark_sp500.csv")
    meta_filepath = os.path.join(raw_dir, "acquisition_metadata.json")

    # Load existing metadata if present
    metadata = {}
    if os.path.exists(meta_filepath):
        try:
            with open(meta_filepath, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # Fund Data Ingestion
    if os.path.exists(funds_filepath):
        print(f"[DATA LOADER] Loading existing raw fund data from {funds_filepath}")
        funds_df = pd.read_csv(funds_filepath)
    else:
        print(f"[DATA LOADER] Downloading raw fund data for {len(symbols)} tickers...")
        all_funds = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            df = df.reset_index()
            df["symbol"] = symbol
            all_funds.append(df)
        
        funds_df = pd.concat(all_funds, ignore_index=True)
        if "Date" in funds_df.columns:
            funds_df["Date"] = pd.to_datetime(funds_df["Date"]).dt.strftime("%Y-%m-%d")
        funds_df.to_csv(funds_filepath, index=False)
        
        metadata["raw_fund_prices.csv"] = {
            "source": "Yahoo Finance (yfinance)",
            "retrieval_datetime": datetime.now().isoformat(),
            "requested_date_range": [start_date, end_date],
            "actual_date_range": [funds_df["Date"].min(), funds_df["Date"].max()],
            "tickers": symbols,
            "status": "DOWNLOADED_AND_SAVED"
        }

    # Benchmark Data Ingestion
    if os.path.exists(bench_filepath):
        print(f"[DATA LOADER] Loading existing raw benchmark data from {bench_filepath}")
        bench_df = pd.read_csv(bench_filepath)
    else:
        print(f"[DATA LOADER] Downloading raw benchmark data ({benchmark_symbol})...")
        ticker = yf.Ticker(benchmark_symbol)
        bench_df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if bench_df.empty:
            bench_df = yf.download(benchmark_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        bench_df = bench_df.reset_index()
        bench_df["symbol"] = benchmark_symbol
        if "Date" in bench_df.columns:
            bench_df["Date"] = pd.to_datetime(bench_df["Date"]).dt.strftime("%Y-%m-%d")
        bench_df.to_csv(bench_filepath, index=False)

        metadata["raw_benchmark_sp500.csv"] = {
            "source": "Yahoo Finance (yfinance)",
            "retrieval_datetime": datetime.now().isoformat(),
            "requested_date_range": [start_date, end_date],
            "actual_date_range": [bench_df["Date"].min(), bench_df["Date"].max()],
            "ticker": benchmark_symbol,
            "status": "DOWNLOADED_AND_SAVED"
        }

    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return funds_df, bench_df, metadata


def acquire_raw_fama_french_factors(
    start_date: str,
    end_date: str,
    raw_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Acquire daily Fama-French 3-Factor and Momentum datasets from Kenneth French Data Library.
    Preserves existing files under data/raw/ without overwriting.
    """
    os.makedirs(raw_dir, exist_ok=True)
    ff_filepath = os.path.join(raw_dir, "raw_fama_french_daily.csv")
    mom_filepath = os.path.join(raw_dir, "raw_momentum_daily.csv")

    # FF3 Ingestion
    if os.path.exists(ff_filepath):
        print(f"[DATA LOADER] Loading existing raw Fama-French 3-Factor data from {ff_filepath}")
        ff_df = pd.read_csv(ff_filepath)
    else:
        print("[DATA LOADER] Downloading raw Fama-French 3-Factor daily data...")
        response = requests.get(FRENCH_FF3_DAILY_URL, timeout=15)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                lines = [line.decode("utf-8", errors="ignore") for line in f.readlines()]
        
        header_idx = -1
        for i, line in enumerate(lines):
            if "Mkt-RF" in line:
                header_idx = i
                break
        
        if header_idx == -1:
            raise ValueError("Failed to locate header in Fama-French 3-Factor zip file.")
        
        csv_data = "".join(lines[header_idx:])
        ff_df = pd.read_csv(io.StringIO(csv_data))
        first_col = ff_df.columns[0]
        ff_df = ff_df.rename(columns={first_col: "Date"})
        ff_df = ff_df.dropna(subset=["Date"])
        ff_df["Date"] = ff_df["Date"].astype(str).str.strip()
        ff_df = ff_df[ff_df["Date"].str.match(r"^\d{8}$")]
        ff_df["Date"] = pd.to_datetime(ff_df["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        
        ff_df = ff_df[(ff_df["Date"] >= start_date) & (ff_df["Date"] <= end_date)].reset_index(drop=True)
        ff_df.to_csv(ff_filepath, index=False)

    # Momentum Ingestion
    if os.path.exists(mom_filepath):
        print(f"[DATA LOADER] Loading existing raw Momentum factor data from {mom_filepath}")
        mom_df = pd.read_csv(mom_filepath)
    else:
        print("[DATA LOADER] Downloading raw Momentum daily factor data...")
        response = requests.get(FRENCH_MOM_DAILY_URL, timeout=15)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                lines = [line.decode("utf-8", errors="ignore") for line in f.readlines()]

        header_idx = -1
        for i, line in enumerate(lines):
            if "Mom" in line or "UMD" in line:
                header_idx = i
                break

        if header_idx == -1:
            raise ValueError("Failed to locate header in Momentum factor zip file.")

        csv_data = "".join(lines[header_idx:])
        mom_df = pd.read_csv(io.StringIO(csv_data))
        first_col = mom_df.columns[0]
        mom_df = mom_df.rename(columns={first_col: "Date"})
        mom_df.columns = [c.strip() for c in mom_df.columns]
        mom_df = mom_df.dropna(subset=["Date"])
        mom_df["Date"] = mom_df["Date"].astype(str).str.strip()
        mom_df = mom_df[mom_df["Date"].str.match(r"^\d{8}$")]
        mom_df["Date"] = pd.to_datetime(mom_df["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        mom_df = mom_df[(mom_df["Date"] >= start_date) & (mom_df["Date"] <= end_date)].reset_index(drop=True)
        mom_df.to_csv(mom_filepath, index=False)

    return ff_df, mom_df


def initialize_screening_metadata(raw_dir: str, candidate_tickers: List[str]) -> str:
    """
    Initialize data/raw/screening_metadata.json for the candidate fund universe.
    Preserves existing metadata file if present.
    """
    os.makedirs(raw_dir, exist_ok=True)
    screening_path = os.path.join(raw_dir, "screening_metadata.json")

    fund_names = {
        "AGTHX": "American Funds Growth Fund of America",
        "FCNTX": "Fidelity Contrafund",
        "DODGX": "Dodge & Cox Stock Fund",
        "FDGRX": "Fidelity Growth Company Fund",
        "TRBCX": "T. Rowe Price Blue Chip Growth Fund",
        "AIVSX": "Investment Company of America",
        "FLPSX": "Fidelity Low-Priced Stock Fund",
        "AMECX": "Income Fund of America",
        "PRHSX": "T. Rowe Price Health Sciences Fund",
        "PAGRX": "Permanent Portfolio Aggressive Growth Fund",
        "FMAGX": "Fidelity Magellan Fund",
        "MIGFX": "MFS Massachusetts Investors Growth Stock Fund"
    }

    if os.path.exists(screening_path):
        print(f"[DATA LOADER] Screening metadata file already exists at {screening_path}")
        return screening_path

    records = []
    comparability = {
        "AMECX": ("EXPLORATORY_STYLE_MISMATCH", "Income-oriented equity strategy; not a clean peer for a broad growth-equity universe."),
        "PRHSX": ("EXPLORATORY_SECTOR_FUND", "Healthcare-sector fund; sector concentration is not captured by the broad four-factor model."),
        "PAGRX": ("EXPLORATORY_NOT_BROAD_EQUITY_COMPARABLE", "Permanent Portfolio Aggressive Growth is a multi-asset strategy. T. Rowe Price Growth Stock is PRGFX and requires a separate data acquisition."),
    }
    for ticker in candidate_tickers:
        record = {
            "ticker": ticker,
            "fund_name": fund_names.get(ticker, f"Mutual Fund {ticker}"),
            "status": "CANDIDATE",
            "active_mandate_evidence": None,
            "expense_ratio_evidence": None,
            "active_share_or_differentiation_evidence": None,
            "verification_status": "UNVERIFIED_CANDIDATE",
            "source_reference_info": f"Candidate ticker {ticker} pending evidence collection in Task 3/4."
        }
        if ticker in comparability:
            record["universe_status"], record["comparability_note"] = comparability[ticker]
        records.append(record)

    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[DATA LOADER] Initialized candidate screening metadata at {screening_path}")
    return screening_path


# =====================================================================
# 2. DATA VALIDATION
# =====================================================================

def validate_dataframe(
    df: pd.DataFrame,
    df_name: str,
    required_cols: List[str],
    date_col: str = "Date",
    id_col: Optional[str] = None,
    price_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive validation checks on a dataframe:
    - Required columns
    - Date parsing & ordering
    - Duplicate dates
    - Missing observation counts
    - Non-positive price check
    """
    report = {
        "dataset_name": df_name,
        "total_rows": len(df),
        "columns_present": list(df.columns),
        "is_valid": True,
        "validation_errors": [],
        "validation_warnings": []
    }

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["validation_errors"].append(f"Missing required columns: {missing_cols}")

    if date_col not in df.columns:
        report["is_valid"] = False
        report["validation_errors"].append(f"Date column '{date_col}' not found.")
        return report

    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
    invalid_dates = df_temp[date_col].isna().sum()
    if invalid_dates > 0:
        report["is_valid"] = False
        report["validation_errors"].append(f"Found {invalid_dates} unparseable date values.")

    if id_col and id_col in df.columns:
        dup_count = df_temp.duplicated(subset=[date_col, id_col]).sum()
    else:
        dup_count = df_temp.duplicated(subset=[date_col]).sum()

    report["duplicate_dates_count"] = int(dup_count)
    if dup_count > 0:
        report["validation_warnings"].append(f"Found {dup_count} duplicate date entries.")

    missing_counts = df.isna().sum().to_dict()
    report["missing_value_counts"] = {k: int(v) for k, v in missing_counts.items()}

    if price_cols:
        for pcol in price_cols:
            if pcol in df.columns:
                non_pos = (df[pcol] <= 0).sum()
                if non_pos > 0:
                    report["is_valid"] = False
                    report["validation_errors"].append(f"Found {non_pos} non-positive values in '{pcol}'.")

    if not df_temp.empty and date_col in df_temp.columns:
        report["min_date"] = str(df_temp[date_col].min().strftime("%Y-%m-%d"))
        report["max_date"] = str(df_temp[date_col].max().strftime("%Y-%m-%d"))
        
        if id_col and id_col in df_temp.columns:
            is_sorted = all(grp[date_col].is_monotonic_increasing for _, grp in df_temp.groupby(id_col))
        else:
            is_sorted = bool(df_temp[date_col].is_monotonic_increasing)
            
        report["is_chronologically_sorted"] = is_sorted
        if not is_sorted:
            report["validation_warnings"].append("Data is not chronologically sorted.")

    return report


# =====================================================================
# 3. PREPROCESSING & RETURN COMPUTATION
# =====================================================================

def preprocess_price_series(df: pd.DataFrame, price_col: str = "Adj Close", symbol_col: Optional[str] = "symbol") -> pd.DataFrame:
    """
    Clean price series, sort chronologically, and compute daily returns:
    R_t = (Price_t - Price_{t-1}) / Price_{t-1}
    Preserves raw source price columns.
    """
    df_clean = df.copy()
    if "Adj Close" not in df_clean.columns and "Adj_Close" in df_clean.columns:
        df_clean = df_clean.rename(columns={"Adj_Close": "Adj Close"})
    
    df_clean["Date"] = pd.to_datetime(df_clean["Date"]).dt.strftime("%Y-%m-%d")

    processed_dfs = []
    groups = df_clean.groupby(symbol_col) if symbol_col and symbol_col in df_clean.columns else [(None, df_clean)]

    for sym, group in groups:
        group = group.sort_values("Date").reset_index(drop=True)
        group = group.drop_duplicates(subset=["Date"]).reset_index(drop=True)
        
        if price_col in group.columns:
            group["daily_return"] = group[price_col].pct_change()
        else:
            raise KeyError(f"Price column '{price_col}' not found in dataset.")

        processed_dfs.append(group)

    result_df = pd.concat(processed_dfs, ignore_index=True)
    return result_df


def preprocess_fama_french_factors(ff3_df: pd.DataFrame, mom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fama-French and Momentum daily datasets.
    Converts percentage values to decimal returns (val / 100).
    Preserves raw factor values before transformation.
    """
    ff_clean = ff3_df.copy()
    mom_clean = mom_df.copy()

    ff_clean["Date"] = pd.to_datetime(ff_clean["Date"]).dt.strftime("%Y-%m-%d")
    mom_clean["Date"] = pd.to_datetime(mom_clean["Date"]).dt.strftime("%Y-%m-%d")

    ff_clean.columns = [c.strip() for c in ff_clean.columns]
    mom_clean.columns = [c.strip() for c in mom_clean.columns]

    factors_df = pd.merge(ff_clean, mom_clean, on="Date", how="inner")
    factors_df = factors_df.sort_values("Date").reset_index(drop=True)

    rename_dict = {}
    for col in factors_df.columns:
        if col.upper() in ["MKT-RF", "MKTRF"]:
            rename_dict[col] = "Mkt-RF"
        elif col.upper() == "SMB":
            rename_dict[col] = "SMB"
        elif col.upper() == "HML":
            rename_dict[col] = "HML"
        elif col.upper() == "RF":
            rename_dict[col] = "RF"
        elif col.upper() in ["MOM", "UMD"]:
            rename_dict[col] = "Mom"

    factors_df = factors_df.rename(columns=rename_dict)

    for col in ["Mkt-RF", "SMB", "HML", "RF", "Mom"]:
        if col in factors_df.columns:
            factors_df[f"raw_{col}"] = pd.to_numeric(factors_df[col], errors="coerce")
            factors_df[col] = factors_df[f"raw_{col}"] / 100.0

    return factors_df


# =====================================================================
# 4. STRICT COMMON-DATE ALIGNMENT
# =====================================================================

def create_aligned_dataset(
    processed_funds_df: pd.DataFrame,
    processed_bench_df: pd.DataFrame,
    processed_factors_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform strict common-date inner join across fund returns, S&P 500 descriptive benchmark,
    and Fama-French multi-factor series. CAPM uses the same Mkt-RF market excess return as the
    four-factor model to avoid inconsistent market definitions.
    Does NOT fabricate dates or perform ffill/bfill.
    """
    bench_sub = processed_bench_df[["Date", "daily_return", "Adj Close"]].copy()
    bench_sub = bench_sub.rename(columns={
        "daily_return": "market_return",
        "Adj Close": "market_adj_close"
    })

    factors_sub = processed_factors_df[["Date", "Mkt-RF", "SMB", "HML", "RF", "Mom"]].copy()

    merged_df = pd.merge(processed_funds_df, bench_sub, on="Date", how="inner")
    merged_df = pd.merge(merged_df, factors_sub, on="Date", how="inner")

    aligned_df = merged_df.dropna(subset=["daily_return", "market_return", "RF", "Mkt-RF"]).reset_index(drop=True)

    aligned_df["excess_fund_return"] = aligned_df["daily_return"] - aligned_df["RF"]
    # Use the Fama-French market excess return for CAPM as well as the
    # four-factor model. This keeps both alpha estimates on the same market
    # definition; the S&P 500 price-index return remains a descriptive proxy.
    aligned_df["excess_market_return"] = aligned_df["Mkt-RF"]

    return aligned_df


# =====================================================================
# PIPELINE EXECUTION FUNCTION
# =====================================================================

def run_data_pipeline(
    project_root: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> Dict[str, Any]:
    """
    Execute complete end-to-end Task 3 Data Acquisition and Preprocessing Pipeline.
    Stores raw files in project_root/data/raw/ and processed in project_root/data/processed/.
    """
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=====================================================================")
    print("STARTING DATA PIPELINE EXECUTION (TASK 3)")
    print("=====================================================================")

    screening_path = initialize_screening_metadata(raw_dir, TARGET_FUNDS)

    funds_raw, bench_raw, acq_meta = acquire_raw_fund_and_benchmark_data(
        TARGET_FUNDS, TARGET_BENCHMARK, start_date, end_date, raw_dir
    )
    ff3_raw, mom_raw = acquire_raw_fama_french_factors(start_date, end_date, raw_dir)

    val_funds = validate_dataframe(funds_raw, "Raw Funds NAVs", ["Date", "Adj Close", "symbol"], id_col="symbol", price_cols=["Adj Close"])
    val_bench = validate_dataframe(bench_raw, "Raw S&P 500 Benchmark", ["Date", "Adj Close"], price_cols=["Adj Close"])
    val_ff3 = validate_dataframe(ff3_raw, "Raw Fama-French 3-Factor", ["Date"])
    val_mom = validate_dataframe(mom_raw, "Raw Momentum Factor", ["Date"])

    proc_funds = preprocess_price_series(funds_raw, price_col="Adj Close", symbol_col="symbol")
    proc_bench = preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col=None)
    proc_factors = preprocess_fama_french_factors(ff3_raw, mom_raw)

    proc_funds.to_csv(os.path.join(processed_dir, "cleaned_fund_returns.csv"), index=False)
    proc_factors.to_csv(os.path.join(processed_dir, "cleaned_factors.csv"), index=False)

    aligned_df = create_aligned_dataset(proc_funds, proc_bench, proc_factors)
    aligned_filepath = os.path.join(processed_dir, "aligned_analysis_dataset.csv")
    aligned_df.to_csv(aligned_filepath, index=False)

    summary = {
        "raw_funds_rows": len(funds_raw),
        "raw_bench_rows": len(bench_raw),
        "raw_ff3_rows": len(ff3_raw),
        "raw_mom_rows": len(mom_raw),
        "aligned_dataset_rows": len(aligned_df),
        "unique_funds_in_aligned": int(aligned_df["symbol"].nunique()),
        "aligned_date_min": str(aligned_df["Date"].min()),
        "aligned_date_max": str(aligned_df["Date"].max()),
        "validation_reports": [val_funds, val_bench, val_ff3, val_mom]
    }

    print("=====================================================================")
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Aligned Dataset Rows: {summary['aligned_dataset_rows']} across {summary['unique_funds_in_aligned']} funds")
    print(f"Date Coverage: {summary['aligned_date_min']} to {summary['aligned_date_max']}")
    print("=====================================================================")

    return summary


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_data_pipeline(root)
