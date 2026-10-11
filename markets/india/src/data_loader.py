"""
Data Loader & Preprocessing Pipeline Module
Indian Fund Manager Skill Detector

Handles raw data acquisition for Indian mutual funds (via mfapi.in / AMFI),
the NIFTY 500 index (^CRSLDX via yfinance - a PRICE index), and the IIM Ahmedabad (IIMA)
Indian Fama-French & Momentum 4-factor dataset.

Performs data validation, return calculation, factor decimal scaling,
and strict common-date alignment.

Market proxy
------------
Yahoo Finance's ^CRSLDX is the NIFTY 500 *price* index: it excludes dividends, so using it as
the CAPM market overstates every fund's alpha by roughly the index dividend yield. The CAPM
market proxy is therefore, in order of preference:

  1. ``nifty500_tri``  - a genuine NIFTY 500 Total Return Index series, if the user places one at
                         data/raw/raw_benchmark_nifty500_tri.csv (download from niftyindices.com);
  2. ``iima_market``   - the IIMA value-weighted market portfolio (Mkt-RF + RF; total returns),
                         which is also the market factor of the 4-factor model, so that CAPM and
                         4-factor alphas are measured against the same market.

The NIFTY 500 price-index return is always kept (column ``nifty500_price_return``) as a
labelled sensitivity series.
"""

import os
import json
import time
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

# Target candidate Indian active mutual funds (Direct Plan - Growth)
TARGET_FUNDS = [
    "122639",  # Parag Parikh Flexi Cap Fund
    "118955",  # HDFC Flexi Cap Fund
    "120586",  # ICICI Prudential Large Cap Fund
    "118834",  # Mirae Asset Large & Midcap Fund
    "119598",  # SBI Large Cap Fund
    "120166",  # Kotak Flexi Cap Fund
    "118989",  # HDFC Mid-Cap Opportunities Fund
    "120505",  # Axis Midcap Fund
    "118668",  # Nippon India Growth Mid Cap Fund
    "118778",  # Nippon India Small Cap Fund
    "120828",  # Quant Small Cap Fund
    "119071",  # DSP Midcap Fund
]

# Bumped whenever FUND_METADATA changes; a stale screening_metadata.json is regenerated.
METADATA_VERSION = "2.0.0"

# NOTE: manager names / tenure dates are curated registry data (not machine-verified).
# `tenure_start` is the date the NAMED manager took charge of the scheme. Where that is after the
# analysis window opens, the scheme's 2019-2023 record is NOT (wholly) attributable to that manager.
FUND_METADATA = {
    "122639": {
        "fund_name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
        "short_name": "PPFAS Flexi Cap",
        "category": "Flexi Cap",
        "lead_manager": "Rajeev Thakkar",
        "tenure_start": "2013-05-28",
        "amc": "PPFAS Mutual Fund",
        "expense_ratio": 0.63,
        "notes": "Holds overseas equities and sizeable cash/arbitrage positions that an India-only "
                 "market proxy and factor set do not capture."
    },
    "118955": {
        "fund_name": "HDFC Flexi Cap Fund - Direct Plan - Growth",
        "short_name": "HDFC Flexi Cap",
        "category": "Flexi Cap",
        "lead_manager": "Roshi Jain",
        "tenure_start": "2022-07-29",
        "amc": "HDFC Mutual Fund",
        "expense_ratio": 0.81,
        "notes": "Run by Prashant Jain until July 2022; the named manager was assigned the scheme "
                 "on 2022-07-29 (earlier registry value 2022-01-14 was incorrect) and has since left HDFC AMC."
    },
    "120586": {
        "fund_name": "ICICI Prudential Large Cap Fund - Direct Plan - Growth",
        "short_name": "ICICI Pru Large Cap",
        "category": "Large Cap",
        "lead_manager": "Anish Tawakley",
        "tenure_start": "2018-09-05",
        "amc": "ICICI Prudential AMC",
        "expense_ratio": 0.92,
        "notes": ""
    },
    "118834": {
        "fund_name": "Mirae Asset Large & Midcap Fund - Direct Plan - Growth",
        "short_name": "Mirae Large & Midcap",
        "category": "Large & Mid Cap",
        "lead_manager": "Neelesh Surana",
        "tenure_start": "2010-07-09",
        "amc": "Mirae Asset Mutual Fund",
        "expense_ratio": 0.61,
        "notes": ""
    },
    "119598": {
        "fund_name": "SBI Large Cap Fund - Direct Plan - Growth",
        "short_name": "SBI Large Cap",
        "category": "Large Cap",
        "lead_manager": "Saurabh Pant",
        "tenure_start": "2020-03-30",
        "amc": "SBI Mutual Fund",
        "expense_ratio": 0.88,
        "notes": ""
    },
    "120166": {
        "fund_name": "Kotak Flexi Cap Fund - Direct Plan - Growth",
        "short_name": "Kotak Flexi Cap",
        "category": "Flexi Cap",
        "lead_manager": "Harsha Upadhyaya",
        "tenure_start": "2012-08-04",
        "amc": "Kotak Mahindra AMC",
        "expense_ratio": 0.64,
        "notes": ""
    },
    "118989": {
        "fund_name": "HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth",
        "short_name": "HDFC Mid-Cap",
        "category": "Mid Cap",
        "lead_manager": "Chirag Setalvad",
        "tenure_start": "2007-06-25",
        "amc": "HDFC Mutual Fund",
        "expense_ratio": 0.74,
        "notes": ""
    },
    "120505": {
        "fund_name": "Axis Midcap Fund - Direct Plan - Growth",
        "short_name": "Axis Midcap",
        "category": "Mid Cap",
        "lead_manager": "Shreyash Devalkar",
        "tenure_start": "2016-11-23",
        "amc": "Axis Mutual Fund",
        "expense_ratio": 0.58,
        "notes": ""
    },
    "118668": {
        "fund_name": "Nippon India Growth Mid Cap Fund - Direct Plan - Growth",
        "short_name": "Nippon Growth Mid Cap",
        "category": "Mid Cap",
        "lead_manager": "Rupesh Patel",
        "tenure_start": "2021-04-18",
        "amc": "Nippon India Mutual Fund",
        "expense_ratio": 0.95,
        "notes": ""
    },
    "118778": {
        "fund_name": "Nippon India Small Cap Fund - Direct Plan - Growth",
        "short_name": "Nippon Small Cap",
        "category": "Small Cap",
        "lead_manager": "Samir Rachh",
        "tenure_start": "2017-01-01",
        "amc": "Nippon India Mutual Fund",
        "expense_ratio": 0.68,
        "notes": ""
    },
    "120828": {
        "fund_name": "Quant Small Cap Fund - Direct Plan - Growth",
        "short_name": "Quant Small Cap",
        "category": "Small Cap",
        "lead_manager": "Sandeep Tandon",
        "tenure_start": "2019-11-01",
        "amc": "Quant Mutual Fund",
        "expense_ratio": 0.77,
        "notes": ""
    },
    "119071": {
        "fund_name": "DSP Midcap Fund - Direct Plan - Growth",
        "short_name": "DSP Midcap",
        "category": "Mid Cap",
        "lead_manager": "Vinit Sambre",
        "tenure_start": "2012-07-01",
        "amc": "DSP Mutual Fund",
        "expense_ratio": 0.79,
        "notes": ""
    }
}

# NIFTY 500 index as secondary / sensitivity benchmark.  ^CRSLDX on Yahoo Finance is the PRICE index.
TARGET_BENCHMARK = "^CRSLDX"
BENCHMARK_FALLBACK = "^NSEI"
BENCHMARK_LABEL = "NIFTY 500 price index (^CRSLDX; excludes dividends)"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2023-12-31"

MARKET_PROXY_TRI = "nifty500_tri"
MARKET_PROXY_IIMA = "iima_market"
TRI_FILENAME = "raw_benchmark_nifty500_tri.csv"

# IIMA Indian 4-Factor Data Library URL
IIMA_FOUR_FACTORS_URL = "https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA/2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"
IIMA_REQUIRED_COLUMNS = ["Date", "SMB", "HML", "WML", "MF", "RF"]


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
    Acquire raw daily NAV series for Indian mutual funds from mfapi.in
    and benchmark index series from Yahoo Finance.
    Preserves existing files under data/raw/ without overwriting.
    """
    os.makedirs(raw_dir, exist_ok=True)
    funds_filepath = os.path.join(raw_dir, "raw_fund_prices.csv")
    bench_filepath = os.path.join(raw_dir, "raw_benchmark_nifty500.csv")
    meta_filepath = os.path.join(raw_dir, "acquisition_metadata.json")

    metadata = {}
    if os.path.exists(meta_filepath):
        try:
            with open(meta_filepath, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # Fund Data Ingestion via mfapi.in
    if os.path.exists(funds_filepath):
        print(f"[DATA LOADER] Loading existing raw Indian fund data from {funds_filepath}")
        funds_df = pd.read_csv(funds_filepath, dtype={"symbol": str})
    else:
        print(f"[DATA LOADER] Downloading raw NAVs for {len(symbols)} Indian funds from mfapi.in...")
        all_funds = []
        for symbol in symbols:
            try:
                url = f"https://api.mfapi.in/mf/{symbol}"
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
                data_records = payload.get("data", [])
                if not data_records:
                    print(f"[WARN] No records found for scheme {symbol}")
                    continue

                df_fund = pd.DataFrame(data_records)
                # Parse DD-MM-YYYY dates
                df_fund["Date"] = pd.to_datetime(df_fund["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
                df_fund["Adj Close"] = pd.to_numeric(df_fund["nav"], errors="coerce")
                df_fund["symbol"] = symbol

                # Filter date range
                df_fund = df_fund[(df_fund["Date"] >= start_date) & (df_fund["Date"] <= end_date)]
                df_fund = df_fund.sort_values("Date").reset_index(drop=True)
                all_funds.append(df_fund[["Date", "Adj Close", "symbol"]])
                time.sleep(0.1)  # courteous request delay
            except Exception as e:
                print(f"[ERROR] Failed downloading scheme {symbol}: {e}")

        if not all_funds:
            raise RuntimeError("Failed to acquire any fund data from mfapi.in.")

        funds_df = pd.concat(all_funds, ignore_index=True)
        funds_df.to_csv(funds_filepath, index=False)

        metadata["raw_fund_prices.csv"] = {
            "source": "AMFI via mfapi.in API",
            "retrieval_datetime": datetime.now().isoformat(),
            "requested_date_range": [start_date, end_date],
            "actual_date_range": [funds_df["Date"].min(), funds_df["Date"].max()],
            "tickers": symbols,
            "status": "DOWNLOADED_AND_SAVED"
        }

    # Benchmark Data Ingestion via Yahoo Finance
    if os.path.exists(bench_filepath):
        print(f"[DATA LOADER] Loading existing raw benchmark data from {bench_filepath}")
        bench_df = pd.read_csv(bench_filepath)
    else:
        print(f"[DATA LOADER] Downloading raw benchmark data ({benchmark_symbol} / NIFTY 500 price index)...")
        ticker_used = benchmark_symbol
        bench_df = pd.DataFrame()
        try:
            ticker = yf.Ticker(benchmark_symbol)
            bench_df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        except Exception:
            pass

        if bench_df.empty:
            print(f"[WARN] Primary benchmark {benchmark_symbol} empty, trying {BENCHMARK_FALLBACK}...")
            ticker_used = BENCHMARK_FALLBACK
            ticker = yf.Ticker(BENCHMARK_FALLBACK)
            bench_df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        bench_df = bench_df.reset_index()
        bench_df["symbol"] = ticker_used
        if "Date" in bench_df.columns:
            bench_df["Date"] = pd.to_datetime(bench_df["Date"]).dt.strftime("%Y-%m-%d")

        bench_df.to_csv(bench_filepath, index=False)

        metadata["raw_benchmark_nifty500.csv"] = {
            "source": "Yahoo Finance (yfinance)",
            "retrieval_datetime": datetime.now().isoformat(),
            "requested_date_range": [start_date, end_date],
            "actual_date_range": [bench_df["Date"].min(), bench_df["Date"].max()],
            "ticker": ticker_used,
            "series_type": "price_index (excludes dividends)",
            "status": "DOWNLOADED_AND_SAVED"
        }

    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return funds_df, bench_df, metadata


def assert_factors_not_synthetic(factors_df: pd.DataFrame, source: str = "IIMA factors") -> None:
    """
    Guard against the placeholder (random-number) factor file that earlier versions of this
    pipeline silently generated when the download failed. Real IIMA factors have a
    time-varying risk-free rate; the placeholder had a constant one.
    """
    if "RF" in factors_df.columns and factors_df["RF"].dropna().nunique() <= 1:
        raise RuntimeError(
            f"{source}: the risk-free column is constant, which indicates a synthetic placeholder file. "
            "Delete data/raw/raw_iima_four_factors_daily.csv and re-run with network access, or download the "
            f"genuine file manually from {IIMA_FOUR_FACTORS_URL}"
        )


def acquire_raw_iima_factors(
    start_date: str,
    end_date: str,
    raw_dir: str
) -> pd.DataFrame:
    """
    Acquire daily Indian Fama-French 3-Factor and Momentum datasets
    from IIM Ahmedabad (IIMA) Data Library (Prof. Agarwalla, Jacob, Varma).
    Preserves existing files under data/raw/ without overwriting.

    There is deliberately NO synthetic fallback: if the download fails the pipeline stops, because
    every downstream statistic would otherwise be computed on fabricated factor returns.
    """
    os.makedirs(raw_dir, exist_ok=True)
    factors_filepath = os.path.join(raw_dir, "raw_iima_four_factors_daily.csv")

    if os.path.exists(factors_filepath):
        print(f"[DATA LOADER] Loading existing raw IIMA 4-Factor data from {factors_filepath}")
        factors_df = pd.read_csv(factors_filepath)
    else:
        print("[DATA LOADER] Downloading raw IIMA 4-Factor daily data from IIM Ahmedabad portal...")
        try:
            resp = requests.get(IIMA_FOUR_FACTORS_URL, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Could not download the IIMA factor file ({e}). No synthetic substitute is generated. "
                f"Download it manually from {IIMA_FOUR_FACTORS_URL} and save it as {factors_filepath}"
            ) from e
        with open(factors_filepath, "wb") as f:
            f.write(resp.content)
        factors_df = pd.read_csv(factors_filepath)

    missing = [c for c in IIMA_REQUIRED_COLUMNS if c not in factors_df.columns]
    if missing:
        raise ValueError(f"IIMA factor file is missing required columns: {missing}")
    assert_factors_not_synthetic(factors_df)

    return factors_df


def load_nifty500_tri(raw_dir: str) -> Optional[pd.DataFrame]:
    """
    Load an optional user-supplied NIFTY 500 Total Return Index history from
    data/raw/raw_benchmark_nifty500_tri.csv (e.g. downloaded from niftyindices.com).
    Returns None if the file is absent. Output columns: Date (ISO), Adj Close (index level), symbol.
    """
    path = os.path.join(raw_dir, TRI_FILENAME)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    value_candidates = [c for c in df.columns if c != date_col and
                        any(k in c.lower() for k in ("total", "tri", "adj close", "close"))]
    if date_col is None or not value_candidates:
        raise ValueError(f"{TRI_FILENAME}: need a Date column and a TRI/Close column; found {list(df.columns)}")
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], format="mixed", dayfirst=True).dt.strftime("%Y-%m-%d"),
        "Adj Close": pd.to_numeric(df[value_candidates[0]], errors="coerce"),
    }).dropna()
    out["symbol"] = "NIFTY500_TRI"
    return out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def initialize_screening_metadata(raw_dir: str, candidate_tickers: List[str]) -> str:
    """
    Write data/raw/screening_metadata.json with Indian fund manager metadata.
    The file is regenerated whenever its ``metadata_version`` differs from METADATA_VERSION
    (it is curated registry data, not a downloaded raw series).
    """
    os.makedirs(raw_dir, exist_ok=True)
    screening_path = os.path.join(raw_dir, "screening_metadata.json")

    if os.path.exists(screening_path):
        try:
            with open(screening_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing and all(r.get("metadata_version") == METADATA_VERSION for r in existing):
                print(f"[DATA LOADER] Screening metadata file already exists at {screening_path}")
                return screening_path
        except Exception:
            pass

    records = []
    for ticker in candidate_tickers:
        meta = FUND_METADATA.get(ticker, {})
        records.append({
            "ticker": ticker,
            "fund_name": meta.get("fund_name", f"Mutual Fund {ticker}"),
            "short_name": meta.get("short_name", ticker),
            "category": meta.get("category", "Equity Active"),
            "lead_manager": meta.get("lead_manager", "Unassigned"),
            "tenure_start": meta.get("tenure_start", "N/A"),
            "amc": meta.get("amc", "Indian AMC"),
            "expense_ratio": meta.get("expense_ratio", 0.75),
            "notes": meta.get("notes", ""),
            "status": "CANDIDATE",
            "active_mandate_evidence": f"SEBI {meta.get('category')} category - actively managed, Direct Plan - Growth option",
            "expense_ratio_evidence": f"Direct Plan Total Expense Ratio: {meta.get('expense_ratio', 0.75)}% "
                                      "(point-in-time disclosure; not adjusted over the sample window)",
            "verification_status": "SCHEME_CODE_VERIFIED_AGAINST_MFAPI; MANAGER_TENURE_NOT_INDEPENDENTLY_VERIFIED",
            "source_reference_info": "AMFI scheme master via mfapi.in; manager/tenure from curated registry",
            "metadata_version": METADATA_VERSION,
        })

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
    Also records ``prev_date`` (the date of the observation the return is measured from) so that
    the alignment step can verify that fund, benchmark and factor returns cover the same interval.
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
            group["prev_date"] = group["Date"].shift(1)
        else:
            raise KeyError(f"Price column '{price_col}' not found in dataset.")

        processed_dfs.append(group)

    result_df = pd.concat(processed_dfs, ignore_index=True)
    return result_df


def preprocess_iima_factors(raw_factors_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Clean IIMA 4-Factor daily dataset.
    Standardizes column names:
      MF -> Mkt-RF (Market excess return)
      WML -> Mom (Carhart Momentum)
      SMB -> SMB (Size factor)
      HML -> HML (Value factor)
      RF -> RF (Risk-Free rate)
    Converts percentage values to decimal returns (val / 100).
    Adds ``iima_market_return`` = Mkt-RF + RF (total market return) and ``prev_date``.
    """
    df = raw_factors_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    # previous factor date, computed on the FULL series before windowing
    df["prev_date"] = df["Date"].shift(1)

    rename_dict = {
        "MF": "Mkt-RF",
        "WML": "Mom",
        "WML_Daily": "Mom",
        "SMB": "SMB",
        "HML": "HML",
        "RF": "RF"
    }
    df = df.rename(columns=rename_dict)

    # Filter date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].reset_index(drop=True)

    for col in ["Mkt-RF", "SMB", "HML", "RF", "Mom"]:
        if col in df.columns:
            df[f"raw_{col}"] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[f"raw_{col}"] / 100.0

    if "Mkt-RF" in df.columns and "RF" in df.columns:
        df["iima_market_return"] = df["Mkt-RF"] + df["RF"]

    return df


# =====================================================================
# 4. STRICT COMMON-DATE, COMMON-INTERVAL ALIGNMENT
# =====================================================================

def create_aligned_dataset(
    processed_funds_df: pd.DataFrame,
    processed_bench_df: pd.DataFrame,
    processed_factors_df: pd.DataFrame,
    processed_tri_df: Optional[pd.DataFrame] = None,
    market_proxy: str = "auto",
) -> pd.DataFrame:
    """
    Perform strict common-date inner join across Indian fund returns,
    NIFTY 500 benchmark returns, and IIMA multi-factor series.
    Does NOT fabricate dates or perform ffill/bfill.

    Synchronous-interval rule: every return is a 1-step return on its own series' calendar, so a
    row is kept only if the fund, the benchmark and the IIMA factors all measure the return from the
    SAME previous date (otherwise, e.g. a missing NAV or a Yahoo gap would pair a 2-day fund return
    with a 1-day market return). Dropped rows are counted in ``df.attrs['n_dropped_async']``.

    market_proxy: "auto" (TRI if supplied else IIMA), "iima_market", "nifty500_tri" or
    "nifty500_price".
    """
    bench_sub = processed_bench_df[["Date", "daily_return", "Adj Close", "prev_date"]].copy()
    bench_sub = bench_sub.rename(columns={
        "daily_return": "nifty500_price_return",
        "Adj Close": "nifty500_adj_close",
        "prev_date": "bench_prev_date",
    })

    factor_cols = ["Date", "Mkt-RF", "SMB", "HML", "RF", "Mom", "prev_date"]
    if "iima_market_return" in processed_factors_df.columns:
        factor_cols.append("iima_market_return")
    factors_sub = processed_factors_df[factor_cols].copy().rename(columns={"prev_date": "factor_prev_date"})
    if "iima_market_return" not in factors_sub.columns:
        factors_sub["iima_market_return"] = factors_sub["Mkt-RF"] + factors_sub["RF"]

    funds = processed_funds_df.copy().rename(columns={"prev_date": "fund_prev_date"})

    merged_df = pd.merge(funds, bench_sub, on="Date", how="inner")
    merged_df = pd.merge(merged_df, factors_sub, on="Date", how="inner")

    sync_mask = (merged_df["fund_prev_date"] == merged_df["bench_prev_date"]) & \
                (merged_df["fund_prev_date"] == merged_df["factor_prev_date"])

    if processed_tri_df is not None:
        tri_sub = processed_tri_df[["Date", "daily_return", "prev_date"]].rename(columns={
            "daily_return": "nifty500_tri_return", "prev_date": "tri_prev_date"})
        merged_df = pd.merge(merged_df, tri_sub, on="Date", how="inner")
        sync_mask = (merged_df["fund_prev_date"] == merged_df["bench_prev_date"]) & \
                    (merged_df["fund_prev_date"] == merged_df["factor_prev_date"]) & \
                    (merged_df["fund_prev_date"] == merged_df["tri_prev_date"])

    complete_mask = merged_df[["daily_return", "nifty500_price_return", "RF", "Mkt-RF", "SMB", "HML", "Mom"]].notna().all(axis=1)
    n_before = int(complete_mask.sum())
    keep = complete_mask & sync_mask
    aligned_df = merged_df[keep].reset_index(drop=True)
    n_dropped_async = n_before - len(aligned_df)

    # Resolve the market proxy used by the CAPM engine
    if market_proxy == "auto":
        market_proxy = MARKET_PROXY_TRI if processed_tri_df is not None else MARKET_PROXY_IIMA
    if market_proxy == MARKET_PROXY_TRI:
        if "nifty500_tri_return" not in aligned_df.columns:
            raise ValueError("market_proxy='nifty500_tri' requested but no TRI series was supplied.")
        aligned_df["market_return"] = aligned_df["nifty500_tri_return"]
    elif market_proxy == MARKET_PROXY_IIMA:
        aligned_df["market_return"] = aligned_df["iima_market_return"]
    elif market_proxy == "nifty500_price":
        aligned_df["market_return"] = aligned_df["nifty500_price_return"]
    else:
        raise ValueError(f"Unknown market_proxy '{market_proxy}'.")
    aligned_df["market_proxy"] = market_proxy

    aligned_df["excess_fund_return"] = aligned_df["daily_return"] - aligned_df["RF"]
    aligned_df["excess_market_return"] = aligned_df["market_return"] - aligned_df["RF"]
    if market_proxy == MARKET_PROXY_IIMA:
        # exactly the IIMA market factor (avoids (Mkt-RF + RF) - RF rounding noise)
        aligned_df["excess_market_return"] = aligned_df["Mkt-RF"]
    aligned_df["excess_nifty500_price_return"] = aligned_df["nifty500_price_return"] - aligned_df["RF"]

    aligned_df = aligned_df.drop(columns=[c for c in ["fund_prev_date", "bench_prev_date", "factor_prev_date", "tri_prev_date"]
                                          if c in aligned_df.columns])
    aligned_df.attrs["n_dropped_async"] = n_dropped_async
    aligned_df.attrs["market_proxy"] = market_proxy
    return aligned_df


# =====================================================================
# PIPELINE EXECUTION FUNCTION
# =====================================================================

def run_data_pipeline(
    project_root: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    market_proxy: str = "auto",
) -> Dict[str, Any]:
    """
    Execute complete end-to-end Indian Mutual Fund Data Acquisition
    and Preprocessing Pipeline.
    """
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=====================================================================")
    print("STARTING INDIAN FUND DATA PIPELINE EXECUTION")
    print("=====================================================================")

    initialize_screening_metadata(raw_dir, TARGET_FUNDS)

    funds_raw, bench_raw, acq_meta = acquire_raw_fund_and_benchmark_data(
        TARGET_FUNDS, TARGET_BENCHMARK, start_date, end_date, raw_dir
    )

    factors_raw = acquire_raw_iima_factors(start_date, end_date, raw_dir)
    tri_raw = load_nifty500_tri(raw_dir)

    print("[DATA LOADER] Preprocessing price series and factor data...")
    funds_proc = preprocess_price_series(funds_raw, price_col="Adj Close", symbol_col="symbol")
    bench_proc = preprocess_price_series(bench_raw, price_col="Adj Close", symbol_col="symbol")
    factors_proc = preprocess_iima_factors(factors_raw, start_date, end_date)
    tri_proc = preprocess_price_series(tri_raw, price_col="Adj Close", symbol_col="symbol") if tri_raw is not None else None

    # Validation (FR-DAT-06): fail loudly on structurally invalid inputs, keep the report.
    validation_reports = [
        validate_dataframe(funds_raw, "raw_fund_prices", ["Date", "Adj Close", "symbol"], id_col="symbol", price_cols=["Adj Close"]),
        validate_dataframe(bench_raw, "raw_benchmark_nifty500", ["Date", "Adj Close"], price_cols=["Adj Close"]),
        validate_dataframe(factors_proc, "iima_factors_in_window", ["Date", "Mkt-RF", "SMB", "HML", "Mom", "RF"]),
    ]
    if tri_proc is not None:
        validation_reports.append(validate_dataframe(tri_raw, "raw_benchmark_nifty500_tri", ["Date", "Adj Close"], price_cols=["Adj Close"]))
    with open(os.path.join(processed_dir, "data_validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(validation_reports, fh, indent=2)
    invalid = [r for r in validation_reports if not r["is_valid"]]
    if invalid:
        raise ValueError("Raw data validation failed: " + json.dumps(
            [{"dataset": r["dataset_name"], "errors": r["validation_errors"]} for r in invalid]))

    print("[DATA LOADER] Performing strict common-date, common-interval alignment...")
    aligned_df = create_aligned_dataset(funds_proc, bench_proc, factors_proc, tri_proc, market_proxy=market_proxy)

    aligned_path = os.path.join(processed_dir, "aligned_analysis_dataset.csv")
    aligned_df.to_csv(aligned_path, index=False)
    print(f"[DATA LOADER] Saved aligned analysis dataset ({len(aligned_df)} rows) to {aligned_path}")

    # Build summary
    unique_dates = aligned_df["Date"].nunique()
    funds_count = aligned_df["symbol"].nunique()
    rows_per_fund = aligned_df.groupby("symbol").size()
    n_dropped = aligned_df.attrs.get("n_dropped_async", 0)

    print("=====================================================================")
    print(f"PIPELINE COMPLETE: {funds_count} Indian Funds, {unique_dates} distinct dates "
          f"({int(rows_per_fund.min())}-{int(rows_per_fund.max())} observations per fund)")
    print(f"Market proxy: {aligned_df.attrs.get('market_proxy')} | asynchronous observations dropped: {n_dropped}")
    print("=====================================================================")

    return {
        "status": "SUCCESS",
        "aligned_rows": len(aligned_df),
        "unique_dates": unique_dates,
        "funds_count": funds_count,
        "obs_per_fund_min": int(rows_per_fund.min()),
        "obs_per_fund_max": int(rows_per_fund.max()),
        "market_proxy": aligned_df.attrs.get("market_proxy"),
        "n_dropped_async": int(n_dropped),
        "date_range": [aligned_df["Date"].min(), aligned_df["Date"].max()]
    }


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    res = run_data_pipeline(root)
    print("Pipeline result:", res)
