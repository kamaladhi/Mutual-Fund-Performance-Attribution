"""
Unit Test Suite for Data Loader & Preprocessing Pipeline
"""

import os
import tempfile
import json
import pandas as pd
import numpy as np
import pytest

from src.data_loader import (
    validate_dataframe,
    preprocess_price_series,
    preprocess_fama_french_factors,
    create_aligned_dataset,
    initialize_screening_metadata,
    acquire_raw_fund_and_benchmark_data
)


@pytest.fixture
def sample_price_df():
    """Synthetic price dataframe for unit testing return calculations and validation."""
    return pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "Adj Close": [100.0, 102.0, 101.0, 103.02],
        "symbol": ["TEST_FUND"] * 4
    })


@pytest.fixture
def sample_factors_df():
    """Synthetic factor dataframes matching raw Dartmouth format."""
    ff3 = pd.DataFrame({
        "Date": ["20230103", "20230104", "20230105", "20230106"],
        "Mkt-RF": [1.0, -0.5, 0.2, 0.8],
        "SMB": [0.1, 0.2, -0.1, 0.0],
        "HML": [-0.2, 0.3, 0.1, -0.4],
        "RF": [0.01, 0.01, 0.01, 0.01]
    })
    mom = pd.DataFrame({
        "Date": ["20230103", "20230104", "20230105", "20230106"],
        "Mom": [0.5, -0.2, 0.4, 0.1]
    })
    return ff3, mom


def test_date_parsing_and_validation(sample_price_df):
    """Test date parsing, required columns, and chronological ordering validation."""
    val_report = validate_dataframe(sample_price_df, "Sample DF", ["Date", "Adj Close", "symbol"], id_col="symbol")
    assert val_report["is_valid"] is True
    assert val_report["duplicate_dates_count"] == 0
    assert val_report["is_chronologically_sorted"] is True


def test_duplicate_detection():
    """Test detection of duplicate dates."""
    dup_df = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-03", "2023-01-04"],
        "Adj Close": [100.0, 100.0, 101.0],
        "symbol": ["F1", "F1", "F1"]
    })
    val_report = validate_dataframe(dup_df, "Dup DF", ["Date", "Adj Close"], id_col="symbol")
    assert val_report["duplicate_dates_count"] == 1
    assert len(val_report["validation_warnings"]) > 0


def test_missing_data_detection():
    """Test detection of missing values."""
    missing_df = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05"],
        "Adj Close": [100.0, np.nan, 102.0],
        "symbol": ["F1", "F1", "F1"]
    })
    val_report = validate_dataframe(missing_df, "Missing DF", ["Date", "Adj Close"])
    assert val_report["missing_value_counts"]["Adj Close"] == 1


def test_return_calculation(sample_price_df):
    """Test daily return calculation R_t = (P_t - P_{t-1}) / P_{t-1}."""
    processed = preprocess_price_series(sample_price_df, price_col="Adj Close", symbol_col="symbol")
    assert "daily_return" in processed.columns
    # Check return at t=1: (102 - 100) / 100 = 0.02
    np.testing.assert_almost_equal(processed.loc[1, "daily_return"], 0.02)
    # Check return at t=2: (101 - 102) / 102 = -0.00980392
    np.testing.assert_almost_equal(processed.loc[2, "daily_return"], (101.0 - 102.0) / 102.0)


def test_factor_percentage_to_decimal_conversion(sample_factors_df):
    """Test scaling raw factor percentages (e.g. 1.0) to decimals (0.01)."""
    ff3_raw, mom_raw = sample_factors_df
    proc_factors = preprocess_fama_french_factors(ff3_raw, mom_raw)
    
    assert "raw_Mkt-RF" in proc_factors.columns
    assert proc_factors.loc[0, "raw_Mkt-RF"] == 1.0
    assert proc_factors.loc[0, "Mkt-RF"] == 0.01  # 1.0 / 100.0
    assert proc_factors.loc[0, "RF"] == 0.0001   # 0.01 / 100.0


def test_alignment_behavior(sample_price_df, sample_factors_df):
    """Test common-date inner join alignment across fund returns, benchmark, and factors."""
    proc_funds = preprocess_price_series(sample_price_df, price_col="Adj Close", symbol_col="symbol")
    
    bench_df = sample_price_df.copy()
    bench_df["symbol"] = "^GSPC"
    proc_bench = preprocess_price_series(bench_df, price_col="Adj Close", symbol_col=None)
    
    ff3_raw, mom_raw = sample_factors_df
    proc_factors = preprocess_fama_french_factors(ff3_raw, mom_raw)
    
    aligned = create_aligned_dataset(proc_funds, proc_bench, proc_factors)
    
    # Row t=0 dropped because daily_return is NaN for t=0
    assert len(aligned) == 3
    assert "excess_fund_return" in aligned.columns
    assert "excess_market_return" in aligned.columns


def test_raw_file_protection():
    """Test that existing raw files under data/raw are not silently overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        test_file = os.path.join(raw_dir, "raw_fund_prices.csv")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Date,Adj Close,symbol\n2023-01-03,100.0,DUMMY\n")
        
        # A cached benchmark prevents this unit test from requiring a live network.
        bench_file = os.path.join(raw_dir, "raw_benchmark_sp500.csv")
        with open(bench_file, "w", encoding="utf-8") as f:
            f.write("Date,Adj Close,Close\n2023-01-03,3800.0,3800.0\n")

        # Call acquire_raw_fund_and_benchmark_data
        funds_df, bench_df, meta = acquire_raw_fund_and_benchmark_data(
            ["AGTHX"], "^GSPC", "2023-01-01", "2023-01-05", raw_dir
        )
        
        # Verify dummy file content was preserved and loaded, not overwritten
        assert "DUMMY" in funds_df["symbol"].values
