"""
Unit Test Suite for Indian Data Loader & Preprocessing Pipeline
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
    preprocess_iima_factors,
    create_aligned_dataset,
    initialize_screening_metadata
)


@pytest.fixture
def sample_price_df():
    """Synthetic Indian fund NAV dataframe."""
    return pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "Adj Close": [100.0, 102.0, 101.0, 103.02],
        "symbol": ["122639"] * 4
    })


@pytest.fixture
def sample_iima_df():
    """Synthetic IIMA Indian 4-factor dataframe."""
    return pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "MF": [1.0, -0.5, 0.2, 0.8],
        "SMB": [0.1, 0.2, -0.1, 0.0],
        "HML": [-0.2, 0.3, 0.1, -0.4],
        "WML": [0.5, -0.2, 0.4, 0.1],
        "RF": [0.02, 0.02, 0.02, 0.02]
    })


def test_date_parsing_and_validation(sample_price_df):
    val_report = validate_dataframe(sample_price_df, "Sample Indian NAV DF", ["Date", "Adj Close", "symbol"], id_col="symbol")
    assert val_report["is_valid"] is True
    assert val_report["duplicate_dates_count"] == 0
    assert val_report["is_chronologically_sorted"] is True


def test_duplicate_detection():
    dup_df = pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-03", "2023-01-04"],
        "Adj Close": [100.0, 100.0, 101.0],
        "symbol": ["122639", "122639", "122639"]
    })
    val_report = validate_dataframe(dup_df, "Dup DF", ["Date", "Adj Close"], id_col="symbol")
    assert val_report["duplicate_dates_count"] == 1
    assert len(val_report["validation_warnings"]) > 0


def test_return_calculation(sample_price_df):
    processed = preprocess_price_series(sample_price_df, price_col="Adj Close", symbol_col="symbol")
    assert "daily_return" in processed.columns
    np.testing.assert_almost_equal(processed.loc[1, "daily_return"], 0.02)
    np.testing.assert_almost_equal(processed.loc[2, "daily_return"], (101.0 - 102.0) / 102.0)


def test_iima_factor_conversion(sample_iima_df):
    proc_factors = preprocess_iima_factors(sample_iima_df, "2023-01-01", "2023-12-31")
    assert "raw_Mkt-RF" in proc_factors.columns
    assert "Mkt-RF" in proc_factors.columns
    assert "Mom" in proc_factors.columns
    assert proc_factors.loc[0, "raw_Mkt-RF"] == 1.0
    assert proc_factors.loc[0, "Mkt-RF"] == 0.01
    assert proc_factors.loc[0, "RF"] == 0.0002


def test_strict_inner_join_alignment(sample_price_df, sample_iima_df):
    proc_funds = preprocess_price_series(sample_price_df)
    proc_bench = preprocess_price_series(sample_price_df.copy())
    proc_bench["symbol"] = "^CRSLDX"
    proc_factors = preprocess_iima_factors(sample_iima_df, "2023-01-01", "2023-12-31")

    aligned = create_aligned_dataset(proc_funds, proc_bench, proc_factors)
    assert len(aligned) == 3
    assert "excess_fund_return" in aligned.columns
    assert "excess_market_return" in aligned.columns
