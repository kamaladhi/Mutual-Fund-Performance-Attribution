"""
Unit Test Suite for Performance Persistence Engine (src/persistence.py)
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.persistence import (
    get_chronological_period_splits,
    run_period_capm_analysis,
    calculate_persistence_metrics,
    PeriodCAPMResult,
    PersistenceSummaryResult
)


@pytest.fixture
def sample_dates_and_df():
    """Generate multi-period synthetic return dataset covering 400 trading days for 2 funds."""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2021-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    
    # Fund A has large positive alpha (+0.005 daily) in all 4 periods
    f_a = 0.005 + 1.0 * np.random.normal(0, 0.01, n) + np.random.normal(0, 0.0005, n)
    # Fund B has large negative alpha (-0.005 daily) in all 4 periods
    f_b = -0.005 + 0.9 * np.random.normal(0, 0.01, n) + np.random.normal(0, 0.0005, n)
    mkt = np.random.normal(0, 0.01, n)

    df_a = pd.DataFrame({
        "Date": dates,
        "symbol": "FUND_A",
        "excess_fund_return": f_a,
        "excess_market_return": mkt
    })
    df_b = pd.DataFrame({
        "Date": dates,
        "symbol": "FUND_B",
        "excess_fund_return": f_b,
        "excess_market_return": mkt
    })

    return dates, pd.concat([df_a, df_b], ignore_index=True)


def test_chronological_period_splitting(sample_dates_and_df):
    """Test 4-period non-overlapping chronological date splitting, full coverage, and boundaries."""
    dates, df = sample_dates_and_df
    splits = get_chronological_period_splits(list(dates), num_periods=4)

    assert len(splits) == 4
    
    # Test no overlapping dates and total coverage
    all_p_dates = []
    for sp in splits:
        assert sp["n_dates"] == 100
        all_p_dates.extend(list(sp["dates"]))
    
    assert len(all_p_dates) == 400
    assert len(set(all_p_dates)) == 400  # Zero overlap
    
    # Test chronological order
    assert splits[0]["end_date"] < splits[1]["start_date"]
    assert splits[1]["end_date"] < splits[2]["start_date"]
    assert splits[2]["end_date"] < splits[3]["start_date"]


def test_period_capm_estimation_and_annualization(sample_dates_and_df):
    """Test period-level CAPM estimation and annualization formula."""
    _, df = sample_dates_and_df
    period_df, fund_dict = run_period_capm_analysis(df, num_periods=4)

    assert len(period_df) == 8  # 2 funds * 4 periods
    assert "FUND_A" in fund_dict
    assert len(fund_dict["FUND_A"]) == 4

    # Check annualization compounding formula ((1 + alpha)^252 - 1) * 100
    p1_a = fund_dict["FUND_A"][0]
    expected_ann = ((1.0 + p1_a.alpha_daily) ** 252 - 1.0) * 100.0
    np.testing.assert_almost_equal(p1_a.alpha_annual_pct, expected_ann)
    assert p1_a.alpha_ci_lower <= p1_a.alpha_daily <= p1_a.alpha_ci_upper


def test_persistence_metrics_calculation(sample_dates_and_df):
    """Test persistence metrics: positive alpha count, persistence rate, mean/median/std."""
    _, df = sample_dates_and_df
    period_df, _ = run_period_capm_analysis(df, num_periods=4)
    summary_df = calculate_persistence_metrics(period_df)

    assert len(summary_df) == 2
    
    row_a = summary_df[summary_df["symbol"] == "FUND_A"].iloc[0]
    row_b = summary_df[summary_df["symbol"] == "FUND_B"].iloc[0]

    assert row_a["total_periods"] == 4
    assert float(row_a["positive_alpha_persistence_rate"]) == 1.0
    assert bool(row_a["alpha_positive_all_periods"]) is True
    assert bool(row_a["alpha_positive_majority"]) is True

    assert float(row_b["positive_alpha_persistence_rate"]) == 0.0
    assert bool(row_b["alpha_positive_all_periods"]) is False
    assert bool(row_b["alpha_positive_majority"]) is False


def test_source_data_immutability(sample_dates_and_df):
    """Test that source dataframes are not mutated in place."""
    _, df = sample_dates_and_df
    df_copy = df.copy(deep=True)

    run_period_capm_analysis(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_invalid_input_handling():
    """Test invalid input error handling (insufficient dates/periods)."""
    with pytest.raises(ValueError, match="Total dates"):
        get_chronological_period_splits(["2023-01-01", "2023-01-02"], num_periods=4)
