"""
Tests for src/descriptive.py (FR-STA-01: annualised return, volatility, Sharpe ratio, drawdown).
"""

import numpy as np
import pandas as pd
import pytest

from src.descriptive import run_descriptive_statistics, summarize_return_series


def test_two_day_hand_computed_case():
    stats = summarize_return_series(np.array([0.10, -0.10]))
    assert stats["cumulative_return_pct"] == pytest.approx(-1.0)                  # 1.1 * 0.9 - 1
    assert stats["max_drawdown_pct"] == pytest.approx((0.99 / 1.10 - 1) * 100)     # peak 1.10 -> trough 0.99
    assert stats["annualized_return_cagr_pct"] == pytest.approx((0.99 ** 126 - 1) * 100)
    assert stats["annualized_return_arithmetic_pct"] == pytest.approx(0.0, abs=1e-12)
    assert stats["annualized_volatility_pct"] == pytest.approx(np.std([0.1, -0.1], ddof=1) * np.sqrt(252) * 100)
    assert stats["sharpe_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_constant_return_has_zero_volatility_and_drawdown():
    stats = summarize_return_series(np.full(252, 0.001))
    assert stats["annualized_return_cagr_pct"] == pytest.approx((1.001 ** 252 - 1) * 100)
    assert stats["annualized_volatility_pct"] == pytest.approx(0.0, abs=1e-12)
    assert stats["max_drawdown_pct"] == 0.0
    assert np.isnan(stats["sharpe_ratio"])                                         # undefined: zero volatility


def test_sharpe_uses_excess_returns():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0008, 0.01, 1000)
    rf = np.full(1000, 0.0003)
    with_rf = summarize_return_series(r, rf)["sharpe_ratio"]
    without = summarize_return_series(r)["sharpe_ratio"]
    assert with_rf < without                                                        # subtracting RF lowers Sharpe
    ex = r - rf
    assert with_rf == pytest.approx(ex.mean() * 252 / (ex.std(ddof=1) * np.sqrt(252)))


def test_nan_returns_are_ignored_and_short_series_rejected():
    s = summarize_return_series(np.array([0.01, np.nan, 0.02, -0.01]))
    assert s["n_observations"] == 3
    with pytest.raises(ValueError):
        summarize_return_series(np.array([0.01]))


def test_run_descriptive_statistics_on_real_aligned_data():
    root = __import__("pathlib").Path(__file__).parent.parent
    aligned = pd.read_csv(root / "data" / "processed" / "aligned_analysis_dataset.csv", dtype={"symbol": str})
    table = run_descriptive_statistics(aligned)
    funds = table[table["kind"] == "fund"]
    markets = table[table["kind"] == "market"]
    assert len(funds) == 12 and set(markets["series"]) >= {"iima_market_return", "nifty500_price_return"}
    assert (funds["annualized_volatility_pct"].between(8, 40)).all()
    assert (funds["max_drawdown_pct"] < 0).all()
    # the dividend-free index must compound slower than the total-return market over the same days
    cagr = markets.set_index("series")["annualized_return_cagr_pct"]
    assert cagr["iima_market_return"] > cagr["nifty500_price_return"]
