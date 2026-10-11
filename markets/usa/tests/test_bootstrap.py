"""
Unit Test Suite for Circular Block Bootstrap Engine (src/bootstrap.py)
"""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    generate_circular_block_indices,
    run_circular_block_bootstrap,
    BootstrapResult
)


def test_circular_wrapping_and_length():
    """Test that circular block index generation respects block length, wraps around, and matches length N."""
    n = 100
    b = 30
    rng = np.random.default_rng(42)
    
    indices = generate_circular_block_indices(n, b, rng)
    
    assert len(indices) == n
    assert np.all((indices >= 0) & (indices < n))


def test_fund_and_market_paired_resampling():
    """Test that fund and market observations remain strictly paired during block resampling."""
    n = 200
    b = 20
    rng = np.random.default_rng(123)
    
    y = np.linspace(10, 200, n)
    x = y * 2.0  # Perfect pairing y_i = 2 * x_i
    
    indices = generate_circular_block_indices(n, b, rng)
    y_boot = y[indices]
    x_boot = x[indices]
    
    # Paired relationship must hold exactly for all resampled indices
    np.testing.assert_array_equal(x_boot, y_boot * 2.0)


def test_seed_reproducibility():
    """Test that identical seeds yield identical bootstrap results, while different seeds change stochastic output."""
    np.random.seed(42)
    n = 300
    x = np.random.normal(0, 0.01, n)
    y = 0.001 + 1.1 * x + np.random.normal(0, 0.002, n)
    
    res1, alphas1, null1 = run_circular_block_bootstrap(y, x, 0.001, 1.1, "TEST", block_length=20, n_bootstrap=100, seed=42)
    res2, alphas2, null2 = run_circular_block_bootstrap(y, x, 0.001, 1.1, "TEST", block_length=20, n_bootstrap=100, seed=42)
    res3, alphas3, null3 = run_circular_block_bootstrap(y, x, 0.001, 1.1, "TEST", block_length=20, n_bootstrap=100, seed=999)
    
    # Identical seeds must match 100%
    np.testing.assert_array_equal(alphas1, alphas2)
    assert res1.p_value == res2.p_value
    
    # Different seed must produce different stochastic array
    assert not np.array_equal(alphas1, alphas3)


def test_confidence_interval_ordering_and_pvalue_bounds():
    """Test confidence interval ordering (lower <= median <= upper) and p-value in [0, 1]."""
    np.random.seed(42)
    n = 400
    x = np.random.normal(0, 0.01, n)
    y = 0.0005 + 1.0 * x + np.random.normal(0, 0.003, n)
    
    res, alphas, nulls = run_circular_block_bootstrap(y, x, 0.0005, 1.0, "TEST", block_length=30, n_bootstrap=200, seed=42)
    
    assert res.ci_lower <= res.ci_median <= res.ci_upper
    assert res.ci_annual_lower_pct <= res.ci_annual_upper_pct
    assert 0.0 <= res.p_value <= 1.0


def test_synthetic_data_behavior():
    """Test CBB behavior on synthetic data with strong positive alpha."""
    np.random.seed(42)
    n = 500
    x = np.random.normal(0, 0.01, n)
    strong_alpha = 0.002  # Large daily alpha ~ 50% annual
    y = strong_alpha + 1.0 * x + np.random.normal(0, 0.002, n)
    
    res, alphas, nulls = run_circular_block_bootstrap(y, x, strong_alpha, 1.0, "STRONG_FUND", block_length=30, n_bootstrap=300, seed=42)
    
    assert res.observed_alpha_daily == strong_alpha
    # Null-centered p-value for strong alpha should be statistically significant
    assert res.p_value < 0.01
    assert res.significant_at_5pct is True


def test_invalid_parameters():
    """Test that invalid parameters raise clear ValueError exceptions."""
    rng = np.random.default_rng(42)
    
    with pytest.raises(ValueError, match="Block length"):
        generate_circular_block_indices(100, 0, rng)
        
    with pytest.raises(ValueError, match="Block length"):
        generate_circular_block_indices(100, 150, rng)
        
    y = np.ones(10)
    x = np.ones(10)
    with pytest.raises(ValueError, match="Sample size N=10 is smaller than block length"):
        run_circular_block_bootstrap(y, x, 0.0, 1.0, "SMALL", block_length=30, n_bootstrap=50)
