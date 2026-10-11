"""
Integrity tests that re-verify the SHIPPED US data and results independently of the pipeline code paths:

  * the aligned dataset is complete, unique and on the Fama-French market definition;
  * fund returns recompute from the raw adjusted prices;
  * the stored regression results equal an independent numpy.lstsq re-estimation;
  * the stored HAC p-values equal a brute-force Newey-West implementation;
  * persistence flags follow the strict-majority rule;
  * final_evaluation.csv is fresh (equals a re-evaluation of its four input files) and equals the JSON.

These tests assert invariants, not the current classifications, so a correct change cannot "fail" them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluator import evaluate_fund, load_and_validate_inputs

ROOT = Path(__file__).parent.parent.resolve()
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def _csv(name, **kw):
    return pd.read_csv(PROC / name, dtype={"symbol": str}, **kw)


@pytest.fixture(scope="module")
def aligned():
    return _csv("aligned_analysis_dataset.csv")


def _brute_force_hac_cov(X, resid, lags):
    n, k = X.shape
    S = sum(resid[t] ** 2 * np.outer(X[t], X[t]) for t in range(n))
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1.0)
        for t in range(l, n):
            S = S + w * resid[t] * resid[t - l] * (np.outer(X[t], X[t - l]) + np.outer(X[t - l], X[t]))
    bread = np.linalg.inv(X.T @ X)
    return bread @ S @ bread * n / (n - k)


# ------------------------------------------------------------------ aligned dataset ---------------

def test_aligned_dataset_is_complete_and_unique(aligned):
    key_cols = ["daily_return", "Mkt-RF", "SMB", "HML", "RF", "Mom", "excess_fund_return", "excess_market_return"]
    assert not aligned[key_cols].isna().any().any()
    assert not aligned.duplicated(["symbol", "Date"]).any()
    assert aligned["symbol"].nunique() == 12
    assert aligned.groupby("symbol").size().nunique() == 1          # every fund has the same number of observations
    assert aligned["Date"].min() >= "2019-01-01" and aligned["Date"].max() <= "2023-12-31"


def test_excess_returns_are_exact_and_capm_uses_the_fama_french_market(aligned):
    np.testing.assert_allclose(aligned["excess_fund_return"], aligned["daily_return"] - aligned["RF"], atol=1e-15)
    np.testing.assert_allclose(aligned["excess_market_return"], aligned["Mkt-RF"], atol=1e-15)


def test_factor_units_are_decimals_not_percent(aligned):
    by_date = aligned.drop_duplicates("Date")
    assert by_date[["Mkt-RF", "SMB", "HML", "Mom"]].abs().max().max() < 0.2
    assert 0.0 <= by_date["RF"].min() and by_date["RF"].max() < 0.01


def test_daily_returns_recompute_from_raw_adjusted_prices(aligned):
    funds = pd.read_csv(RAW / "raw_fund_prices.csv").sort_values(["symbol", "Date"])
    funds["ret"] = funds.groupby("symbol")["Adj Close"].pct_change()
    merged = aligned[["symbol", "Date", "daily_return"]].merge(funds[["symbol", "Date", "ret"]], on=["symbol", "Date"])
    assert len(merged) == len(aligned)
    np.testing.assert_allclose(merged["daily_return"], merged["ret"], atol=1e-12)


def test_no_return_is_an_implausible_data_error(aligned):
    assert aligned["daily_return"].abs().max() < 0.25


# ------------------------------------------------------------------ stored regressions ------------

def test_stored_capm_results_equal_independent_lstsq(aligned):
    capm = _csv("capm_results.csv").set_index("symbol")
    for sym, g in aligned.groupby("symbol"):
        g = g.sort_values("Date")
        X = np.column_stack([np.ones(len(g)), g["excess_market_return"].values])
        y = g["excess_fund_return"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        row = capm.loc[sym]
        assert row["n_observations"] == len(g)
        assert row["alpha_daily"] == pytest.approx(coef[0], rel=1e-8, abs=1e-14)
        assert row["beta"] == pytest.approx(coef[1], rel=1e-8)
        assert row["r_squared"] == pytest.approx(1 - resid @ resid / np.sum((y - y.mean()) ** 2), rel=1e-8)


def test_stored_factor_results_equal_independent_lstsq(aligned):
    fac = _csv("factor_model_results.csv").set_index("symbol")
    for sym, g in aligned.groupby("symbol"):
        g = g.sort_values("Date")
        X = np.column_stack([np.ones(len(g)), g[["Mkt-RF", "SMB", "HML", "Mom"]].values])
        y = g["excess_fund_return"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        row = fac.loc[sym]
        np.testing.assert_allclose(
            [row["alpha_daily"], row["beta_market"], row["beta_smb"], row["beta_hml"], row["beta_momentum"]],
            coef, rtol=1e-7, atol=1e-13)


def test_stored_hac_pvalues_equal_brute_force_newey_west(aligned):
    capm = _csv("capm_results.csv").set_index("symbol")
    for sym in ["FDGRX", "AGTHX", "PRHSX"]:
        g = aligned[aligned["symbol"] == sym].sort_values("Date")
        X = np.column_stack([np.ones(len(g)), g["excess_market_return"].values])
        y = g["excess_fund_return"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        lags = int(capm.loc[sym, "hac_lags"])
        assert lags == int(np.floor(4 * (len(g) / 100) ** (2 / 9)))
        cov = _brute_force_hac_cov(X, resid, lags)
        t_stat = coef[0] / np.sqrt(cov[0, 0])
        assert capm.loc[sym, "alpha_pvalue_hac"] == pytest.approx(2 * stats.t.sf(abs(t_stat), len(g) - 2), rel=1e-6)


# ------------------------------------------------------------------ bootstrap / persistence -------

def test_bootstrap_output_invariants():
    boot = _csv("bootstrap_results.csv")
    B = boot["n_bootstrap"].iloc[0]
    assert (boot["p_value"] >= 1.0 / (B + 1) - 1e-12).all() and (boot["p_value"] <= 1).all()
    assert (boot["ci_lower"] <= boot["ci_median"]).all() and (boot["ci_median"] <= boot["ci_upper"]).all()
    assert (boot["block_length"] == 30).all() and (boot["seed"] == 42).all()


def test_bootstrap_point_estimate_matches_capm():
    boot = _csv("bootstrap_results.csv").set_index("symbol")
    capm = _csv("capm_results.csv").set_index("symbol")
    np.testing.assert_allclose(boot["observed_alpha_daily"], capm.loc[boot.index, "alpha_daily"], atol=1e-15)


def test_persistence_periods_partition_the_sample(aligned):
    per = _csv("persistence_period_results.csv")
    assert per.groupby("symbol")["n_observations"].sum().eq(aligned.groupby("symbol").size()).all()
    for _, g in per.groupby("symbol"):
        g = g.sort_values("period_id")
        assert list(g["period_id"]) == ["P1", "P2", "P3", "P4"]
        assert (g["end_date"].values[:-1] < g["start_date"].values[1:]).all()


def test_persistence_summary_follows_the_strict_majority_rule():
    per = _csv("persistence_period_results.csv")
    summ = _csv("persistence_summary.csv").set_index("symbol")
    for sym, g in per.groupby("symbol"):
        pos = int((g["alpha_daily"] > 0).sum())
        assert summ.loc[sym, "positive_alpha_periods"] == pos
        assert bool(summ.loc[sym, "alpha_positive_majority"]) == (pos > 2)       # strictly more than half of 4
        assert bool(summ.loc[sym, "alpha_positive_all_periods"]) == (pos == 4)


# ------------------------------------------------------------------ evaluation freshness ----------

def test_final_evaluation_is_fresh_and_matches_json():
    merged = load_and_validate_inputs(ROOT)
    recomputed = pd.DataFrame([evaluate_fund(r) for _, r in merged.iterrows()]).set_index("symbol")
    stored = _csv("final_evaluation.csv").set_index("symbol")
    assert set(recomputed.index) == set(stored.index)
    for col in ["total_score", "performance_score", "significance_score", "factor_score", "persistence_score",
                "style_luck_adjustment"]:
        np.testing.assert_allclose(stored.loc[recomputed.index, col], recomputed[col], atol=1e-9)
    assert (stored.loc[recomputed.index, "final_category"] == recomputed["final_category"]).all()
    assert set(stored["evaluation_version"]) == {"2.3.0"}

    as_json = pd.DataFrame(json.loads((PROC / "final_evaluation.json").read_text(encoding="utf-8")))
    as_json["symbol"] = as_json["symbol"].astype(str)
    assert (as_json.set_index("symbol").loc[stored.index, "final_category"] == stored["final_category"]).all()
