"""
Integrity tests that re-verify the SHIPPED data and results independently of the pipeline code paths:

  * every aligned return covers the same one-step interval for the fund, the benchmark and the factors;
  * the stored regression results equal an independent numpy.lstsq re-estimation;
  * the stored HAC p-values equal a brute-force Newey-West implementation;
  * final_evaluation.csv is fresh (equals a re-evaluation of its four input files);
  * bootstrap and persistence outputs satisfy their structural invariants.
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
    """Loop-based Newey-West covariance (Bartlett weights, n/(n-k) small-sample factor)."""
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
    key_cols = ["daily_return", "market_return", "RF", "Mkt-RF", "SMB", "HML", "Mom",
                "excess_fund_return", "excess_market_return"]
    assert not aligned[key_cols].isna().any().any()
    assert not aligned.duplicated(["symbol", "Date"]).any()
    assert aligned["symbol"].nunique() == 12
    assert aligned.groupby("symbol").size().min() >= 1200
    assert aligned["Date"].min() >= "2019-01-01" and aligned["Date"].max() <= "2023-12-31"


def test_excess_returns_are_exact(aligned):
    np.testing.assert_allclose(aligned["excess_fund_return"], aligned["daily_return"] - aligned["RF"], atol=1e-15)
    assert set(aligned["market_proxy"]) == {"iima_market"}
    np.testing.assert_allclose(aligned["excess_market_return"], aligned["Mkt-RF"], atol=1e-15)


def test_factor_units_are_decimals_not_percent(aligned):
    by_date = aligned.drop_duplicates("Date")
    assert by_date[["Mkt-RF", "SMB", "HML", "Mom"]].abs().max().max() < 0.2
    assert 0.0 < by_date["RF"].min() and by_date["RF"].max() < 0.01
    assert by_date["RF"].nunique() > 50                                   # genuine, time-varying T-bill series


def test_every_aligned_return_covers_the_same_interval_in_fund_benchmark_and_factors(aligned):
    funds = pd.read_csv(RAW / "raw_fund_prices.csv", dtype={"symbol": str}).sort_values(["symbol", "Date"])
    funds["prev"] = funds.groupby("symbol")["Date"].shift(1)
    bench = pd.read_csv(RAW / "raw_benchmark_nifty500.csv").sort_values("Date")
    bench["prev"] = bench["Date"].shift(1)
    iima = pd.read_csv(RAW / "raw_iima_four_factors_daily.csv").sort_values("Date")
    iima["prev"] = iima["Date"].shift(1)

    m = aligned[["symbol", "Date"]].merge(funds[["symbol", "Date", "prev"]], on=["symbol", "Date"], how="left") \
        .merge(bench[["Date", "prev"]].rename(columns={"prev": "bprev"}), on="Date", how="left") \
        .merge(iima[["Date", "prev"]].rename(columns={"prev": "iprev"}), on="Date", how="left")
    assert m[["prev", "bprev", "iprev"]].notna().all().all()
    assert (m["prev"] == m["bprev"]).all() and (m["prev"] == m["iprev"]).all()


def test_daily_returns_recompute_from_raw_navs(aligned):
    funds = pd.read_csv(RAW / "raw_fund_prices.csv", dtype={"symbol": str}).sort_values(["symbol", "Date"])
    funds["ret"] = funds.groupby("symbol")["Adj Close"].pct_change()
    merged = aligned[["symbol", "Date", "daily_return"]].merge(funds[["symbol", "Date", "ret"]], on=["symbol", "Date"])
    assert len(merged) == len(aligned)
    np.testing.assert_allclose(merged["daily_return"], merged["ret"], atol=1e-12)


def test_no_return_is_an_implausible_data_error(aligned):
    assert aligned["daily_return"].abs().max() < 0.25
    assert aligned.drop_duplicates("Date")["market_return"].abs().max() < 0.25


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
        sigma2 = resid @ resid / (len(g) - 2)
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
        assert row["alpha_stderr"] == pytest.approx(se, rel=1e-8)
        assert row["alpha_pvalue"] == pytest.approx(2 * stats.t.sf(abs(coef[0] / se), len(g) - 2), rel=1e-6, abs=1e-12)
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
    for sym in ["122639", "118778", "119598"]:                           # a spread of funds keeps the test fast
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
    assert (boot["p_value_bh"] >= boot["p_value"] - 1e-12).all()
    # directional flag never fires for a non-positive alpha
    assert not (boot["significantly_positive_5pct"] & (boot["observed_alpha_daily"] <= 0)).any()
    assert (boot["significantly_positive_5pct"] == (boot["significant_at_5pct"] & (boot["observed_alpha_daily"] > 0))).all()
    assert (boot["block_length"] == 30).all() and (boot["seed"] == 42).all()


def test_bootstrap_point_estimate_matches_capm():
    boot = _csv("bootstrap_results.csv").set_index("symbol")
    capm = _csv("capm_results.csv").set_index("symbol")
    np.testing.assert_allclose(boot["observed_alpha_daily"], capm.loc[boot.index, "alpha_daily"], atol=1e-15)


def test_persistence_periods_partition_the_sample(aligned):
    per = _csv("persistence_period_results.csv")
    assert per.groupby("symbol")["n_observations"].sum().eq(aligned.groupby("symbol").size()).all()
    for sym, g in per.groupby("symbol"):
        g = g.sort_values("period_id")
        assert list(g["period_id"]) == ["P1", "P2", "P3", "P4"]
        assert (g["end_date"].values[:-1] < g["start_date"].values[1:]).all()     # non-overlapping, chronological


def test_persistence_summary_matches_period_alphas():
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

    as_json = pd.DataFrame(json.loads((PROC / "final_evaluation.json").read_text(encoding="utf-8")))
    as_json["symbol"] = as_json["symbol"].astype(str)
    assert (as_json.set_index("symbol").loc[stored.index, "final_category"] == stored["final_category"]).all()


def test_report_artifacts_exist():
    for rel in ["results/tables/skill_leaderboard.csv", "results/tables/alpha_comparison.csv",
                "results/figures/evidence_index.png", "results/figures/alpha_comparison.png",
                "results/figures/subperiod_alpha.png", "data/processed/benchmark_sensitivity.csv",
                "data/processed/bootstrap_block_sensitivity.csv", "data/processed/descriptive_statistics.csv",
                "data/processed/run_manifest.json", "data/processed/data_validation_report.json"]:
        assert (ROOT / rel).exists(), rel
