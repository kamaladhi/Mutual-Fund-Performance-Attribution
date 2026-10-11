"""
Independent numerical check of the SAVED results against statsmodels (both markets).

scripts/validate_project.py re-implements the project's own formulas, so it cannot catch a formula that is wrong
in both places. This test compares the saved regression results with a separate, widely used implementation:

  * OLS coefficients and classical p-values        (statsmodels OLS)
  * Newey-West HAC alpha p-values                  (statsmodels cov_type="HAC", Bartlett kernel, n/(n-k) correction)
  * Benjamini-Hochberg q-values                    (statsmodels multipletests, method="fdr_bh")

Skipped automatically if statsmodels is not installed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")
multitest = pytest.importorskip("statsmodels.stats.multitest")

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ["india", "usa"]


def _proc(market, name):
    return pd.read_csv(ROOT / "markets" / market / "data" / "processed" / name, dtype={"symbol": str})


def _hac_lags(n):
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@pytest.fixture(scope="module", params=MARKETS)
def market_data(request):
    market = request.param
    return market, _proc(market, "aligned_analysis_dataset.csv"), _proc(market, "capm_results.csv").set_index("symbol"), \
        _proc(market, "factor_model_results.csv").set_index("symbol"), _proc(market, "final_evaluation.csv").set_index("symbol")


def test_capm_matches_statsmodels(market_data):
    market, aligned, capm, _, _ = market_data
    for sym, g in aligned.groupby("symbol"):
        g = g.sort_values("Date")
        X = sm.add_constant(g["excess_market_return"].to_numpy())
        y = g["excess_fund_return"].to_numpy()
        ols = sm.OLS(y, X).fit()
        hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(y)), "use_correction": True}, use_t=True)
        row = capm.loc[sym]
        assert row["alpha_daily"] == pytest.approx(ols.params[0], rel=1e-9, abs=1e-15), (market, sym)
        assert row["beta"] == pytest.approx(ols.params[1], rel=1e-9), (market, sym)
        assert row["alpha_pvalue"] == pytest.approx(ols.pvalues[0], rel=1e-6, abs=1e-12), (market, sym)
        assert row["r_squared"] == pytest.approx(ols.rsquared, rel=1e-9), (market, sym)
        assert row["alpha_pvalue_hac"] == pytest.approx(hac.pvalues[0], rel=1e-6, abs=1e-12), (market, sym)


def test_four_factor_matches_statsmodels(market_data):
    market, aligned, _, factor, _ = market_data
    for sym, g in aligned.groupby("symbol"):
        g = g.sort_values("Date")
        X = sm.add_constant(g[["Mkt-RF", "SMB", "HML", "Mom"]].to_numpy())
        y = g["excess_fund_return"].to_numpy()
        ols = sm.OLS(y, X).fit()
        hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(len(y)), "use_correction": True}, use_t=True)
        row = factor.loc[sym]
        np.testing.assert_allclose(
            [row["alpha_daily"], row["beta_market"], row["beta_smb"], row["beta_hml"], row["beta_momentum"]],
            ols.params, rtol=1e-8, atol=1e-14, err_msg=f"{market} {sym}")
        assert row["alpha_pvalue"] == pytest.approx(ols.pvalues[0], rel=1e-6, abs=1e-12), (market, sym)
        assert row["alpha_pvalue_hac"] == pytest.approx(hac.pvalues[0], rel=1e-6, abs=1e-12), (market, sym)


def test_benjamini_hochberg_q_values_match_statsmodels(market_data):
    market, _, _, _, final = market_data
    for p_col, q_col in [("capm_alpha_pvalue", "capm_alpha_pvalue_bh"),
                         ("factor_alpha_pvalue", "factor_alpha_pvalue_bh"),
                         ("bootstrap_p_value", "bootstrap_p_value_bh_computed")]:
        expected = multitest.multipletests(final[p_col].to_numpy(float), method="fdr_bh")[1]
        np.testing.assert_allclose(final[q_col].to_numpy(float), expected, rtol=1e-12, atol=1e-14, err_msg=f"{market} {q_col}")
