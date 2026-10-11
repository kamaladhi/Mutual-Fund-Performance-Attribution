"""Independent numerical-integrity audit for saved fund-manager analysis outputs.

Run from the repository root: python scripts/validate_project.py
This validates stored-series/model-output consistency; it does not certify external
source identity, total-return benchmark completeness, or causal manager skill.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TOL = 2e-8

def hac_cov(X: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    n, k = X.shape
    xu = X * residuals[:, None]
    S = xu.T @ xu
    for lag in range(1, max(0, min(int(lags), n - 1)) + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        S += w * (gamma + gamma.T)
    inv = np.linalg.inv(X.T @ X)
    cov = inv @ S @ inv
    if n > k:
        cov *= n / (n - k)
    return cov

def hac_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n >= 2 else 0

def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    qs = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(qs, 1.0)
    return out

def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)

def audit_market(key: str, failures: list[str]) -> dict:
    project = ROOT / "markets" / key
    data_dir = project / "data" / "processed"
    aligned = pd.read_csv(data_dir / "aligned_analysis_dataset.csv", parse_dates=["Date"], dtype={"symbol": str})
    capm = pd.read_csv(data_dir / "capm_results.csv", dtype={"symbol": str}).set_index("symbol")
    factor = pd.read_csv(data_dir / "factor_model_results.csv", dtype={"symbol": str}).set_index("symbol")
    final = pd.read_csv(data_dir / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
    symbols = sorted(aligned.symbol.unique())
    check(len(symbols) == 12, f"{key}: expected 12 funds, got {len(symbols)}", failures)
    check(not aligned.duplicated(["symbol", "Date"]).any(), f"{key}: duplicate symbol/date rows", failures)
    check(aligned.groupby("symbol").Date.apply(lambda x: x.is_monotonic_increasing).all(), f"{key}: dates not ordered within symbol", failures)
    check(set(symbols) == set(capm.index) == set(factor.index) == set(final.index), f"{key}: symbol universe mismatch across outputs", failures)
    if key == "usa":
        check(np.allclose(aligned["excess_market_return"], aligned["Mkt-RF"], atol=TOL, rtol=0), "USA: CAPM market series is not aligned to Fama-French Mkt-RF", failures)
    max_capm_alpha_err = 0.0
    max_factor_alpha_err = 0.0
    max_hac_p_err = 0.0
    for symbol, g in aligned.groupby("symbol", sort=True):
        g = g.sort_values("Date")
        y = g["excess_fund_return"].to_numpy(float)
        x = g["excess_market_return"].to_numpy(float)
        X = np.column_stack([np.ones(len(y)), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        u = y - X @ b
        max_capm_alpha_err = max(max_capm_alpha_err, abs(float(capm.loc[symbol, "alpha_daily"]) - b[0]))
        cov = hac_cov(X, u, hac_lags(len(y)))
        t = b[0] / np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else 0.0
        p = float(2 * stats.t.sf(abs(t), df=len(y) - X.shape[1]))
        if "alpha_pvalue_hac" in capm.columns:
            max_hac_p_err = max(max_hac_p_err, abs(float(capm.loc[symbol, "alpha_pvalue_hac"]) - p))
        XF = np.column_stack([np.ones(len(y)), g["Mkt-RF"], g["SMB"], g["HML"], g["Mom"]])
        bf = np.linalg.lstsq(XF, y, rcond=None)[0]
        uf = y - XF @ bf
        max_factor_alpha_err = max(max_factor_alpha_err, abs(float(factor.loc[symbol, "alpha_daily"]) - bf[0]))
        covf = hac_cov(XF, uf, hac_lags(len(y)))
        tf = bf[0] / np.sqrt(covf[0, 0]) if covf[0, 0] > 0 else 0.0
        pf = float(2 * stats.t.sf(abs(tf), df=len(y) - XF.shape[1]))
        if "alpha_pvalue_hac" in factor.columns:
            max_hac_p_err = max(max_hac_p_err, abs(float(factor.loc[symbol, "alpha_pvalue_hac"]) - pf))
    check(max_capm_alpha_err < TOL, f"{key}: CAPM alpha reproduction error {max_capm_alpha_err:.3g}", failures)
    check(max_factor_alpha_err < TOL, f"{key}: factor alpha reproduction error {max_factor_alpha_err:.3g}", failures)
    check(max_hac_p_err < 2e-7, f"{key}: HAC p-value reproduction error {max_hac_p_err:.3g}", failures)
    # Recompute q-values from the final raw p-values within this market.
    for pcol, qcol in [("capm_alpha_pvalue", "capm_alpha_pvalue_bh"), ("factor_alpha_pvalue", "factor_alpha_pvalue_bh"), ("bootstrap_p_value", "bootstrap_p_value_bh_computed")]:
        if pcol in final.columns and qcol in final.columns:
            expected = bh(final[pcol].to_numpy(float))
            observed = final[qcol].to_numpy(float)
            check(np.allclose(expected, observed, atol=2e-10, rtol=0), f"{key}: {qcol} does not match BH correction", failures)
    check(final["total_score"].between(0, 100).all(), f"{key}: score outside [0,100]", failures)
    check(final["final_category"].notna().all(), f"{key}: missing classification", failures)
    return {
        "market": key,
        "funds": len(symbols),
        "rows": len(aligned),
        "date_start": str(aligned.Date.min().date()),
        "date_end": str(aligned.Date.max().date()),
        "max_capm_alpha_reproduction_error": max_capm_alpha_err,
        "max_factor_alpha_reproduction_error": max_factor_alpha_err,
        "max_hac_pvalue_reproduction_error": max_hac_p_err,
        "classification_counts": final.final_category.value_counts().to_dict(),
    }

def main() -> int:
    failures: list[str] = []
    summaries = [audit_market(key, failures) for key in ("india", "usa")]
    # Independently recompute returns from the stored raw adjusted NAV/price series.
    for key in ("india", "usa"):
        raw_path = ROOT / "markets" / key / "data" / "raw" / "raw_fund_prices.csv"
        raw = pd.read_csv(raw_path, parse_dates=["Date"], dtype={"symbol": str})
        raw = raw.sort_values(["symbol", "Date"])
        raw["return_recomputed"] = raw.groupby("symbol")["Adj Close"].pct_change()
        aligned = pd.read_csv(ROOT / "markets" / key / "data" / "processed" / "aligned_analysis_dataset.csv", parse_dates=["Date"], dtype={"symbol": str})
        chk = aligned.merge(raw[["symbol", "Date", "return_recomputed"]], on=["symbol", "Date"], how="left", validate="one_to_one")
        mask = chk["daily_return"].notna() & chk["return_recomputed"].notna()
        err = float(np.max(np.abs(chk.loc[mask, "daily_return"].to_numpy(float) - chk.loc[mask, "return_recomputed"].to_numpy(float)))) if mask.any() else float("inf")
        check(err < 1e-9, f"{key}: daily returns differ from raw adjusted NAV/price pct_change (max error {err:.3g})", failures)
    # One rule set for both markets: same evaluation version, same label vocabulary, strict-majority persistence.
    vocabulary = {"Strong Evidence of Skill", "Strong Skill Candidate (Preliminary)", "Moderate Evidence",
                  "Positive but Inconclusive", "Weak/Inconclusive", "Likely Style/Luck Driven"}
    versions: dict[str, set[str]] = {}
    for key in ("india", "usa"):
        proc = ROOT / "markets" / key / "data" / "processed"
        final = pd.read_csv(proc / "final_evaluation.csv", dtype={"symbol": str})
        versions[key] = set(final["evaluation_version"].astype(str))
        check(set(final["final_category"]) <= vocabulary, f"{key}: classification label outside the shared vocabulary: {sorted(set(final['final_category']) - vocabulary)}", failures)
        per = pd.read_csv(proc / "persistence_period_results.csv", dtype={"symbol": str})
        summ = pd.read_csv(proc / "persistence_summary.csv", dtype={"symbol": str}).set_index("symbol")
        for sym, g in per.groupby("symbol"):
            positives = int((g["alpha_daily"] > 0).sum())
            check(bool(summ.loc[sym, "alpha_positive_majority"]) == (positives > len(g) / 2), f"{key}/{sym}: alpha_positive_majority is not the strict-majority rule", failures)
    check(len(versions["india"] | versions["usa"]) == 1, f"markets classified under different rule versions: {versions}", failures)
    # Combined export consistency.
    combined = pd.read_csv(ROOT / "data" / "combined_results.csv", dtype={"symbol": str})
    check(len(combined) == 24, f"combined export: expected 24 rows, got {len(combined)}", failures)
    check(combined.groupby("market").symbol.nunique().to_dict() == {"India": 12, "USA": 12}, "combined export market counts mismatch", failures)
    summaries.append({"market": "shared rule set", "evaluation_versions": {k: sorted(v) for k, v in versions.items()}})
    report = {"status": "PASS" if not failures else "FAIL", "checks": summaries, "failures": failures,
              "scope_note": "Numerical reproduction from saved local series only (the independent statsmodels cross-check lives in tests/test_independent_numerics.py); external source identity, benchmark total-return completeness, fund-universe comparability and manager attribution remain separate caveats."}
    out = ROOT / "docs" / "AUDIT_REPORT.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
