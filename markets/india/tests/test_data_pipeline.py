"""
Pipeline-level tests for src/data_loader.py (added in review, version 2.0.0):
no synthetic fallback, synchronous-interval alignment, market-proxy selection, metadata refresh.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATES = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]


def _iima_raw(dates=DATES):
    n = len(dates)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Date": dates,
        "SMB": rng.normal(0, 0.5, n), "HML": rng.normal(0, 0.5, n), "WML": rng.normal(0, 0.5, n),
        "MF": rng.normal(0.05, 1.0, n), "RF": np.linspace(0.02, 0.03, n),
    })


def _prices(dates, symbol, start=100.0, step=0.5):
    return pd.DataFrame({"Date": dates, "Adj Close": [start + step * i for i in range(len(dates))], "symbol": symbol})


def _frames(fund_dates=DATES, bench_dates=DATES, factor_dates=DATES, tri_dates=None):
    funds = dl.preprocess_price_series(_prices(fund_dates, "F1"))
    bench = dl.preprocess_price_series(_prices(bench_dates, "^CRSLDX", 9000.0, 20.0))
    factors = dl.preprocess_iima_factors(_iima_raw(factor_dates), "2023-01-01", "2023-12-31")
    tri = dl.preprocess_price_series(_prices(tri_dates, "TRI", 20000.0, 50.0)) if tri_dates else None
    return funds, bench, factors, tri


# ------------------------------------------------------------------ no fabricated data ------------

def test_download_failure_raises_instead_of_fabricating_factors(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("offline")
    monkeypatch.setattr(dl.requests, "get", offline)
    with pytest.raises(RuntimeError, match="No synthetic substitute"):
        dl.acquire_raw_iima_factors("2019-01-01", "2023-12-31", str(tmp_path))
    assert not (tmp_path / "raw_iima_four_factors_daily.csv").exists()


def test_constant_risk_free_placeholder_is_rejected(tmp_path):
    df = _iima_raw()
    df["RF"] = 0.024                                  # fingerprint of the old synthetic fallback
    df.to_csv(tmp_path / "raw_iima_four_factors_daily.csv", index=False)
    with pytest.raises(RuntimeError, match="synthetic"):
        dl.acquire_raw_iima_factors("2023-01-01", "2023-12-31", str(tmp_path))


def test_factor_file_missing_columns_is_rejected(tmp_path):
    _iima_raw().drop(columns=["WML"]).to_csv(tmp_path / "raw_iima_four_factors_daily.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        dl.acquire_raw_iima_factors("2023-01-01", "2023-12-31", str(tmp_path))


def test_real_factor_file_is_genuine_iima_data():
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "raw_iima_four_factors_daily.csv")
    assert raw["RF"].dropna().nunique() > 100          # time-varying T-bill series, not a constant
    assert raw["Date"].min() <= "1994-01-01" and raw["Date"].max() >= "2023-12-29"


# ------------------------------------------------------------------ alignment ---------------------

def test_missing_nav_day_return_is_not_paired_with_one_day_market_return():
    funds, bench, factors, _ = _frames(fund_dates=[d for d in DATES if d != DATES[3]])   # fund lacks 2023-01-05
    aligned = dl.create_aligned_dataset(funds, bench, factors)
    # fund return on 01-06 spans 01-04 -> 01-06 (two market steps) and must be dropped
    assert set(aligned["Date"]) == {"2023-01-03", "2023-01-04", "2023-01-09"}
    assert aligned.attrs["n_dropped_async"] == 1


def test_benchmark_gap_is_also_detected():
    funds, bench, factors, _ = _frames(bench_dates=[d for d in DATES if d != DATES[2]])   # Yahoo lacks 2023-01-04
    aligned = dl.create_aligned_dataset(funds, bench, factors)
    assert "2023-01-05" not in set(aligned["Date"])      # bench return on 01-05 spans two days
    assert "2023-01-04" not in set(aligned["Date"])      # no bench observation at all


def test_no_forward_fill_missing_dates_stay_missing():
    funds, bench, factors, _ = _frames(factor_dates=[d for d in DATES if d != DATES[1]])
    aligned = dl.create_aligned_dataset(funds, bench, factors)
    assert "2023-01-03" not in set(aligned["Date"])
    assert aligned[["daily_return", "nifty500_price_return", "Mkt-RF", "SMB", "HML", "Mom", "RF"]].notna().all().all()


# ------------------------------------------------------------------ market proxy ------------------

def test_default_market_proxy_is_iima_total_return_market():
    funds, bench, factors, _ = _frames()
    aligned = dl.create_aligned_dataset(funds, bench, factors)
    assert set(aligned["market_proxy"]) == {"iima_market"}
    np.testing.assert_array_equal(aligned["excess_market_return"].values, aligned["Mkt-RF"].values)
    np.testing.assert_allclose(aligned["market_return"], aligned["Mkt-RF"] + aligned["RF"])


def test_price_index_return_is_kept_as_labelled_sensitivity_series():
    funds, bench, factors, _ = _frames()
    aligned = dl.create_aligned_dataset(funds, bench, factors)
    np.testing.assert_allclose(aligned["excess_nifty500_price_return"],
                               aligned["nifty500_price_return"] - aligned["RF"])


def test_genuine_tri_series_is_preferred_when_supplied():
    funds, bench, factors, tri = _frames(tri_dates=DATES)
    aligned = dl.create_aligned_dataset(funds, bench, factors, tri)
    assert set(aligned["market_proxy"]) == {"nifty500_tri"}
    np.testing.assert_allclose(aligned["market_return"], aligned["nifty500_tri_return"])


def test_explicit_price_proxy_and_unknown_proxy():
    funds, bench, factors, _ = _frames()
    aligned = dl.create_aligned_dataset(funds, bench, factors, market_proxy="nifty500_price")
    np.testing.assert_allclose(aligned["market_return"], aligned["nifty500_price_return"])
    with pytest.raises(ValueError, match="Unknown market_proxy"):
        dl.create_aligned_dataset(funds, bench, factors, market_proxy="bogus")


def test_load_nifty500_tri_parses_nse_style_file(tmp_path):
    (tmp_path / dl.TRI_FILENAME).write_text(
        "Date,Total Returns Index\n03 Jan 2023,26000.5\n02 Jan 2023,25900.25\n", encoding="utf-8")
    tri = dl.load_nifty500_tri(str(tmp_path))
    assert list(tri["Date"]) == ["2023-01-02", "2023-01-03"]
    assert dl.load_nifty500_tri(str(tmp_path / "missing")) is None


# ------------------------------------------------------------------ preprocessing -----------------

def test_prev_date_is_recorded_for_synchronisation():
    funds, _, factors, _ = _frames()
    assert funds.loc[1, "prev_date"] == "2023-01-02"
    assert pd.isna(funds.loc[0, "prev_date"])
    assert factors.loc[2, "prev_date"] == "2023-01-03"


def test_iima_market_return_is_excess_plus_rf_in_decimals():
    factors = dl.preprocess_iima_factors(_iima_raw(), "2023-01-01", "2023-12-31")
    np.testing.assert_allclose(factors["iima_market_return"], factors["Mkt-RF"] + factors["RF"])
    assert factors["Mkt-RF"].abs().max() < 0.2        # percent -> decimal conversion happened


def test_validation_flags_non_positive_nav():
    bad = pd.DataFrame({"Date": ["2023-01-02", "2023-01-03"], "Adj Close": [100.0, -1.0], "symbol": "F"})
    report = dl.validate_dataframe(bad, "bad", ["Date", "Adj Close"], id_col="symbol", price_cols=["Adj Close"])
    assert report["is_valid"] is False
    assert any("non-positive" in e for e in report["validation_errors"])


# ------------------------------------------------------------------ metadata ----------------------

def test_stale_screening_metadata_is_regenerated(tmp_path):
    stale = [{"ticker": "118955", "lead_manager": "Roshi Jain", "tenure_start": "2022-01-14", "metadata_version": "1.0.0"}]
    (tmp_path / "screening_metadata.json").write_text(json.dumps(stale), encoding="utf-8")
    dl.initialize_screening_metadata(str(tmp_path), dl.TARGET_FUNDS)
    records = json.loads((tmp_path / "screening_metadata.json").read_text(encoding="utf-8"))
    assert len(records) == 12
    hdfc = next(r for r in records if r["ticker"] == "118955")
    assert hdfc["tenure_start"] == "2022-07-29"
    assert all(r["metadata_version"] == dl.METADATA_VERSION for r in records)
    assert all("NOT_INDEPENDENTLY_VERIFIED" in r["verification_status"] for r in records)


def test_current_screening_metadata_is_left_untouched(tmp_path):
    dl.initialize_screening_metadata(str(tmp_path), dl.TARGET_FUNDS)
    path = tmp_path / "screening_metadata.json"
    before = path.read_bytes()
    dl.initialize_screening_metadata(str(tmp_path), dl.TARGET_FUNDS)
    assert path.read_bytes() == before


def test_fund_metadata_covers_all_target_funds():
    assert set(dl.FUND_METADATA) == set(dl.TARGET_FUNDS) and len(dl.TARGET_FUNDS) == 12
    for meta in dl.FUND_METADATA.values():
        assert pd.Timestamp(meta["tenure_start"]) < pd.Timestamp("2026-01-01")


# ------------------------------------------------------------------ integration (offline) ---------

def test_pipeline_runs_offline_from_cached_raw_files(tmp_path):
    shutil.copytree(PROJECT_ROOT / "data" / "raw", tmp_path / "data" / "raw")
    summary = dl.run_data_pipeline(str(tmp_path))
    assert summary["status"] == "SUCCESS" and summary["funds_count"] == 12
    assert summary["market_proxy"] == "iima_market"
    assert summary["n_dropped_async"] >= 0 and summary["obs_per_fund_min"] > 1150
    aligned = pd.read_csv(tmp_path / "data" / "processed" / "aligned_analysis_dataset.csv")
    assert not aligned[["daily_return", "excess_fund_return", "excess_market_return", "SMB", "HML", "Mom"]].isna().any().any()
    report = json.loads((tmp_path / "data" / "processed" / "data_validation_report.json").read_text(encoding="utf-8"))
    assert all(r["is_valid"] for r in report)
