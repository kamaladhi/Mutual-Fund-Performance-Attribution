"""
End-to-end reproducibility test: run the complete pipeline (offline, from the cached raw files) in a
temporary copy of the project with a small bootstrap, twice, and verify outputs and determinism.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import run_pipeline

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="module")
def pipeline_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(PROJECT_ROOT / "data" / "raw", root / "data" / "raw")
    run_pipeline.main(str(root), n_bootstrap=40, block_length=20, seed=7)
    return root


def test_pipeline_produces_every_artifact(pipeline_root):
    expected = [
        "data/processed/aligned_analysis_dataset.csv", "data/processed/capm_results.csv",
        "data/processed/capm_sensitivity_nifty500_price.csv", "data/processed/bootstrap_results.csv",
        "data/processed/bootstrap_block_sensitivity.csv", "data/processed/factor_model_results.csv",
        "data/processed/persistence_period_results.csv", "data/processed/persistence_summary.csv",
        "data/processed/descriptive_statistics.csv", "data/processed/final_evaluation.csv",
        "data/processed/benchmark_sensitivity.csv", "data/processed/run_manifest.json",
        "results/tables/skill_leaderboard.csv", "results/tables/alpha_comparison.csv",
        "results/figures/evidence_index.png", "results/figures/alpha_comparison.png",
        "results/figures/subperiod_alpha.png", "results/figures/block_length_sensitivity.png",
    ]
    missing = [p for p in expected if not (pipeline_root / p).exists()]
    assert not missing, missing


def test_pipeline_outputs_are_consistent(pipeline_root):
    final = pd.read_csv(pipeline_root / "data" / "processed" / "final_evaluation.csv", dtype={"symbol": str})
    assert len(final) == 12 and final["total_score"].between(0, 100).all()
    assert set(final["evaluation_version"]) == {"2.3.0"}
    boot = pd.read_csv(pipeline_root / "data" / "processed" / "bootstrap_results.csv")
    assert (boot["n_bootstrap"] == 40).all() and (boot["block_length"] == 20).all() and (boot["seed"] == 7).all()
    sens = pd.read_csv(pipeline_root / "data" / "processed" / "benchmark_sensitivity.csv", dtype={"symbol": str})
    assert set(sens["market_proxy"]) == {"iima_market", "nifty500_price", "nifty500_price_plus_assumed_dividend_yield"}
    assert sens.groupby("market_proxy").size().eq(12).all()
    manifest = json.loads((pipeline_root / "data" / "processed" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["market_proxy"] == "iima_market" and manifest["bootstrap"]["n_bootstrap"] == 40


def test_primary_proxy_row_of_sensitivity_table_equals_the_main_result(pipeline_root):
    proc = pipeline_root / "data" / "processed"
    final = pd.read_csv(proc / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
    sens = pd.read_csv(proc / "benchmark_sensitivity.csv", dtype={"symbol": str})
    primary = sens[sens["market_proxy"] == "iima_market"].set_index("symbol")
    assert (primary["final_category"] == final.loc[primary.index, "final_category"]).all()
    assert (primary["total_score"] - final.loc[primary.index, "total_score"]).abs().max() < 1e-9


def test_pipeline_is_deterministic(pipeline_root, tmp_path):
    second = tmp_path / "again"
    shutil.copytree(PROJECT_ROOT / "data" / "raw", second / "data" / "raw")
    run_pipeline.main(str(second), n_bootstrap=40, block_length=20, seed=7)
    a = (pipeline_root / "data" / "processed" / "final_evaluation.csv").read_text(encoding="utf-8")
    b = (second / "data" / "processed" / "final_evaluation.csv").read_text(encoding="utf-8")
    assert a == b
