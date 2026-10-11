"""
Tests for app.py (Streamlit Fund Manager Skill Detector Dashboard)
=================================================================
Validates app data loader, file detection, schema requirements, fund dynamic loading,
score bounds, error handling for missing files/columns, and source immutability.
"""

import os
import tempfile
import pandas as pd
import pytest
import app


class TestAppDashboard:
    """Test suite for app.py dashboard logic and data loading."""

    def test_01_app_imports_successfully(self):
        """Verify that app module imports without errors."""
        assert hasattr(app, "load_dashboard_data")
        assert hasattr(app, "main")
        assert hasattr(app, "fmt_pct")

    def test_02_required_data_files_detected(self):
        """Verify that all required data files exist in data/processed."""
        proc_dir = os.path.join(".", "data", "processed")
        required_files = [
            "final_evaluation.csv",
            "final_evaluation.json",
            "capm_results.csv",
            "bootstrap_results.csv",
            "factor_model_results.csv",
            "persistence_summary.csv",
            "persistence_period_results.csv",
        ]
        for f in required_files:
            path = os.path.join(proc_dir, f)
            assert os.path.exists(path), f"Required file missing: {path}"

    def test_03_expected_columns_exist(self):
        """Verify expected columns exist in final_evaluation.csv."""
        final_csv = os.path.join(".", "data", "processed", "final_evaluation.csv")
        df = pd.read_csv(final_csv)
        expected_cols = [
            "symbol",
            "final_category",
            "total_score",
            "evaluation_version",
            "final_explanation",
            "capm_alpha_annual_pct",
            "factor_alpha_annual_pct",
            "alpha_reduction_pct",
            "bootstrap_p_value",
            "positive_alpha_persistence_rate",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_04_all_funds_load_dynamically(self):
        """Verify that all 12 candidate funds are loaded dynamically by data loader."""
        data, err = app.load_dashboard_data(".")
        assert err is None
        assert data is not None
        df = data["final_eval"]
        assert len(df) == 12
        assert df["symbol"].nunique() == 12
        expected_symbols = {
            "AGTHX", "FCNTX", "DODGX", "FDGRX", "TRBCX", "AIVSX",
            "FLPSX", "AMECX", "PRHSX", "PAGRX", "FMAGX", "MIGFX"
        }
        assert set(df["symbol"]) == expected_symbols

    def test_05_category_counts_calculated_dynamically(self):
        """Verify category counts are dynamically computed."""
        data, err = app.load_dashboard_data(".")
        assert err is None
        df = data["final_eval"]
        cat_counts = df["final_category"].value_counts().to_dict()
        assert sum(cat_counts.values()) == 12
        # Invariant (not a pinned answer): every fund carries one of the six shared-rule categories.
        known = {
            "Strong Evidence of Skill", "Strong Skill Candidate (Preliminary)", "Moderate Evidence",
            "Positive but Inconclusive", "Weak/Inconclusive", "Likely Style/Luck Driven",
        }
        assert set(cat_counts) <= known, f"unexpected categories: {set(cat_counts) - known}"

    def test_06_fund_selection_retrieval(self):
        """Verify fund selection retrieves correct data record."""
        data, err = app.load_dashboard_data(".")
        assert err is None
        df = data["final_eval"]
        for sym in df["symbol"]:
            fund_row = df[df["symbol"] == sym]
            assert len(fund_row) == 1
            assert fund_row.iloc[0]["symbol"] == sym

    def test_07_score_fields_read_correctly_and_bounded(self):
        """Verify score fields are valid floats strictly bounded within [0, 100]."""
        data, err = app.load_dashboard_data(".")
        assert err is None
        df = data["final_eval"]
        scores = df["total_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 100.0).all()
        assert scores.min() == 0.0
        assert scores.max() == 62.0

    def test_08_dashboard_does_not_modify_source_files(self):
        """Verify load_dashboard_data is strictly read-only and preserves mtimes."""
        proc_dir = os.path.join(".", "data", "processed")
        files_to_check = [
            os.path.join(proc_dir, "final_evaluation.csv"),
            os.path.join(proc_dir, "capm_results.csv"),
            os.path.join(proc_dir, "bootstrap_results.csv"),
            os.path.join(proc_dir, "factor_model_results.csv"),
            os.path.join(proc_dir, "persistence_summary.csv"),
        ]
        mtimes_before = {f: os.path.getmtime(f) for f in files_to_check}
        
        # Call data loader multiple times
        data1, _ = app.load_dashboard_data(".")
        data2, _ = app.load_dashboard_data(".")
        
        mtimes_after = {f: os.path.getmtime(f) for f in files_to_check}
        assert mtimes_before == mtimes_after

    def test_09_missing_file_handling(self):
        """Verify graceful error reporting when a data file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data, err = app.load_dashboard_data(tmpdir)
            assert data is None
            assert err is not None
            assert "Missing required processed data files" in err

    def test_10_missing_column_handling(self):
        """Verify error reporting when a required column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc_dir = os.path.join(tmpdir, "data", "processed")
            os.makedirs(proc_dir, exist_ok=True)
            
            # Create dummy CSVs with missing required symbol/category columns
            incomplete_df = pd.DataFrame({"symbol": ["FDGRX"]})  # missing final_category etc.
            incomplete_df.to_csv(os.path.join(proc_dir, "final_evaluation.csv"), index=False)
            
            for name in ["capm_results.csv", "bootstrap_results.csv", "factor_model_results.csv", "persistence_summary.csv", "persistence_period_results.csv", "final_evaluation.json"]:
                pd.DataFrame({"symbol": ["FDGRX"]}).to_csv(os.path.join(proc_dir, name), index=False)
                
            data, err = app.load_dashboard_data(tmpdir)
            assert data is None
            assert err is not None
            assert "missing required columns" in err
