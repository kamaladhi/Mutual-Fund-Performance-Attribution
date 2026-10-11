"""
Tests for app.py (Streamlit Fund Manager Skill Detector Dashboard)
=================================================================
Validates app data loader, file detection, schema requirements, fund dynamic loading,
score bounds, error handling for missing files/columns, and source immutability.
"""

import os
import re
import tempfile
from pathlib import Path
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
            "122639", "118955", "120586", "118834", "119598", "120166",
            "118989", "120505", "118668", "118778", "120828", "119071"
        }
        assert set(df["symbol"]) == expected_symbols

    def test_05_category_counts_calculated_dynamically(self):
        """Verify category counts are dynamically computed."""
        data, err = app.load_dashboard_data(".")
        assert err is None
        df = data["final_eval"]
        cat_counts = df["final_category"].value_counts().to_dict()
        valid = {"Strong Evidence of Skill", "Strong Skill Candidate (Preliminary)", "Moderate Evidence", "Positive but Inconclusive", "Weak/Inconclusive", "Likely Style/Luck Driven"}
        assert set(cat_counts) <= valid
        assert sum(cat_counts.values()) == 12

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
        assert scores.notna().all()

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


class TestDashboardRegressions:
    """Regression tests for defects found in the review of the original dashboard."""

    @staticmethod
    def _source():
        return (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")

    def test_11_css_has_no_blanket_background_rule(self):
        """A background-color on bare div/span rules blanked every st.dataframe canvas."""
        css = re.search(r"<style>(.*?)</style>", self._source(), re.S).group(1)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            tokens = [t.strip() for t in selector.split(",")]
            if "background" in body:
                assert not any(t in ("div", "span", "p", "label", "html", "body") or t.startswith('[class*=')
                               for t in tokens), f"blanket background rule on: {selector.strip()}"

    def test_12_no_false_claims_left_in_app_text(self):
        src = self._source()
        assert "NIFTY 500 TRI (^CRSLDX)" not in src          # ^CRSLDX is a PRICE index
        assert "11-condition decision matrix" not in src      # never matched the real rubric
        assert "non-normal p-value inflation" not in src
        assert "1,227 common days" not in src                 # sample facts are derived from the data
        # the bootstrap test is two-sided with H0: alpha = 0, so a one-sided "H0: alpha <= 0" claim is wrong
        assert not re.search(r"H_0:\s*\\+alpha\s*\\+le", src)

    def test_13_formatting_helpers(self):
        assert app.fmt_p(0.00004) == "<0.0001" and app.fmt_p(0.0123456) == "0.0123"
        assert app.fmt_p(float("nan")) == "N/A"
        assert "NEGATIVE" in app.p_label(0.01, alpha_positive=False)
        assert app.p_label(0.2) == "not significant" and app.p_label(0.07) == "marginal (10%)"
        assert app.fmt_pct(1.234, include_sign=True) == "+1.23%" and app.fmt_pct(float("nan")) == "N/A"

    def test_14_sidebar_facts_come_from_the_data(self):
        data, err = app.load_dashboard_data(".")
        assert err is None
        facts = app.build_sidebar_facts(data)
        assert facts["sample_start"] == str(data["pers_periods"]["start_date"].min())
        assert facts["n_funds"] == 12 and facts["block_length"] == 30
        assert facts["market_proxy"] in app.MARKET_PROXY_LABELS

    def test_15_optional_artifacts_loaded(self):
        data, err = app.load_dashboard_data(".")
        assert err is None
        for key in ("block_sensitivity", "benchmark_sensitivity", "descriptive", "capm_nifty_price", "manifest"):
            assert data[key] is not None, key

    def test_16_dashboard_renders_for_every_fund_without_errors(self, monkeypatch):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest
        monkeypatch.chdir(Path(__file__).parent.parent)
        at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
        assert not at.exception, [e.value for e in at.exception]
        assert [t.label for t in at.tabs] == ["CAPM REGRESSION", "BLOCK BOOTSTRAP RESAMPLING",
                                              "CARHART 4-FACTOR DECOMPOSITION", "SUBPERIOD PERSISTENCE",
                                              "UNIVERSE LEADERBOARD"]
        options = at.sidebar.selectbox[0].options
        assert len(options) == 12
        seen_attribution_warning = False
        for i, option in enumerate(options):
            at.sidebar.selectbox[0].select_index(i).run()
            assert not at.exception, (option, [e.value for e in at.exception])
            labels = {m.label: m.value for m in at.metric}
            assert "Evidence Index" in labels and "Alpha p-value (HAC)" in labels
            if option.startswith("118955"):                      # HDFC Flexi Cap: manager arrived mid-2022
                seen_attribution_warning = any("Attribution caveat" in w.value for w in at.warning)
        assert seen_attribution_warning
