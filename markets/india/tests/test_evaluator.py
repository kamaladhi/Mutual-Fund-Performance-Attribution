"""
test_evaluator.py -- Comprehensive tests for the Task 8 Evaluation Engine
==========================================================================
Tests are organised into groups:

  Group A  -- Input loading & validation (tests 1-4)
  Group B  -- Evidence flag logic (tests 5-11)
  Group C  -- Scoring (tests 12-13)
  Group D  -- Category classification (tests 14-17)
  Group E  -- Explanation generation (tests 18-19)
  Group F  -- End-to-end evaluation (tests 20-22)

Tests use:
  * Real project files (``data/processed/*.csv``) for loading/integration tests.
  * Synthetic records for flag/score/category/explanation unit tests so that
    the tests are independent of the current 12-fund numerical results.

IMPORTANT: synthetic records are built in-memory; they do NOT touch or modify
any on-disk files.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Resolve the project root from the test-file location
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from src.evaluator import (
    CAT_MODERATE,
    CAT_STRONG,
    CAT_STYLE_LUCK,
    CAT_WEAK,
    EVALUATION_VERSION,
    INPUT_FILES,
    OUTPUT_CSV,
    OUTPUT_JSON,
    classify_fund,
    compute_evidence_flags,
    compute_scores,
    evaluate_fund,
    generate_explanation,
    load_and_validate_inputs,
    run_evaluation,
    _assert_columns_present,
    _assert_pvalue_range,
    _assert_rate_range,
    _assert_unique_symbols,
)


# ===========================================================================
# HELPERS -- synthetic row builders
# ===========================================================================

def _row(
    symbol="TEST",
    capm_alpha_annual_pct=2.0,
    capm_alpha_daily=2.0 / 252,
    capm_alpha_pvalue=0.03,
    capm_beta=1.0,
    capm_r_squared=0.90,
    bootstrap_p_value=0.04,
    bootstrap_significant=True,
    bootstrap_ci_lower=0.0001,
    bootstrap_ci_upper=0.0010,
    factor_alpha_annual_pct=1.5,
    factor_alpha_daily=1.5 / 252,
    factor_alpha_pvalue=0.04,
    factor_alpha_significant=True,
    factor_market_beta=0.95,
    factor_smb_beta=0.10,
    factor_hml_beta=-0.05,
    factor_momentum_beta=0.02,
    factor_r_squared=0.94,
    alpha_reduction_pct=25.0,
    positive_alpha_persistence_rate=0.75,
    significant_positive_alpha_persistence_rate=0.25,
    alpha_mean_annual_pct=1.5,
    alpha_median_annual_pct=1.8,
    alpha_std_annual_pct=2.0,
    alpha_positive_majority=True,
    alpha_positive_all_periods=False,
    capm_alpha_pvalue_bh=1.0,
    factor_alpha_pvalue_bh=1.0,
    bootstrap_p_value_bh_computed=1.0,
) -> pd.Series:
    """Return a synthetic merged-row Series with all evaluator-required fields."""
    return pd.Series({
        "symbol":                      symbol,
        "capm_alpha_annual_pct":       capm_alpha_annual_pct,
        "capm_alpha_daily":            capm_alpha_daily,
        "capm_alpha_pvalue":           capm_alpha_pvalue,
        "capm_beta":                   capm_beta,
        "capm_r_squared":              capm_r_squared,
        "bootstrap_p_value":           bootstrap_p_value,
        "bootstrap_significant":       bootstrap_significant,
        "bootstrap_ci_lower":          bootstrap_ci_lower,
        "bootstrap_ci_upper":          bootstrap_ci_upper,
        "factor_alpha_annual_pct":     factor_alpha_annual_pct,
        "factor_alpha_daily":          factor_alpha_daily,
        "factor_alpha_pvalue":         factor_alpha_pvalue,
        "factor_alpha_significant":    factor_alpha_significant,
        "factor_market_beta":          factor_market_beta,
        "factor_smb_beta":             factor_smb_beta,
        "factor_hml_beta":             factor_hml_beta,
        "factor_momentum_beta":        factor_momentum_beta,
        "factor_r_squared":            factor_r_squared,
        "alpha_reduction_pct":         alpha_reduction_pct,
        "positive_alpha_persistence_rate":             positive_alpha_persistence_rate,
        "significant_positive_alpha_persistence_rate": significant_positive_alpha_persistence_rate,
        "alpha_mean_annual_pct":       alpha_mean_annual_pct,
        "alpha_median_annual_pct":     alpha_median_annual_pct,
        "alpha_std_annual_pct":        alpha_std_annual_pct,
        "alpha_positive_majority":     alpha_positive_majority,
        "alpha_positive_all_periods":  alpha_positive_all_periods,
        "capm_alpha_pvalue_bh": capm_alpha_pvalue_bh,
        "factor_alpha_pvalue_bh": factor_alpha_pvalue_bh,
        "bootstrap_p_value_bh_computed": bootstrap_p_value_bh_computed,
    })


# ===========================================================================
# GROUP A -- INPUT LOADING & VALIDATION
# ===========================================================================

class TestInputLoading:
    """Tests 1-4: file loading, join integrity, duplicate/missing detection."""

    def test_01_all_four_source_files_load(self):
        """Test 1: All four source CSVs can be loaded without error."""
        merged = load_and_validate_inputs(PROJECT_ROOT)
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) > 0, "Merged DataFrame must not be empty."

    def test_02_multi_file_join_produces_correct_fund_set(self):
        """Test 2: The join produces exactly the expected 12 fund symbols."""
        merged = load_and_validate_inputs(PROJECT_ROOT)
        expected = {
            "122639", "118955", "120586", "118834", "119598", "120166",
            "118989", "120505", "118668", "118778", "120828", "119071"
        }
        assert set(merged["symbol"].tolist()) == expected

    def test_03_missing_fund_detection(self):
        """Test 3: Validator raises ValueError when a fund is missing from a source."""
        # Build a minimal 2-fund CAPM df and a 1-fund bootstrap df -> join fails
        capm_df = pd.DataFrame({
            "symbol": ["FUND_A", "FUND_B"],
            "capm_alpha_daily": [0.0001, 0.0002],
            "bootstrap_p_value": [np.nan, np.nan],  # missing in bootstrap
            "factor_alpha_daily": [np.nan, np.nan],
            "positive_alpha_persistence_rate": [np.nan, np.nan],
        })
        # We test _assert_no_missing_numerics indirectly via load, but since
        # we cannot mock file reads easily, we test validation helpers directly.
        from src.evaluator import _assert_no_missing_numerics
        with pytest.raises(ValueError, match="Missing values"):
            _assert_no_missing_numerics(capm_df, ["bootstrap_p_value"], "test")

    def test_04_duplicate_fund_detection(self):
        """Test 4: Validator raises ValueError for duplicate symbols."""
        df_with_dupes = pd.DataFrame({
            "symbol": ["FUND_A", "FUND_A"],
            "value":  [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="Duplicate symbols"):
            _assert_unique_symbols(df_with_dupes, "test_source")

    def test_04b_pvalue_range_validation(self):
        """Test 4b: p-value outside [0,1] raises ValueError."""
        df = pd.DataFrame({"symbol": ["BAD"], "capm_alpha_pvalue": [1.5]})
        with pytest.raises(ValueError, match="out of"):
            _assert_pvalue_range(df, ["capm_alpha_pvalue"], "test")

    def test_04c_rate_range_validation(self):
        """Test 4c: persistence rate outside [0,1] raises ValueError."""
        df = pd.DataFrame({"symbol": ["BAD"], "positive_alpha_persistence_rate": [-0.1]})
        with pytest.raises(ValueError, match="out of"):
            _assert_rate_range(df, ["positive_alpha_persistence_rate"], "test")

    def test_04d_missing_column_detection(self):
        """Test 4d: Missing required column raises ValueError."""
        df = pd.DataFrame({"symbol": ["F"], "wrong_col": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            _assert_columns_present(df, ["symbol", "capm_alpha_daily"], "test")


# ===========================================================================
# GROUP B -- EVIDENCE FLAG LOGIC
# ===========================================================================

class TestEvidenceFlags:
    """Tests 5-11: verify each Boolean evidence flag fires correctly."""

    def test_05_positive_alpha_flag_true(self):
        """Test 5a: capm_positive is True when CAPM annual alpha > 0."""
        flags = compute_evidence_flags(_row(capm_alpha_annual_pct=1.0))
        assert flags["capm_positive"] is True

    def test_05_positive_alpha_flag_false(self):
        """Test 5b: capm_positive is False when CAPM annual alpha <= 0."""
        flags = compute_evidence_flags(_row(capm_alpha_annual_pct=-0.5))
        assert flags["capm_positive"] is False

    def test_05_positive_alpha_flag_zero(self):
        """Test 5c: capm_positive is False when alpha exactly equals 0."""
        flags = compute_evidence_flags(_row(capm_alpha_annual_pct=0.0))
        assert flags["capm_positive"] is False

    def test_06_capm_significance_flag(self):
        """Test 6: capm_significant fires at p < 0.05, not at p >= 0.05."""
        assert compute_evidence_flags(_row(capm_alpha_pvalue=0.04))["capm_significant"] is True
        assert compute_evidence_flags(_row(capm_alpha_pvalue=0.05))["capm_significant"] is False
        assert compute_evidence_flags(_row(capm_alpha_pvalue=0.10))["capm_significant"] is False

    def test_07_bootstrap_significance_from_flag(self):
        """Test 7a: bootstrap_significant fires when significant_at_5pct flag is True."""
        flags = compute_evidence_flags(
            _row(bootstrap_significant=True, bootstrap_p_value=0.30)
        )
        assert flags["bootstrap_significant"] is True

    def test_07_bootstrap_significance_from_pvalue(self):
        """Test 7b: bootstrap_significant fires when p_value < 0.05 (flag=False)."""
        flags = compute_evidence_flags(
            _row(bootstrap_significant=False, bootstrap_p_value=0.03)
        )
        assert flags["bootstrap_significant"] is True

    def test_07_bootstrap_significance_false(self):
        """Test 7c: bootstrap_significant is False when both flag=False and p>=0.05."""
        flags = compute_evidence_flags(
            _row(bootstrap_significant=False, bootstrap_p_value=0.50)
        )
        assert flags["bootstrap_significant"] is False

    def test_08_bootstrap_ci_positive(self):
        """Test 8: bootstrap_ci_positive requires ci_lower > 0."""
        assert compute_evidence_flags(_row(bootstrap_ci_lower=0.0001))["bootstrap_ci_positive"] is True
        assert compute_evidence_flags(_row(bootstrap_ci_lower=0.0))["bootstrap_ci_positive"] is False
        assert compute_evidence_flags(_row(bootstrap_ci_lower=-0.001))["bootstrap_ci_positive"] is False

    def test_09_factor_survival(self):
        """Test 9: factor_alpha_survives requires positive AND significant factor alpha."""
        # Both positive and significant -> survives
        f = compute_evidence_flags(_row(factor_alpha_annual_pct=1.0, factor_alpha_pvalue=0.03, factor_alpha_significant=True))
        assert f["factor_alpha_survives"] is True
        # Positive but not significant -> does not survive
        f = compute_evidence_flags(_row(factor_alpha_annual_pct=1.0, factor_alpha_pvalue=0.80, factor_alpha_significant=False))
        assert f["factor_alpha_survives"] is False
        # Negative -> does not survive
        f = compute_evidence_flags(_row(factor_alpha_annual_pct=-0.5, factor_alpha_pvalue=0.03, factor_alpha_significant=True))
        assert f["factor_alpha_survives"] is False

    def test_10_persistence_flags(self):
        """Test 10: persistence flags reflect alpha_positive_majority / all_periods."""
        f_all  = compute_evidence_flags(_row(alpha_positive_majority=True,  alpha_positive_all_periods=True))
        f_maj  = compute_evidence_flags(_row(alpha_positive_majority=True,  alpha_positive_all_periods=False))
        f_none = compute_evidence_flags(_row(alpha_positive_majority=False, alpha_positive_all_periods=False))
        assert f_all["persistent_all_periods"]       is True
        assert f_all["persistent_positive_majority"] is True
        assert f_maj["persistent_positive_majority"] is True
        assert f_maj["persistent_all_periods"]       is False
        assert f_none["persistent_positive_majority"] is False
        assert f_none["persistent_all_periods"]       is False

    def test_11_style_luck_detection(self):
        """Test 11: style_explanation requires capm_positive AND alpha_materially_reduced."""
        # capm_positive AND reduction >= 50% -> style_explanation True
        f = compute_evidence_flags(_row(capm_alpha_annual_pct=2.0, alpha_reduction_pct=60.0))
        assert f["style_explanation"] is True
        assert f["alpha_materially_reduced"] is True
        # capm_negative -> style_explanation False even with high reduction
        f = compute_evidence_flags(_row(capm_alpha_annual_pct=-1.0, alpha_reduction_pct=80.0))
        assert f["style_explanation"] is False
        # Low reduction -> False
        f = compute_evidence_flags(_row(capm_alpha_annual_pct=2.0, alpha_reduction_pct=30.0))
        assert f["style_explanation"] is False


# ===========================================================================
# GROUP C -- SCORING
# ===========================================================================

class TestScoring:
    """Tests 12-13: score component bounds and total clamping."""

    def test_12_score_components_within_bounds(self):
        """Test 12: all score components stay within their documented bounds."""
        # Test across a variety of synthetic records
        test_cases = [
            _row(capm_alpha_annual_pct=5.0, capm_alpha_pvalue=0.005,
                 bootstrap_significant=True, bootstrap_ci_lower=0.001,
                 factor_alpha_annual_pct=3.0, factor_alpha_pvalue=0.01,
                 factor_alpha_significant=True, alpha_reduction_pct=10.0,
                 alpha_positive_all_periods=True,
                 significant_positive_alpha_persistence_rate=0.75),
            _row(capm_alpha_annual_pct=-1.0, capm_alpha_pvalue=0.80,
                 bootstrap_significant=False, bootstrap_ci_lower=-0.002,
                 factor_alpha_annual_pct=-0.5, factor_alpha_pvalue=0.90,
                 factor_alpha_significant=False, alpha_reduction_pct=110.0,
                 alpha_positive_majority=False, alpha_positive_all_periods=False),
            _row(capm_alpha_annual_pct=0.5, capm_alpha_pvalue=0.08,
                 bootstrap_significant=False, bootstrap_ci_lower=-0.0001,
                 factor_alpha_annual_pct=0.2, factor_alpha_pvalue=0.20,
                 factor_alpha_significant=False, alpha_reduction_pct=60.0,
                 alpha_positive_majority=True, alpha_positive_all_periods=False),
        ]
        bounds = {
            "performance_score":     (0, 20),
            "significance_score":    (0, 25),
            "factor_score":          (0, 25),
            "persistence_score":     (0, 20),
            "style_luck_adjustment": (-10, 10),
        }
        for row in test_cases:
            flags  = compute_evidence_flags(row)
            scores = compute_scores(row, flags)
            for key, (lo, hi) in bounds.items():
                assert lo <= scores[key] <= hi, (
                    f"{key}={scores[key]} outside [{lo},{hi}] for symbol={row['symbol']}"
                )

    def test_13_total_score_within_0_100(self):
        """Test 13: total_score is always in [0, 100] across all 12 real funds."""
        merged = load_and_validate_inputs(PROJECT_ROOT)
        for _, row in merged.iterrows():
            flags  = compute_evidence_flags(row)
            scores = compute_scores(row, flags)
            assert 0.0 <= scores["total_score"] <= 100.0, (
                f"total_score={scores['total_score']} for {row['symbol']}"
            )


# ===========================================================================
# GROUP D -- CATEGORY CLASSIFICATION
# ===========================================================================

class TestCategoryClassification:
    """Tests 14-17: verify all four categories and precedence logic."""

    def _classify(self, row: pd.Series) -> str:
        flags  = compute_evidence_flags(row)
        scores = compute_scores(row, flags)
        return classify_fund(flags, scores, row)

    def test_14_strong_category(self):
        """Test 14: Strong Evidence of Skill requires all five criteria."""
        strong_row = _row(
            capm_alpha_annual_pct=4.0,
            capm_alpha_pvalue=0.02,
            bootstrap_significant=True,
            bootstrap_p_value=0.02,
            bootstrap_ci_lower=0.0002,
            factor_alpha_annual_pct=2.5,
            factor_alpha_pvalue=0.03,
            factor_alpha_significant=True,
            alpha_reduction_pct=20.0,  # below 75% -> no style override
            alpha_positive_majority=True,
            alpha_positive_all_periods=True,
            capm_alpha_pvalue_bh=0.02,
            factor_alpha_pvalue_bh=0.03,
            bootstrap_p_value_bh_computed=0.03,
        )
        assert self._classify(strong_row) == CAT_STRONG

    def test_15_moderate_category_positive_significant_capm(self):
        """Test 15: Moderate fired when CAPM is positive+significant but factor does not survive."""
        mod_row = _row(
            capm_alpha_annual_pct=2.0,
            capm_alpha_pvalue=0.04,
            bootstrap_significant=False,
            bootstrap_ci_lower=-0.001,
            factor_alpha_annual_pct=-0.5,  # factor alpha negative -> not survives
            factor_alpha_pvalue=0.80,
            factor_alpha_significant=False,
            alpha_reduction_pct=120.0,     # but > 75% -> style kicks in first?
            alpha_positive_majority=False,
        )
        # Because reduction > 75% AND factor not positive+significant -> STYLE overrides
        assert self._classify(mod_row) == CAT_STYLE_LUCK

        # Now set reduction < 75% -> Moderate should fire
        mod_row2 = _row(
            capm_alpha_annual_pct=2.0,
            capm_alpha_pvalue=0.04,
            bootstrap_significant=False,
            bootstrap_ci_lower=-0.001,
            factor_alpha_annual_pct=0.5,   # positive but not sig
            factor_alpha_pvalue=0.20,
            factor_alpha_significant=False,
            alpha_reduction_pct=50.0,       # < 75% -> no style override
            alpha_positive_majority=False,
        )
        assert self._classify(mod_row2) == CAT_MODERATE

    def test_16_weak_inconclusive_category(self):
        """Test 16: Weak/Inconclusive is the default when no strong/moderate/style criteria fire."""
        weak_row = _row(
            capm_alpha_annual_pct=0.3,   # positive but tiny
            capm_alpha_pvalue=0.40,      # not significant
            bootstrap_significant=False,
            bootstrap_p_value=0.60,
            bootstrap_ci_lower=-0.001,
            factor_alpha_annual_pct=0.1, # positive but not sig
            factor_alpha_pvalue=0.50,
            factor_alpha_significant=False,
            alpha_reduction_pct=40.0,    # < 75% -> no style
            alpha_positive_majority=False,
            alpha_positive_all_periods=False,
        )
        # capm_positive is True, bootstrap_significant is False,
        # capm_significant is False -> moderate condition fails
        # factor_alpha_survives is False
        # persistent_positive_majority is False
        assert self._classify(weak_row) == CAT_WEAK

    def test_17_style_luck_category_precedence(self):
        """Test 17: Likely Style/Luck Driven overrides even a 'strong-looking' fund
        when alpha_reduction_pct >= 75 and factor alpha does not survive."""
        style_row = _row(
            capm_alpha_annual_pct=10.0,   # very high CAPM alpha
            capm_alpha_pvalue=0.001,
            bootstrap_significant=True,
            bootstrap_p_value=0.001,
            bootstrap_ci_lower=0.002,
            factor_alpha_annual_pct=-0.5, # but factor alpha is negative
            factor_alpha_pvalue=0.90,
            factor_alpha_significant=False,
            alpha_reduction_pct=80.0,     # >= 75 -> triggers style
            alpha_positive_majority=True,
        )
        assert self._classify(style_row) == CAT_STYLE_LUCK

    def test_17b_style_luck_does_not_fire_when_factor_survives(self):
        """Test 17b: Style/Luck does NOT fire when factor alpha is positive+significant."""
        row = _row(
            capm_alpha_annual_pct=5.0,
            capm_alpha_pvalue=0.02,
            bootstrap_significant=True,
            bootstrap_p_value=0.02,
            bootstrap_ci_lower=0.0002,
            factor_alpha_annual_pct=2.0,
            factor_alpha_pvalue=0.03,
            factor_alpha_significant=True,
            alpha_reduction_pct=80.0,    # > 75% but factor survives
            alpha_positive_majority=True,
        )
        # factor_alpha_positive=True AND significant=True -> style condition fails
        # Should NOT be CAT_STYLE_LUCK
        cat = self._classify(row)
        assert cat != CAT_STYLE_LUCK

    def test_16b_category_precedence_is_deterministic(self):
        """Test 16b: Identical inputs produce identical category on repeated calls."""
        row = _row()
        cats = [self._classify(row) for _ in range(10)]
        assert len(set(cats)) == 1, "Category is not deterministic."


# ===========================================================================
# GROUP E -- EXPLANATION GENERATION
# ===========================================================================

class TestExplanationGeneration:
    """Tests 18-19: explanation is deterministic and rule-based."""

    def _explain(self, row: pd.Series) -> str:
        flags  = compute_evidence_flags(row)
        scores = compute_scores(row, flags)
        cat    = classify_fund(flags, scores, row)
        return generate_explanation(cat, flags, scores, row)

    def test_18_explanation_is_deterministic(self):
        """Test 18: Identical inputs always produce identical explanation text."""
        row = _row()
        texts = [self._explain(row) for _ in range(10)]
        assert len(set(texts)) == 1

    def test_19_explanation_does_not_claim_causality(self):
        """Test 19: Explanation never uses forbidden causal language."""
        forbidden = [
            "the manager definitely has skill",
            "the manager is proven skilled",
            "caused by the manager",
            "proves",
        ]
        rows = [
            _row(capm_alpha_annual_pct=5.0, capm_alpha_pvalue=0.02,
                 bootstrap_significant=True, bootstrap_p_value=0.02,
                 bootstrap_ci_lower=0.0002, factor_alpha_annual_pct=2.5,
                 factor_alpha_pvalue=0.03, factor_alpha_significant=True,
                 alpha_reduction_pct=15.0, alpha_positive_majority=True),
            _row(capm_alpha_annual_pct=-1.0, capm_alpha_pvalue=0.50,
                 bootstrap_significant=False, bootstrap_ci_lower=-0.002,
                 factor_alpha_annual_pct=-0.5, factor_alpha_pvalue=0.90,
                 factor_alpha_significant=False, alpha_reduction_pct=200.0,
                 alpha_positive_majority=False),
            _row(),
        ]
        for row in rows:
            text = self._explain(row).lower()
            for phrase in forbidden:
                assert phrase.lower() not in text, (
                    f"Forbidden phrase '{phrase}' found in explanation for {row['symbol']}."
                )

    def test_19b_style_explanation_mentions_factor(self):
        """Test 19b: Style/Luck explanation mentions factor exposures."""
        style_row = _row(
            capm_alpha_annual_pct=3.0, capm_alpha_pvalue=0.01,
            factor_alpha_annual_pct=-0.2, factor_alpha_pvalue=0.90,
            factor_alpha_significant=False, alpha_reduction_pct=80.0,
            alpha_positive_majority=False,
        )
        text = self._explain(style_row)
        assert "factor" in text.lower()
        assert "Likely Style/Luck Driven" in text

    def test_19c_strong_explanation_contains_note(self):
        """Test 19c: Strong explanation always includes the uncertainty note."""
        strong_row = _row(
            capm_alpha_annual_pct=4.0, capm_alpha_pvalue=0.02,
            bootstrap_significant=True, bootstrap_p_value=0.02,
            bootstrap_ci_lower=0.0002, factor_alpha_annual_pct=2.5,
            factor_alpha_pvalue=0.03, factor_alpha_significant=True,
            alpha_reduction_pct=10.0, alpha_positive_majority=True,
            capm_alpha_pvalue_bh=0.02,
            factor_alpha_pvalue_bh=0.03,
            bootstrap_p_value_bh_computed=0.03,
        )
        text = self._explain(strong_row)
        assert "not a confirmed demonstration" in text or "does not establish" in text


# ===========================================================================
# GROUP F -- END-TO-END & INTEGRATION
# ===========================================================================

class TestEndToEnd:
    """Tests 20-22: full pipeline on real project files."""

    def test_20_multi_fund_evaluation_produces_one_row_per_fund(self, eval_root):
        """Test 20: run_evaluation returns exactly one row per fund symbol."""
        results = run_evaluation(eval_root)
        assert len(results) == results["symbol"].nunique()
        assert len(results) == 12

    def test_20b_no_missing_final_categories(self, eval_root):
        """Test 20b: Every fund has a non-null final_category."""
        results = run_evaluation(eval_root)
        assert results["final_category"].notna().all()

    def test_20c_no_missing_explanations(self, eval_root):
        """Test 20c: Every fund has a non-empty final_explanation."""
        results = run_evaluation(eval_root)
        assert results["final_explanation"].notna().all()
        assert (results["final_explanation"].str.len() > 10).all()

    def test_21_existing_input_files_not_modified(self, eval_root):
        """Test 21: run_evaluation does NOT modify any source CSV."""
        def _hashes(root):
            h = {}
            for key, rel in INPUT_FILES.items():
                content = (root / rel).read_bytes()
                h[key] = hash(content)
            return h

        root    = eval_root
        before  = _hashes(root)
        run_evaluation(root)
        after   = _hashes(root)
        assert before == after, (
            "Source files were modified by run_evaluation. Differences: "
            + str({k: (before[k], after[k]) for k in before if before[k] != after[k]})
        )

    def test_22_idempotency_two_runs_produce_identical_results(self, eval_root):
        """Test 22: Running the evaluator twice produces identical CSV content."""
        run_evaluation(eval_root)
        csv_path = eval_root / OUTPUT_CSV
        content1 = csv_path.read_text(encoding="utf-8")

        run_evaluation(eval_root)
        content2 = csv_path.read_text(encoding="utf-8")

        assert content1 == content2, (
            "Evaluator is not idempotent: two runs produced different CSV content."
        )

    def test_22b_idempotency_json_matches_csv_symbols(self, eval_root):
        """Test 22b: JSON output contains same symbols as CSV output."""
        results  = run_evaluation(eval_root)
        json_path = eval_root / OUTPUT_JSON
        with open(json_path, encoding="utf-8") as fh:
            json_data = json.load(fh)
        csv_symbols  = sorted(results["symbol"].tolist())
        json_symbols = sorted(rec["symbol"] for rec in json_data)
        assert csv_symbols == json_symbols

    def test_22c_output_scores_all_within_bounds(self, eval_root):
        """Test 22c: All score fields in real output are within documented bounds."""
        results = run_evaluation(eval_root)
        bounds = {
            "performance_score":     (0, 20),
            "significance_score":    (0, 25),
            "factor_score":          (0, 25),
            "persistence_score":     (0, 20),
            "style_luck_adjustment": (-10, 10),
            "total_score":           (0, 100),
        }
        for key, (lo, hi) in bounds.items():
            assert (results[key] >= lo).all() and (results[key] <= hi).all(), (
                f"{key} out of bounds: {results[[key, 'symbol']].to_dict()}"
            )

    def test_22d_evaluation_version_field_present_in_all_rows(self, eval_root):
        """Test 22d: evaluation_version field is in every output row."""
        results = run_evaluation(eval_root)
        assert "evaluation_version" in results.columns
        assert (results["evaluation_version"] == EVALUATION_VERSION).all()
