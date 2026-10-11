"""
Logic-correction tests for src/evaluator.py (evaluation version 2.0.0).

Each test pins a defect found in the v1.0.0 review so that it cannot silently return:
  * significance credit for a significantly NEGATIVE alpha;
  * "Likely Style/Luck Driven" applied to funds with no outperformance to explain;
  * non-monotone "Moderate" rule (adding persistence demoted a fund);
  * persistence-sign alone conferring Moderate evidence.
"""

import pandas as pd
import pytest

from src.evaluator import (
    CAT_MODERATE, CAT_POSITIVE_INCONCLUSIVE, CAT_PROMISING, CAT_STRONG, CAT_STYLE_LUCK, CAT_WEAK, EVALUATION_VERSION,
    classify_fund, compute_evidence_flags, compute_scores, generate_explanation,
    load_attribution_context, run_evaluation, _primary_strengths, _primary_weaknesses,
)
from tests.test_evaluator import PROJECT_ROOT, _row


def _evaluate(row):
    flags = compute_evidence_flags(row)
    scores = compute_scores(row, flags)
    category = classify_fund(flags, scores, row)
    return flags, scores, category


def test_version_bumped():
    assert EVALUATION_VERSION == "2.3.0"


def test_promising_candidate_requires_convergent_but_not_conclusive_evidence():
    row = _row(
        capm_alpha_annual_pct=7.0,
        capm_alpha_daily=7.0 / 25200,
        capm_alpha_pvalue=0.074,
        bootstrap_p_value=0.061,
        bootstrap_significant=False,
        bootstrap_ci_lower=-0.00001,
        factor_alpha_annual_pct=7.9,
        factor_alpha_pvalue=0.045,
        factor_alpha_significant=True,
        alpha_reduction_pct=-12.0,
        alpha_positive_majority=True,
        alpha_positive_all_periods=False,
    )
    flags, scores, category = _evaluate(row)
    assert category == CAT_PROMISING
    explanation = generate_explanation(category, flags, scores, row)
    assert "Skill Candidate (Preliminary)" in explanation
    assert "not a confirmed demonstration" in explanation


def test_promising_candidate_does_not_ignore_failed_factor_or_persistence_checks():
    base = dict(
        capm_alpha_annual_pct=7.0, capm_alpha_daily=7.0 / 25200,
        capm_alpha_pvalue=0.074, bootstrap_p_value=0.061,
        bootstrap_significant=False, bootstrap_ci_lower=-0.00001,
        factor_alpha_annual_pct=7.9, factor_alpha_pvalue=0.045,
        factor_alpha_significant=True, alpha_reduction_pct=-12.0,
        alpha_positive_majority=True, alpha_positive_all_periods=False,
    )
    assert _evaluate(_row(**{**base, "factor_alpha_pvalue": 0.20, "factor_alpha_significant": False}))[2] != CAT_PROMISING
    assert _evaluate(_row(**{**base, "alpha_positive_majority": False}))[2] != CAT_PROMISING
    assert _evaluate(_row(**{**base, "bootstrap_p_value": 0.15}))[2] != CAT_PROMISING


# ---------------------------------------------------------------- sign-aware significance -------

def _negative_significant_row(**kw):
    base = dict(capm_alpha_annual_pct=-4.0, capm_alpha_daily=-4.0 / 25200, capm_alpha_pvalue=0.01,
                bootstrap_p_value=0.01, bootstrap_significant=False, bootstrap_ci_lower=-0.0004,
                factor_alpha_annual_pct=-3.5, factor_alpha_pvalue=0.02, factor_alpha_significant=True,
                alpha_reduction_pct=12.5, alpha_positive_majority=False, alpha_positive_all_periods=False,
                positive_alpha_persistence_rate=0.0, significant_positive_alpha_persistence_rate=0.0)
    base.update(kw)
    return _row(**base)


def test_significantly_negative_alpha_earns_no_significance_or_performance_points():
    flags, scores, category = _evaluate(_negative_significant_row())
    assert flags["capm_significant"] is False                  # "significant" means significantly POSITIVE
    assert flags["capm_significantly_negative"] is True
    assert scores["significance_score"] == 0.0                 # v1 awarded 15 here
    assert scores["performance_score"] == 0.0
    assert category == CAT_WEAK


def test_bootstrap_guard_is_direction_aware():
    row = _negative_significant_row(bootstrap_p_value=0.01, bootstrap_significant=False)
    assert compute_evidence_flags(row)["bootstrap_significant"] is False   # p<0.05 but alpha<0
    pos = _row(capm_alpha_annual_pct=3.0, bootstrap_p_value=0.01, bootstrap_significant=False)
    assert compute_evidence_flags(pos)["bootstrap_significant"] is True


def test_negative_alpha_is_reported_as_weakness_not_strength():
    flags, _, _ = _evaluate(_negative_significant_row())
    strengths = " ".join(_primary_strengths(flags))
    weaknesses = " ".join(_primary_weaknesses(flags))
    assert "significant" not in strengths.lower()
    assert "significantly negative" in weaknesses


def test_weak_explanation_for_significantly_negative_alpha_says_underperformance():
    row = _negative_significant_row()
    flags, scores, category = _evaluate(row)
    text = generate_explanation(category, flags, scores, row)
    assert "under-performance" in text and "Weak/Inconclusive" in text


# ---------------------------------------------------------------- style / luck eligibility -------

def test_negative_alpha_is_never_labelled_style_luck():
    row = _row(capm_alpha_annual_pct=-2.0, capm_alpha_daily=-2 / 25200, capm_alpha_pvalue=0.4,
               bootstrap_significant=False, bootstrap_p_value=0.4, bootstrap_ci_lower=-0.001,
               factor_alpha_annual_pct=-5.0, factor_alpha_pvalue=0.3, factor_alpha_significant=False,
               alpha_reduction_pct=150.0, alpha_positive_majority=False, alpha_positive_all_periods=False)
    flags, scores, category = _evaluate(row)
    assert category == CAT_WEAK
    assert scores["style_luck_adjustment"] == 0.0
    text = generate_explanation(category, flags, scores, row)
    assert "observed outperformance" not in text


def test_style_luck_needs_a_noteworthy_positive_capm_alpha():
    insignificant = _row(capm_alpha_annual_pct=0.10, capm_alpha_pvalue=0.95, bootstrap_significant=False,
                         bootstrap_p_value=0.9, bootstrap_ci_lower=-0.001, factor_alpha_annual_pct=-2.5,
                         factor_alpha_pvalue=0.36, factor_alpha_significant=False, alpha_reduction_pct=2628.0,
                         alpha_positive_majority=False)
    flags, scores, category = _evaluate(insignificant)
    assert category == CAT_WEAK                                  # nothing to "explain away" (v1: Style/Luck)
    assert scores["style_luck_adjustment"] == 0.0

    marginal = _row(**{**{k: insignificant[k] for k in insignificant.index}, "capm_alpha_annual_pct": 4.0,
                       "capm_alpha_pvalue": 0.08})
    _, scores_m, category_m = _evaluate(marginal)
    assert category_m == CAT_STYLE_LUCK
    assert scores_m["style_luck_adjustment"] == -10.0


# ---------------------------------------------------------------- monotone Moderate rule --------

def test_factor_survival_alone_is_moderate_even_with_persistence_and_negative_capm():
    kwargs = dict(capm_alpha_annual_pct=-0.5, capm_alpha_daily=-0.5 / 25200, capm_alpha_pvalue=0.6,
                  bootstrap_p_value=0.6, bootstrap_significant=False, bootstrap_ci_lower=-0.001,
                  factor_alpha_annual_pct=4.0, factor_alpha_pvalue=0.01, factor_alpha_significant=True,
                  alpha_reduction_pct=-900.0)
    with_persistence = _evaluate(_row(**kwargs, alpha_positive_majority=True))[2]
    without_persistence = _evaluate(_row(**kwargs, alpha_positive_majority=False))[2]
    assert without_persistence == CAT_MODERATE
    assert with_persistence == CAT_MODERATE                      # v1: Weak (adding evidence demoted the fund)


def test_sign_persistence_alone_needs_marginal_significance_for_moderate():
    base = dict(capm_alpha_annual_pct=2.0, bootstrap_significant=False, bootstrap_p_value=0.5, bootstrap_ci_lower=-0.001,
                factor_alpha_annual_pct=0.5, factor_alpha_pvalue=0.4, factor_alpha_significant=False,
                alpha_reduction_pct=20.0, alpha_positive_majority=True, alpha_positive_all_periods=False)
    assert _evaluate(_row(**base, capm_alpha_pvalue=0.50))[2] == CAT_POSITIVE_INCONCLUSIVE
    assert _evaluate(_row(**base, capm_alpha_pvalue=0.08))[2] == CAT_MODERATE


def test_strong_still_requires_all_five_criteria():
    strong = dict(capm_alpha_annual_pct=4.0, capm_alpha_pvalue=0.02, bootstrap_significant=True, bootstrap_p_value=0.02,
                  bootstrap_ci_lower=0.0002, factor_alpha_annual_pct=2.5, factor_alpha_pvalue=0.03,
                  factor_alpha_significant=True, alpha_reduction_pct=20.0, alpha_positive_majority=True,
                  capm_alpha_pvalue_bh=0.02, factor_alpha_pvalue_bh=0.03, bootstrap_p_value_bh_computed=0.03)
    assert _evaluate(_row(**strong))[2] == CAT_STRONG
    for broken in (dict(capm_alpha_pvalue=0.20), dict(bootstrap_significant=False, bootstrap_p_value=0.2),
                   dict(factor_alpha_pvalue=0.30, factor_alpha_significant=False), dict(alpha_positive_majority=False)):
        assert _evaluate(_row(**{**strong, **broken}))[2] != CAT_STRONG
    text = generate_explanation(CAT_STRONG, *_evaluate(_row(**strong))[:2], _row(**strong))
    assert "five core evidence criteria" in text


# ---------------------------------------------------------------- real-data guarantees ----------

def test_attribution_context_flags_partial_manager_tenure():
    ctx = load_attribution_context(PROJECT_ROOT).set_index("symbol")
    assert ctx.loc["118955", "tenure_overlap_pct"] < 40            # HDFC Flexi Cap: manager arrived mid-2022
    assert "Prashant Jain" in ctx.loc["118955", "attribution_note"]
    assert ctx.loc["122639", "tenure_overlap_pct"] == pytest.approx(100.0)
    assert "overseas" in ctx.loc["122639", "attribution_note"]


def test_real_output_has_v2_columns_and_no_nans_in_decision_fields(eval_root):
    results = run_evaluation(eval_root)
    for col in ("capm_alpha_pvalue", "bootstrap_p_value", "bootstrap_p_value_bh", "factor_alpha_pvalue",
                "alpha_reduction_pts_annual", "total_score", "tenure_overlap_pct"):
        assert col in results.columns and results[col].notna().all(), col
    assert set(results["evaluation_version"]) == {"2.3.0"}


def test_no_negative_alpha_fund_is_scored_for_significance_or_style_in_real_data(eval_root):
    results = run_evaluation(eval_root)
    neg = results[results["capm_alpha_annual_pct"] <= 0]
    assert (neg["performance_score"] == 0).all()
    assert (neg["significance_score"] == 0).all()
    assert (neg["final_category"] != CAT_STYLE_LUCK).all()
