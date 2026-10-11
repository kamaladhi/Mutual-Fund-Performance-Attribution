"""
Shared-rule tests for src/evaluator.py and src/persistence.py (evaluation version 2.3.0).

The US market is classified by the SAME rule set as the Indian market. Each test below pins one rule so
that it cannot silently drift; the repository-level test (tests/test_rule_parity.py) additionally checks
that the rule code is identical in both markets.
"""

import pandas as pd

from src.evaluator import (
    CAT_MODERATE, CAT_POSITIVE_INCONCLUSIVE, CAT_PROMISING, CAT_STRONG, CAT_STYLE_LUCK, CAT_WEAK, EVALUATION_VERSION,
    classify_fund, compute_evidence_flags, compute_scores, generate_explanation,
    _primary_strengths, _primary_weaknesses,
)
from src.persistence import calculate_persistence_metrics, get_chronological_period_splits
from test_evaluator import _row


def _evaluate(row):
    flags = compute_evidence_flags(row)
    scores = compute_scores(row, flags)
    category = classify_fund(flags, scores, row)
    return flags, scores, category


def test_version_is_the_shared_rule_version():
    assert EVALUATION_VERSION == "2.3.0"


# ---------------------------------------------------------------- persistence: strict majority ----------

def _period_frame(symbol, alphas, p_values):
    rows = []
    for i, (a, p) in enumerate(zip(alphas, p_values), start=1):
        rows.append({"symbol": symbol, "period_id": f"P{i}", "alpha_daily": a,
                     "alpha_annual_pct": ((1 + a) ** 252 - 1) * 100, "alpha_pvalue": p, "alpha_pvalue_hac": p})
    return pd.DataFrame(rows)


def test_two_of_four_is_not_a_majority():
    row = calculate_persistence_metrics(_period_frame("TIE", [0.001, -0.001, 0.002, -0.002], [0.5] * 4)).iloc[0]
    assert row["positive_alpha_periods"] == 2
    assert bool(row["alpha_positive_majority"]) is False          # a coin-flip tie is not persistence
    assert row["positive_alpha_persistence_rate"] == 0.5


def test_three_of_four_is_a_majority_but_not_all():
    row = calculate_persistence_metrics(_period_frame("MAJ", [0.001, 0.001, 0.002, -0.002], [0.5] * 4)).iloc[0]
    assert bool(row["alpha_positive_majority"]) is True and bool(row["alpha_positive_all_periods"]) is False


def test_subperiod_significance_uses_the_hac_pvalue_when_present():
    frame = _period_frame("HAC", [0.001] * 4, [0.01] * 4)
    frame["alpha_pvalue_hac"] = [0.20, 0.20, 0.20, 0.01]
    assert calculate_persistence_metrics(frame).iloc[0]["significant_positive_alpha_periods"] == 1


def test_split_covers_every_date_exactly_once_for_uneven_lengths():
    dates = [f"2022-01-{d:02d}" for d in range(1, 28)]
    splits = get_chronological_period_splits(dates, 4)
    assert [s["n_dates"] for s in splits] == [6, 7, 7, 7]
    assert sorted(set().union(*[s["dates"] for s in splits])) == dates


# ---------------------------------------------------------------- sign-aware significance ---------------

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
    assert flags["capm_significant"] is False and flags["capm_significantly_negative"] is True
    assert scores["significance_score"] == 0.0 and scores["performance_score"] == 0.0
    assert category == CAT_WEAK


def test_bootstrap_guard_is_direction_aware():
    assert compute_evidence_flags(_negative_significant_row())["bootstrap_significant"] is False
    pos = _row(capm_alpha_annual_pct=3.0, bootstrap_p_value=0.01, bootstrap_significant=False)
    assert compute_evidence_flags(pos)["bootstrap_significant"] is True


def test_negative_alpha_is_reported_as_weakness_not_strength():
    flags, _, _ = _evaluate(_negative_significant_row())
    assert "significant" not in " ".join(_primary_strengths(flags)).lower()
    assert "significantly negative" in " ".join(_primary_weaknesses(flags))


# ---------------------------------------------------------------- style / luck eligibility --------------

def test_style_luck_needs_a_noteworthy_positive_capm_alpha():
    insignificant = _row(capm_alpha_annual_pct=0.10, capm_alpha_pvalue=0.95, bootstrap_significant=False,
                         bootstrap_p_value=0.9, bootstrap_ci_lower=-0.001, factor_alpha_annual_pct=-2.5,
                         factor_alpha_pvalue=0.36, factor_alpha_significant=False, alpha_reduction_pct=2628.0,
                         alpha_positive_majority=False)
    _, scores, category = _evaluate(insignificant)
    assert category == CAT_WEAK and scores["style_luck_adjustment"] == 0.0     # nothing to explain away

    marginal = _row(**{**{k: insignificant[k] for k in insignificant.index}, "capm_alpha_annual_pct": 4.0,
                       "capm_alpha_pvalue": 0.08})
    _, scores_m, category_m = _evaluate(marginal)
    assert category_m == CAT_STYLE_LUCK and scores_m["style_luck_adjustment"] == -10.0


def test_style_luck_score_penalty_needs_marginal_significance():
    """Under the pre-2.0 USA rules any positive CAPM alpha with a >= 25% reduction was penalised (even p=0.80)."""
    row = _row(capm_alpha_annual_pct=0.36, capm_alpha_pvalue=0.80, bootstrap_significant=False, bootstrap_p_value=0.8,
               bootstrap_ci_lower=-0.001, factor_alpha_annual_pct=0.14, factor_alpha_pvalue=0.92,
               factor_alpha_significant=False, alpha_reduction_pct=62.0, alpha_positive_majority=False)
    assert _evaluate(row)[1]["style_luck_adjustment"] == 0.0


# ---------------------------------------------------------------- Moderate gate / tiers ----------------

def test_factor_survival_alone_is_moderate_even_with_persistence_and_negative_capm():
    kwargs = dict(capm_alpha_annual_pct=-0.5, capm_alpha_daily=-0.5 / 25200, capm_alpha_pvalue=0.6,
                  bootstrap_p_value=0.6, bootstrap_significant=False, bootstrap_ci_lower=-0.001,
                  factor_alpha_annual_pct=4.0, factor_alpha_pvalue=0.01, factor_alpha_significant=True,
                  alpha_reduction_pct=-900.0)
    assert _evaluate(_row(**kwargs, alpha_positive_majority=True))[2] == CAT_MODERATE
    assert _evaluate(_row(**kwargs, alpha_positive_majority=False))[2] == CAT_MODERATE


def test_sign_persistence_alone_needs_marginal_significance_for_moderate():
    base = dict(capm_alpha_annual_pct=2.0, bootstrap_significant=False, bootstrap_p_value=0.5, bootstrap_ci_lower=-0.001,
                factor_alpha_annual_pct=0.5, factor_alpha_pvalue=0.4, factor_alpha_significant=False,
                alpha_reduction_pct=20.0, alpha_positive_majority=True, alpha_positive_all_periods=False)
    assert _evaluate(_row(**base, capm_alpha_pvalue=0.50))[2] == CAT_POSITIVE_INCONCLUSIVE
    assert _evaluate(_row(**base, capm_alpha_pvalue=0.08))[2] == CAT_MODERATE


def test_candidate_does_not_ignore_failed_factor_persistence_or_bootstrap_checks():
    base = dict(capm_alpha_annual_pct=7.0, capm_alpha_daily=7.0 / 25200, capm_alpha_pvalue=0.074,
                bootstrap_p_value=0.061, bootstrap_significant=False, bootstrap_ci_lower=-0.00001,
                factor_alpha_annual_pct=7.9, factor_alpha_pvalue=0.045, factor_alpha_significant=True,
                alpha_reduction_pct=-12.0, alpha_positive_majority=True, alpha_positive_all_periods=False)
    assert _evaluate(_row(**base))[2] == CAT_PROMISING
    assert _evaluate(_row(**{**base, "factor_alpha_pvalue": 0.20, "factor_alpha_significant": False}))[2] != CAT_PROMISING
    assert _evaluate(_row(**{**base, "alpha_positive_majority": False}))[2] != CAT_PROMISING
    assert _evaluate(_row(**{**base, "bootstrap_p_value": 0.15}))[2] != CAT_PROMISING
    assert _evaluate(_row(**{**base, "capm_alpha_pvalue": 0.30}))[2] != CAT_PROMISING


def test_candidate_explanation_states_the_bootstrap_threshold_truthfully():
    base = dict(capm_alpha_annual_pct=7.0, capm_alpha_daily=7.0 / 25200, capm_alpha_pvalue=0.074,
                bootstrap_significant=False, bootstrap_ci_lower=-0.00001,
                factor_alpha_annual_pct=7.9, factor_alpha_pvalue=0.045, factor_alpha_significant=True,
                alpha_reduction_pct=-12.0, alpha_positive_majority=True, alpha_positive_all_periods=False)
    def explain(row):
        flags, scores, category = _evaluate(row)
        assert category == CAT_PROMISING
        return generate_explanation(category, flags, scores, row)

    t_above = explain(_row(**base, bootstrap_p_value=0.061))
    t_below = explain(_row(**base, bootstrap_p_value=0.03))
    assert "not below the strict 0.05 threshold" in t_above
    assert "not below the strict 0.05 threshold" not in t_below and "below 0.05" in t_below


def test_strong_still_requires_all_five_criteria_and_the_bh_gate():
    strong = dict(capm_alpha_annual_pct=4.0, capm_alpha_pvalue=0.02, bootstrap_significant=True, bootstrap_p_value=0.02,
                  bootstrap_ci_lower=0.0002, factor_alpha_annual_pct=2.5, factor_alpha_pvalue=0.03,
                  factor_alpha_significant=True, alpha_reduction_pct=20.0, alpha_positive_majority=True,
                  capm_alpha_pvalue_bh=0.02, factor_alpha_pvalue_bh=0.03, bootstrap_p_value_bh_computed=0.03)
    assert _evaluate(_row(**strong))[2] == CAT_STRONG
    for broken in (dict(capm_alpha_pvalue=0.20), dict(bootstrap_significant=False, bootstrap_p_value=0.2),
                   dict(factor_alpha_pvalue=0.30, factor_alpha_significant=False), dict(alpha_positive_majority=False),
                   dict(factor_alpha_pvalue_bh=0.54)):
        assert _evaluate(_row(**{**strong, **broken}))[2] != CAT_STRONG
