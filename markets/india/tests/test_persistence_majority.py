"""
Strict-majority semantics and HAC significance for src/persistence.py (added in review, version 2.0.0).
"""

import pandas as pd

from src.persistence import calculate_persistence_metrics, get_chronological_period_splits


def _period_frame(symbol, alphas, p_values):
    rows = []
    for i, (a, p) in enumerate(zip(alphas, p_values), start=1):
        rows.append({"symbol": symbol, "period_id": f"P{i}", "alpha_daily": a,
                     "alpha_annual_pct": ((1 + a) ** 252 - 1) * 100, "alpha_pvalue": p, "alpha_pvalue_hac": p})
    return pd.DataFrame(rows)


def test_two_of_four_is_not_a_majority():
    frame = _period_frame("TIE", [0.001, -0.001, 0.002, -0.002], [0.5] * 4)
    row = calculate_persistence_metrics(frame).iloc[0]
    assert row["positive_alpha_periods"] == 2
    assert bool(row["alpha_positive_majority"]) is False          # a coin-flip tie is not persistence
    assert bool(row["alpha_positive_all_periods"]) is False
    assert row["positive_alpha_persistence_rate"] == 0.5


def test_three_of_four_is_a_majority_but_not_all():
    frame = _period_frame("MAJ", [0.001, 0.001, 0.002, -0.002], [0.5] * 4)
    row = calculate_persistence_metrics(frame).iloc[0]
    assert bool(row["alpha_positive_majority"]) is True
    assert bool(row["alpha_positive_all_periods"]) is False


def test_four_of_four_is_all_periods():
    frame = _period_frame("ALL", [0.001] * 4, [0.5] * 4)
    row = calculate_persistence_metrics(frame).iloc[0]
    assert bool(row["alpha_positive_all_periods"]) is True and bool(row["alpha_positive_majority"]) is True


def test_significance_in_subperiods_uses_hac_pvalue_when_present():
    frame = _period_frame("HAC", [0.001] * 4, [0.01] * 4)
    frame["alpha_pvalue_hac"] = [0.20, 0.20, 0.20, 0.01]           # OLS says 4 significant, HAC says 1
    row = calculate_persistence_metrics(frame).iloc[0]
    assert row["significant_positive_alpha_periods"] == 1


def test_split_covers_every_date_exactly_once_for_uneven_lengths():
    dates = [f"2022-01-{d:02d}" for d in range(1, 28)]            # 27 dates -> periods of 6,7,7,7
    splits = get_chronological_period_splits(dates, 4)
    assert [s["n_dates"] for s in splits] == [6, 7, 7, 7]
    assert sorted(set().union(*[s["dates"] for s in splits])) == dates
