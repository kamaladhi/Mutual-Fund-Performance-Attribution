"""
Both markets must be classified by ONE rule set.

The two market projects keep separate ``src`` packages (they share module names), so the rule code lives in
two files. This test compares the syntax trees of every rule function and threshold constant in
markets/india/src/evaluator.py and markets/usa/src/evaluator.py (docstrings and comments ignored). If a rule is
changed in one market and not the other, this test fails - which is exactly the inconsistency that previously
made the cross-market comparison unreliable.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = {m: ROOT / "markets" / m / "src" / "evaluator.py" for m in ("india", "usa")}
PERSISTENCE = {m: ROOT / "markets" / m / "src" / "persistence.py" for m in ("india", "usa")}
STATS = {m: ROOT / "markets" / m / "src" / "stats_utils.py" for m in ("india", "usa")}

EVALUATOR_FUNCTIONS = [
    "compute_evidence_flags", "compute_scores", "classify_fund",
    "_primary_strengths", "_primary_weaknesses", "generate_explanation", "evaluate_fund", "_opt",
]
EVALUATOR_CONSTANTS = [
    "EVALUATION_VERSION", "CAT_STRONG", "CAT_PROMISING", "CAT_POSITIVE_INCONCLUSIVE", "CAT_MODERATE", "CAT_WEAK",
    "CAT_STYLE_LUCK", "ALPHA_PVALUE_THRESHOLD", "BOOTSTRAP_PVALUE_THRESHOLD", "MARGINAL_PVALUE_THRESHOLD",
    "STYLE_REDUCTION_THRESHOLD", "MATERIAL_REDUCTION_THRESHOLD", "STRONG_ALPHA_PCT", "MODERATE_ALPHA_PCT",
]
PERSISTENCE_FUNCTIONS = ["get_chronological_period_splits", "calculate_persistence_metrics"]
STATS_FUNCTIONS = ["default_hac_lags", "newey_west_cov", "two_sided_t_pvalue", "benjamini_hochberg"]


def _strip_docstring(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                and isinstance(first.value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return node


def _functions(path: Path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {n.name: _strip_docstring(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    return {name: ast.dump(found[name]) for name in names if name in found}, set(names) - set(found)


def _constants(path: Path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target in names:
            out[target] = ast.dump(value)
    return out


@pytest.mark.parametrize("name", EVALUATOR_FUNCTIONS)
def test_evaluator_rule_functions_are_identical_in_both_markets(name):
    india, miss_i = _functions(EVALUATORS["india"], EVALUATOR_FUNCTIONS)
    usa, miss_u = _functions(EVALUATORS["usa"], EVALUATOR_FUNCTIONS)
    assert not miss_i and not miss_u, f"missing rule functions: india={miss_i} usa={miss_u}"
    assert india[name] == usa[name], f"{name}() differs between markets/india and markets/usa"


def test_evaluator_thresholds_and_labels_are_identical_in_both_markets():
    india, usa = _constants(EVALUATORS["india"], EVALUATOR_CONSTANTS), _constants(EVALUATORS["usa"], EVALUATOR_CONSTANTS)
    assert set(india) == set(EVALUATOR_CONSTANTS) == set(usa)
    differing = [k for k in EVALUATOR_CONSTANTS if india[k] != usa[k]]
    assert not differing, f"constants differ between markets: {differing}"


@pytest.mark.parametrize("name", PERSISTENCE_FUNCTIONS)
def test_persistence_rules_are_identical_in_both_markets(name):
    india, usa = _functions(PERSISTENCE["india"], PERSISTENCE_FUNCTIONS)[0], _functions(PERSISTENCE["usa"], PERSISTENCE_FUNCTIONS)[0]
    assert india[name] == usa[name]


@pytest.mark.parametrize("name", STATS_FUNCTIONS)
def test_statistical_helpers_are_identical_in_both_markets(name):
    india, usa = _functions(STATS["india"], STATS_FUNCTIONS)[0], _functions(STATS["usa"], STATS_FUNCTIONS)[0]
    assert india[name] == usa[name]


def test_the_parity_check_can_actually_fail():
    """Guard against a vacuous test: changing one threshold must change the compared representation."""
    base = ast.dump(ast.parse("MARGINAL_PVALUE_THRESHOLD: float = 0.10").body[0].value)
    changed = ast.dump(ast.parse("MARGINAL_PVALUE_THRESHOLD: float = 0.15").body[0].value)
    assert base != changed
