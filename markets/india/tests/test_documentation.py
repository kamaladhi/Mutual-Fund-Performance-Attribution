"""
Documentation-consistency tests: the README result tables must equal the shipped outputs, the documented
registry must equal the code registry, and claims proven false in review must not reappear in the docs.
"""

import re
from pathlib import Path

import pandas as pd

from src import data_loader as dl
from src import reporting

ROOT = Path(__file__).parent.parent.resolve()
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
NON_REPORT_DOCS = [p for p in DOCS if p.name != "REVIEW_REPORT.md"]


def _text(path):
    return path.read_text(encoding="utf-8")


def test_readme_results_table_matches_final_evaluation():
    final = pd.read_csv(ROOT / "data" / "processed" / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
    block = re.search(r"<!-- RESULTS_TABLE_START -->(.*?)<!-- RESULTS_TABLE_END -->", _text(ROOT / "README.md"), re.S).group(1)
    rows = re.findall(r"\|\s*`(\d+)`\s*\|[^|]*\|[^|]*\|\s*\*\*([^*|]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|", block)
    assert len(rows) == 12, "README results table is missing funds - run `python run_pipeline.py`"
    for code, category, score in rows:
        assert final.loc[code, "final_category"] == category
        assert abs(final.loc[code, "total_score"] - float(score)) < 0.05


def test_readme_sensitivity_table_matches_the_saved_sensitivity_output():
    text = _text(ROOT / "README.md")
    block = re.search(r"<!-- SENSITIVITY_TABLE_START -->(.*?)<!-- SENSITIVITY_TABLE_END -->", text, re.S).group(1)
    assert block.strip() == reporting.build_readme_sensitivity_table(str(ROOT)).strip()


def test_saved_sensitivity_output_is_current_for_the_primary_proxy():
    """benchmark_sensitivity.csv must be regenerated whenever the rules change: its primary-proxy block has to
    reproduce final_evaluation.csv exactly (an earlier version of the file pre-dated the BH gate)."""
    final = pd.read_csv(ROOT / "data" / "processed" / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
    sens = pd.read_csv(ROOT / "data" / "processed" / "benchmark_sensitivity.csv", dtype={"symbol": str})
    primary = sens[sens["market_proxy"] == "iima_market"].set_index("symbol")
    assert set(primary.index) == set(final.index)
    assert (primary.loc[final.index, "final_category"] == final["final_category"]).all()
    assert (primary.loc[final.index, "total_score"] - final["total_score"]).abs().max() < 1e-9


def test_no_proxy_reports_strong_evidence_that_fails_the_multiple_testing_gate():
    """Strong Evidence needs BH q < 0.10 for every test family; with 12 funds that is only possible if some
    p-value is far below 0.05. PPFAS's four-factor q-value is 0.54, so it can never be 'Strong' under any proxy."""
    final = pd.read_csv(ROOT / "data" / "processed" / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
    sens = pd.read_csv(ROOT / "data" / "processed" / "benchmark_sensitivity.csv", dtype={"symbol": str})
    strong = sens[sens["final_category"] == "Strong Evidence of Skill"]
    # the four-factor q-value does not depend on the CAPM market proxy
    assert (final.loc[strong["symbol"], "factor_alpha_pvalue_bh"] < 0.10).all()


def test_documented_registry_matches_code_registry():
    text = _text(ROOT / "docs" / "DATA_SOURCES.md")
    for code, meta in dl.FUND_METADATA.items():
        row = next(line for line in text.splitlines() if line.startswith(f"| `{code}`"))
        assert meta["lead_manager"] in row and meta["tenure_start"] in row and meta["category"] in row
        assert f"{meta['expense_ratio']:.2f}%" in row


def test_every_source_module_is_documented_in_readme_and_architecture():
    readme, arch = _text(ROOT / "README.md"), _text(ROOT / "docs" / "ARCHITECTURE.md")
    for module in sorted((ROOT / "src").glob("*.py")):
        if module.name == "__init__.py":
            continue
        assert module.name in readme, f"{module.name} missing from README structure"
        assert module.name in arch, f"{module.name} missing from ARCHITECTURE"


def test_stale_or_false_claims_are_gone_from_the_documentation():
    forbidden = [
        "1,227", "calibrated synthetic", "calibrated Indian factor", "VERIFIED_ACTIVE_INDIAN_DIRECT",
        "76 tests", "76 Tests", "mandated SEBI", "mandated under SEBI", "SEBI mandate Direct Plan",
        "Falls back to calibrated", "U.S. Version", "US Fund Manager Skill Detector",
        "Implementation is identical to the US", "NIFTY 500 TRI (^CRSLDX)", "NIFTY 500 Total Returns Index (TRI).",
    ]
    for path in NON_REPORT_DOCS:
        text = _text(path)
        for phrase in forbidden:
            assert phrase not in text, f"stale claim '{phrase}' in {path.name}"


def test_docs_state_the_corrected_facts():
    methodology = _text(ROOT / "docs" / "METHODOLOGY.md")
    assert "price index" in _text(ROOT / "docs" / "DATA_SOURCES.md")
    assert "strictly more than half" in methodology
    assert "Newey-West" in methodology and "synchronous" in methodology.lower()
    assert "2.8 times" in methodology                                  # b=30 is NOT the n^(1/3) rule of thumb
    assert "REVIEW_REPORT.md" in _text(ROOT / "README.md")


def test_review_report_exists_and_has_a_verdict():
    report = _text(ROOT / "docs" / "REVIEW_REPORT.md")
    assert "Verdict" in report and "What is correct" in report and "What was wrong" in report
