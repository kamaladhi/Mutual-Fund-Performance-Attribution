"""
Shared pytest fixtures.

Tests that *run* the evaluator write final_evaluation.csv/json; they must do so in a temporary copy of the
inputs so that running the test-suite can never modify the project's shipped deliverables.
"""

import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="module")
def eval_root(tmp_path_factory):
    """A temporary project root holding copies of the evaluator's inputs and optional context files."""
    root = tmp_path_factory.mktemp("eval_root")
    shutil.copytree(PROJECT_ROOT / "data" / "processed", root / "data" / "processed")
    (root / "data" / "raw").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "screening_metadata.json", root / "data" / "raw" / "screening_metadata.json")
    return root
