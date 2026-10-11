"""
Repository-level checks for the combined project.

These assert invariants of the saved results (shared rule version, complete class vocabulary, an export that is
reproducible from its sources, caveats that survive into the combined table). They deliberately do NOT pin a
category distribution: a correct change to the data or the rules must not be able to "fail" them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import combined_data as cd  # noqa: E402

INDIA = ROOT / "markets" / "india"
USA = ROOT / "markets" / "usa"
COMBINED = ROOT / "data" / "combined_results.csv"


def test_both_market_projects_are_present():
    for project in (INDIA, USA):
        assert (project / "app.py").exists()
        assert (project / "src").is_dir()
        assert (project / "data" / "processed" / "final_evaluation.csv").exists()
        assert (project / "tests").is_dir()


def test_combined_export_has_both_markets_and_24_funds():
    df = pd.read_csv(COMBINED, dtype={"symbol": str})
    assert len(df) == 24
    assert set(df["market"]) == {"India", "USA"}
    assert df.groupby("market")["symbol"].nunique().to_dict() == {"India": 12, "USA": 12}


def test_both_markets_share_one_rule_version():
    df, _, issues = cd.all_results()
    assert not issues, issues
    assert cd.rules_version(df) == "2.3.0"
    assert df.groupby("market")["version"].nunique().max() == 1


def test_every_fund_has_a_recognised_class_and_all_six_classes_are_representable():
    df, _, _ = cd.all_results()
    assert set(df["category"]) <= set(cd.CATEGORY_ORDER)
    counts = cd.category_counts(df)
    assert list(counts.index) == cd.CATEGORY_ORDER                  # all six classes always listed, strongest first
    assert counts["Total"].sum() == 24
    assert (counts[["India", "USA"]].sum(axis=1) == counts["Total"]).all()


def test_the_combined_export_is_reproducible_from_the_market_files():
    """data/combined_results.csv has a generator; it must equal a fresh build from the saved market results."""
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_combined_results.py"), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_export_preserves_source_labels_and_alphas():
    df = pd.read_csv(COMBINED, dtype={"symbol": str})
    for col in ("final_category", "capm_alpha_annual_pct", "factor_alpha_annual_pct", "evaluation_version"):
        assert col in df.columns and df[col].notna().all()
    for market, folder in (("India", INDIA), ("USA", USA)):
        source = pd.read_csv(folder / "data" / "processed" / "final_evaluation.csv", dtype={"symbol": str}).set_index("symbol")
        mine = df[df["market"] == market].set_index("symbol")
        assert (mine.loc[source.index, "final_category"] == source["final_category"]).all()
        assert (mine.loc[source.index, "capm_alpha_annual_pct"] - source["capm_alpha_annual_pct"]).abs().max() < 1e-12


def test_us_ticker_identity_is_not_mislabeled():
    metadata = json.loads((USA / "data" / "raw" / "screening_metadata.json").read_text(encoding="utf-8"))
    pagrx = next(item for item in metadata if item.get("ticker") == "PAGRX")
    assert pagrx["fund_name"] == "Permanent Portfolio Aggressive Growth Fund"
    assert "PRGFX" in pagrx["identity_note"]


def test_caveats_survive_into_the_combined_table():
    df, _, _ = cd.all_results()
    hdfc = df[(df["market"] == "India") & (df["symbol"] == "118955")].iloc[0]
    assert hdfc["tenure_overlap_pct"] < 40 and "28%" in hdfc["caveat"]            # manager arrived mid-2022
    assert "Prashant Jain" in hdfc["caveat_detail"]
    ppfas = df[(df["market"] == "India") & (df["symbol"] == "122639")].iloc[0]
    assert "overseas" in ppfas["caveat_detail"].lower()
    usa = df[df["market"] == "USA"].set_index("symbol")
    for ticker in ("AMECX", "PRHSX", "PAGRX"):
        assert "not a broad-equity peer" in usa.loc[ticker, "caveat"]
    assert (usa.drop(index=["AMECX", "PRHSX", "PAGRX"])["caveat"] == "").all()


def test_a_label_outside_the_shared_vocabulary_is_flagged_not_hidden(tmp_path, monkeypatch):
    """If a market were re-evaluated with an old rule set (e.g. the old 'Strong Potential Candidate' label) the data
    layer must say so instead of silently merging the labels."""
    market = dict(cd.MARKETS["USA"])
    fake = tmp_path / "usa"
    (fake / "data" / "processed").mkdir(parents=True)
    (fake / "data" / "raw").mkdir(parents=True)
    src = pd.read_csv(USA / "data" / "processed" / "final_evaluation.csv")
    src.loc[0, "final_category"] = "Strong Potential Candidate"
    src.loc[0, "evaluation_version"] = "1.5.0"
    src.to_csv(fake / "data" / "processed" / "final_evaluation.csv", index=False)
    market["path"] = fake
    monkeypatch.setitem(cd.MARKETS, "USA", market)
    df, _, issues = cd.all_results()
    assert any("unrecognised classification label" in i for i in issues)
    assert any("DIFFERENT rule versions" in i for i in issues)
    assert cd.rules_version(df) is None
