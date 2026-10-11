"""
Headless render test of the combined dashboard (Streamlit's AppTest): every page must render without an exception
or an error banner, show the evidence classes in use (the preliminary tier under the name "Strong"; the two classes with
no funds are hidden unless a fund falls into them), and surface the per-fund caveats.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import combined_data as cd  # noqa: E402

APP = str(ROOT / "app.py")


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    parts += [s.value for s in at.success] + [w.value for w in at.warning] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


@pytest.mark.parametrize("page", ["Overview", "India", "USA", "Cross-market comparison", "Benchmark robustness",
                                  "Fund explorer", "Methodology & data"])
def test_every_page_renders_without_exceptions_or_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_overview_shows_the_four_classes_in_use_and_keeps_the_preliminary_tier_honest(app):
    app.sidebar.radio[0].set_value("Overview").run()
    text = _text(app)
    assert text.count('class="card"') == 4, "expected one card per class in use (the two empty classes are not shown)"
    for label in ("Strong", "Moderate", "Positive, inconclusive", "Weak / inconclusive"):
        assert label in text, f"class '{label}' is not visible on the Overview"
    assert "preliminary" in text.lower(), "the 'Strong' label must stay qualified as a preliminary tier"
    assert any("One rule set for both markets" in s.value for s in app.success)


def test_ui_names_and_class_visibility_rules():
    import pandas as pd
    import app as dashboard

    assert dashboard.ui("Strong Skill Candidate (Preliminary)") == "Strong"
    assert dashboard.ui("Moderate Evidence") == "Moderate Evidence"
    main_scale = ["Strong Skill Candidate (Preliminary)", "Moderate Evidence", "Positive but Inconclusive", "Weak/Inconclusive"]
    assert dashboard.shown_classes(pd.Series(["Weak/Inconclusive", "Moderate Evidence"])) == main_scale
    # a normally hidden class appears as soon as a fund falls into it, so no fund is ever left without a visible class
    assert "Likely Style/Luck Driven" in dashboard.shown_classes(pd.Series(["Likely Style/Luck Driven"]))
    assert "Strong Evidence of Skill" in dashboard.shown_classes(pd.Series(["Strong Evidence of Skill"]))
    assert set(dashboard.shown_classes(pd.Series(cd.CATEGORY_ORDER))) == set(cd.CATEGORY_ORDER)


def test_market_pages_explain_every_fund(app):
    for page in ("India", "USA"):
        app.sidebar.radio[0].set_value(page).run()
        assert len(app.expander) == 12, f"{page}: expected one explanation per fund"


def test_attribution_and_peer_caveats_reach_the_screen(app):
    app.sidebar.radio[0].set_value("India").run()
    assert "28%" in "\n".join(m.value for m in app.markdown) or any("28%" in str(d.value) for d in app.dataframe)
    app.sidebar.radio[0].set_value("USA").run()
    assert any("Universe caution" in m.value for m in app.markdown)


def test_fund_explorer_works_for_every_fund(app):
    app.sidebar.radio[0].set_value("Fund explorer").run()
    for market in cd.MARKETS:
        app.selectbox(key="fx_market").set_value(market).run()
        for option in list(app.selectbox(key="fx_fund").options):
            app.selectbox(key="fx_fund").set_value(option).run()
            assert not app.exception, (market, option, [e.value for e in app.exception])


def test_a_rule_version_mismatch_is_reported_not_hidden(monkeypatch):
    real = cd.all_results

    def mismatched():
        df, reg, issues = real()
        df = df.copy()
        df.loc[df["market"] == "USA", "version"] = "1.5.0"
        return df, reg, issues + ["The markets were classified under DIFFERENT rule versions (test)."]

    monkeypatch.setattr(cd, "all_results", mismatched)
    import streamlit as st
    st.cache_data.clear()                      # the app caches the loaded results; make it load the patched ones
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert any("NOT classified by the same rule version" in e.value for e in at.error)
