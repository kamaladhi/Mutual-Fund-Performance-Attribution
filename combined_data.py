"""
Data layer of the combined Fund Manager Skill Detector (no Streamlit import).

Used by the dashboard (app.py), by scripts/build_combined_results.py and by the tests, so that the
numbers shown on screen, the exported table and the checks all come from the same code.

Both markets are classified by ONE rule set (evaluation v2.3.0; see docs/CLASSIFICATION_GUIDE.md), so the
six category labels are identical in India and the USA and are shown as they are: nothing is merged,
renamed or re-labelled here. This module only reads the saved ``final_evaluation.csv`` files and the
screening metadata; it never estimates anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

MARKETS: dict[str, dict[str, Any]] = {
    "India": {
        "key": "india",
        "path": ROOT / "markets" / "india",
        "market_note": "IIMA value-weighted total-return market portfolio for CAPM and the four-factor model "
                       "(the NIFTY 500 price index is used only in the benchmark-robustness check).",
        "factor_note": "IIMA Indian size, value and momentum factors (Agarwalla, Jacob, Varma).",
        "universe_note": "12 hand-picked Indian equity schemes, Direct Plan - Growth, 2019-2023.",
    },
    "USA": {
        "key": "usa",
        "path": ROOT / "markets" / "usa",
        "market_note": "Fama-French Mkt-RF (total-return market excess return) for CAPM and the four-factor model "
                       "(the S&P 500 price index is descriptive only).",
        "factor_note": "Fama-French size, value and Carhart momentum factors.",
        "universe_note": "12 hand-picked US mutual funds, 2019-2023; three are exploratory non-peers (AMECX, PRHSX, PAGRX).",
    },
}

# Order = strongest evidence first. The same six labels are produced by both markets' evaluators.
CATEGORY_ORDER: list[str] = [
    "Strong Evidence of Skill",
    "Strong Skill Candidate (Preliminary)",
    "Moderate Evidence",
    "Positive but Inconclusive",
    "Weak/Inconclusive",
    "Likely Style/Luck Driven",
]

CATEGORY_INFO: dict[str, dict[str, str]] = {
    "Strong Evidence of Skill": {
        "color": "#166534",
        "short": "Strong evidence",
        "meaning": "Every strict test passes: a significantly positive CAPM alpha (HAC p < 0.05), a significantly positive "
                   "bootstrap, a significant positive four-factor alpha, positive alpha in at least 3 of 4 subperiods, "
                   "and Benjamini-Hochberg q < 0.10 for all three tests.",
    },
    "Strong Skill Candidate (Preliminary)": {
        "color": "#059669",
        "short": "Skill candidate",
        "meaning": "A screening tier, not confirmed skill: positive CAPM alpha with marginal evidence (p < 0.10), a significant "
                   "positive four-factor alpha, positive alpha in at least 3 of 4 subperiods and a bootstrap p < 0.10.",
    },
    "Moderate Evidence": {
        "color": "#2563eb",
        "short": "Moderate",
        "meaning": "At least one material positive signal (a significant CAPM alpha, a surviving four-factor alpha, "
                   "persistence with marginal significance, or a significant bootstrap) without the convergence needed above.",
    },
    "Positive but Inconclusive": {
        "color": "#0ea5e9",
        "short": "Positive, inconclusive",
        "meaning": "Both alphas are positive and CAPM alpha is positive in at least 3 of 4 subperiods, but nothing is "
                   "statistically significant. A direction, not evidence.",
    },
    "Weak/Inconclusive": {
        "color": "#64748b",
        "short": "Weak / inconclusive",
        "meaning": "No convincing positive evidence: the alpha is non-positive, or positive without statistical support. "
                   "This is not proof that the manager has no skill.",
    },
    "Likely Style/Luck Driven": {
        "color": "#d97706",
        "short": "Style / luck driven",
        "meaning": "A positive, at-least-marginally significant CAPM alpha of which at least 75% disappears after the "
                   "factor adjustment, leaving no significant four-factor alpha.",
    },
}

SENSITIVITY_PROXY_LABELS: dict[str, str] = {
    "iima_market": "IIMA total-return market (primary)",
    "nifty500_price": "NIFTY 500 price index (excludes dividends)",
    "nifty500_price_plus_assumed_dividend_yield": "NIFTY 500 price + assumed 1.3% dividend yield",
    "nifty500_tri": "NIFTY 500 total-return index",
}

TENURE_WARN_PCT = 90.0   # same threshold as the India evaluator


# ----------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------

def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _short_caveat(row: pd.Series) -> str:
    """One short line shown next to a fund wherever the table is read at a glance."""
    parts: list[str] = []
    if str(row.get("universe_status", "")).startswith("EXPLORATORY"):
        parts.append("not a broad-equity peer")
    overlap = row.get("tenure_overlap_pct", np.nan)
    if pd.notna(overlap) and overlap < TENURE_WARN_PCT:
        parts.append(f"named manager in charge {overlap:.0f}% of window")
    if str(row.get("attribution_note", "") or "").strip() and not parts and "overseas" in str(row["attribution_note"]).lower():
        parts.append("overseas / cash holdings not captured by the factors")
    return "; ".join(parts)


def _full_caveat(row: pd.Series) -> str:
    notes: list[str] = []
    for col in ("comparability_note", "attribution_note"):
        value = row.get(col, "")
        if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
            notes.append(value.strip())
    return " ".join(notes)


# ----------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------

def load_market(market_name: str) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Load one market's saved evaluation, enriched with registry metadata. Returns (table, registry, issues)."""
    cfg = MARKETS[market_name]
    project: Path = cfg["path"]
    result_path = project / "data" / "processed" / "final_evaluation.csv"
    issues: list[str] = []

    if not result_path.exists():
        return pd.DataFrame(), [], [f"Missing required file: {result_path.relative_to(ROOT)}"]
    try:
        df = pd.read_csv(result_path, dtype={"symbol": str})
    except Exception as exc:  # noqa: BLE001 - reported to the user, never silently ignored
        return pd.DataFrame(), [], [f"Could not read the {market_name} evaluation: {exc}"]

    missing = sorted({"symbol", "final_category"} - set(df.columns))
    if missing:
        return pd.DataFrame(), [], [f"{market_name} results are missing required columns: {', '.join(missing)}"]

    registry = _read_json(project / "data" / "raw" / "screening_metadata.json", [])
    by_ticker: dict[str, dict[str, Any]] = {}
    if isinstance(registry, list):
        for item in registry:
            ticker = str(item.get("ticker", item.get("symbol", ""))).strip()
            if ticker:
                by_ticker[ticker] = item

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["market"] = market_name

    def meta(field: str, default: str = "") -> pd.Series:
        return df["symbol"].map(lambda s: by_ticker.get(s, {}).get(field, default))

    df["fund_name"] = df["symbol"].map(
        lambda s: by_ticker.get(s, {}).get("fund_name", by_ticker.get(s, {}).get("short_name", s)))
    df["short_name"] = df["symbol"].map(
        lambda s: by_ticker.get(s, {}).get("short_name") or by_ticker.get(s, {}).get("fund_name", s))
    df["lead_manager"] = meta("lead_manager")
    df["amc"] = meta("amc")
    df["universe_status"] = meta("universe_status", "CORE_CANDIDATE")
    df["comparability_note"] = meta("comparability_note")
    if "tenure_start" not in df.columns:
        df["tenure_start"] = meta("tenure_start")
    for col in ("attribution_note",):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "tenure_overlap_pct" not in df.columns:
        df["tenure_overlap_pct"] = np.nan

    df["category"] = df["final_category"].fillna("Unclassified").astype(str)
    df["category_rank"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)}).fillna(len(CATEGORY_ORDER)).astype(int)

    df["score"] = _num(df, "total_score")
    df["capm_alpha_pct"] = _num(df, "capm_alpha_annual_pct")
    df["factor_alpha_pct"] = _num(df, "factor_alpha_annual_pct")
    df["capm_pvalue"] = _num(df, "capm_alpha_pvalue")
    df["factor_pvalue"] = _num(df, "factor_alpha_pvalue")
    df["bootstrap_pvalue"] = _num(df, "bootstrap_p_value")
    df["capm_q"] = _num(df, "capm_alpha_pvalue_bh")
    df["factor_q"] = _num(df, "factor_alpha_pvalue_bh")
    df["bootstrap_q"] = _num(df, "bootstrap_p_value_bh_computed")
    df["persistence_rate"] = _num(df, "positive_alpha_persistence_rate")
    df["positive_periods"] = (df["persistence_rate"] * 4).round()
    df["persistence_label"] = df["positive_periods"].map(lambda v: "n/a" if pd.isna(v) else f"{int(v)} of 4")
    df["version"] = df["evaluation_version"].astype(str) if "evaluation_version" in df.columns else "unknown"

    df["caveat"] = df.apply(_short_caveat, axis=1)
    df["caveat_detail"] = df.apply(_full_caveat, axis=1)

    if len(df) == 0:
        issues.append(f"{market_name}: the evaluation file contains no funds.")
    if df["symbol"].duplicated().any():
        issues.append(f"{market_name}: duplicate fund symbols in final_evaluation.csv.")
    unknown = sorted(set(df["category"]) - set(CATEGORY_ORDER))
    if unknown:
        issues.append(f"{market_name}: unrecognised classification label(s) {unknown} - the evaluator and the dashboard "
                      "are on different rule versions.")
        df.loc[df["category"].isin(unknown), "category"] = "Unclassified"
    return df, registry if isinstance(registry, list) else [], issues


def all_results() -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], list[str]]:
    """Both markets in one table (India first), the screening registries, and every data-quality issue found."""
    frames, registries, issues = [], {}, []
    for market in MARKETS:
        df, registry, problems = load_market(market)
        registries[market] = registry
        issues.extend(problems)
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        versions = combined.groupby("market")["version"].agg(lambda s: ", ".join(sorted(set(s))))
        if combined["version"].nunique() > 1:
            issues.append("The markets were classified under DIFFERENT rule versions ("
                          + "; ".join(f"{m}: v{v}" for m, v in versions.items())
                          + "). Their classes are not comparable until both are re-evaluated with the same rules.")
        order = {m: i for i, m in enumerate(MARKETS)}
        combined = (combined.assign(_m=combined["market"].map(order))
                    .sort_values(["_m", "symbol"]).drop(columns="_m").reset_index(drop=True))
    return combined, registries, issues


def rules_version(df: pd.DataFrame) -> str | None:
    """The single rule-set version shared by all loaded funds, or None if they differ."""
    if df.empty or df["version"].nunique() != 1:
        return None
    return str(df["version"].iloc[0])


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Funds per class and market. ALL six classes always appear (zeros included), strongest first."""
    markets = list(MARKETS)
    table = (df.groupby(["category", "market"]).size().unstack(fill_value=0)
             .reindex(index=CATEGORY_ORDER, columns=markets, fill_value=0))
    table["Total"] = table.sum(axis=1)
    return table.astype(int)


def load_benchmark_sensitivity() -> pd.DataFrame | None:
    """India's classification re-run under each alternative market proxy (None if the file is absent)."""
    path = MARKETS["India"]["path"] / "data" / "processed" / "benchmark_sensitivity.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype={"symbol": str})
    except Exception:  # noqa: BLE001
        return None


def output_fingerprint() -> tuple[int, ...]:
    """Modification times of the files the dashboard reads; used as a cache key so fresh outputs are picked up."""
    stamps: list[int] = []
    for cfg in MARKETS.values():
        for rel in ("data/processed/final_evaluation.csv", "data/raw/screening_metadata.json",
                    "data/processed/benchmark_sensitivity.csv"):
            path = cfg["path"] / rel
            stamps.append(path.stat().st_mtime_ns if path.exists() else 0)
    return tuple(stamps)


# ----------------------------------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------------------------------

def export_combined_csv(path: Path | None = None) -> Path:
    """Write the combined display table (all source columns preserved) to data/combined_results.csv."""
    df, _, issues = all_results()
    if df.empty:
        raise RuntimeError("; ".join(issues) or "no market results to export")
    out = Path(path) if path else ROOT / "data" / "combined_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    return out
