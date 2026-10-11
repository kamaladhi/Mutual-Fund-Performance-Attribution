"""
evaluator.py -- Integrated Fund Manager Skill Evaluation Engine
Task 8 of the Fund Manager Skill Detector project (US market).

The rule layer (evidence flags, scores, category classification, explanations) is deliberately
identical in markets/india/src/evaluator.py and markets/usa/src/evaluator.py, so that both markets
are classified by ONE rule set. tests/test_rule_parity.py (repository root) enforces this by
comparing the syntax trees of those functions; change a rule in both files or the test fails.

PURPOSE
-------
Synthesise the outputs from four independent analytical layers into a
final, transparent, rule-based assessment of the strength of evidence
consistent with genuine, repeatable fund-manager skill:

    1. CAPM analysis          (capm_results.csv)
    2. Circular Block Bootstrap (bootstrap_results.csv)
    3. Carhart 4-Factor Model  (factor_model_results.csv)
    4. Performance Persistence (persistence_summary.csv)

DESIGN PRINCIPLES
-----------------
  Deterministic  -- identical inputs always produce identical outputs.
  Transparent    -- every rule, threshold, weight is documented here and in
                    docs/METHODOLOGY.md section 6.
  Evidence-only  -- never claims to prove manager skill; classifies evidence
                    strength only.
  No ML          -- classification uses hard Boolean rules.
  Versioned      -- EVALUATION_VERSION tags every output row.
  Direction-aware-- only POSITIVE alpha can earn significance / skill credit.

EVALUATION VERSION
------------------
  1.0.0 -- initial release (Task 8).
  2.0.0 -- review corrections (see docs/REVIEW_REPORT.md):
             * significance credit is sign-aware (a significantly negative alpha earns none);
             * "Likely Style/Luck Driven" requires a CAPM alpha that is at least marginally
               significant (p < 0.10, positive) -- there must be outperformance to explain away;
             * "Moderate Evidence" is monotone in the evidence (factor survival alone suffices;
               sign-persistence alone needs marginal significance);
             * "majority" persistence is STRICTLY more than half of the subperiods;
             * inference uses Newey-West HAC p-values; bootstrap flag is direction-aware.
  2.1.0 -- adds a distinct candidate category for a narrow, pre-defined pattern of convergent
             but not conclusive evidence; it is not equivalent to confirmed Strong Evidence of Skill.
  2.2.0 -- renames the preliminary candidate tier to "Strong Skill Candidate (Preliminary)" for clarity;
             strict confirmed-skill criteria remain unchanged.
  2.3.0 -- adds market-level Benjamini-Hochberg q-values to the confirmed-skill gate.
             The USA market was brought onto this same rule set in the combined v2 release.

CATEGORY DEFINITIONS (6 mutually exclusive categories)
-------------------------------------------------------
  Strong Evidence of Skill
      All five core criteria satisfied:
        (a) CAPM alpha > 0
        (b) CAPM alpha significant (HAC p < 0.05)
        (c) bootstrap: alpha significantly positive at 5%
        (d) 4-factor alpha > 0 AND significant (factor_alpha_survives)
        (e) alpha positive in a strict majority of subperiods

  Strong Skill Candidate (Preliminary)
      Positive marginal CAPM evidence, positive significant four-factor alpha,
      positive alpha in at least 3/4 subperiods, and bootstrap p < 0.10. This is
      a screening tier, not confirmed skill.

  Moderate Evidence
      Any one material positive signal without the convergence required above: a significantly
      positive CAPM alpha; a surviving (positive, significant) four-factor alpha; a strict-majority
      persistence with marginally significant CAPM alpha; or a significantly positive bootstrap.

  Positive but Inconclusive
      Both alpha estimates are positive and CAPM alpha is positive in at least
      3/4 subperiods, but no higher category applies. This is directional only.

  Weak/Inconclusive
      Insufficient positive direction or evidence. Default fallback.

  Likely Style/Luck Driven
      positive, at least marginally significant CAPM alpha, of which
      alpha_reduction_pct >= 75% is absorbed by factor tilts, and the factor alpha
      does not survive.  Checked FIRST as an overriding condition.

PRECEDENCE ORDER (first match wins)
-----------------------------------
  1. Likely Style/Luck Driven  (override when style explanation is strong)
  2. Strong Evidence of Skill
  3. Strong Skill Candidate (Preliminary)
  4. Moderate Evidence
  5. Positive but Inconclusive
  6. Weak/Inconclusive  (default)

SCORING FORMULA (0-100)
-----------------------
  performance_score    (0-20)
  significance_score   (0-25)
  factor_score         (0-25)
  persistence_score    (0-20)
  style_luck_adjustment (-10 to +10)
  total_score = clamp(sum, 0, 100)
  Full formula in docs/METHODOLOGY.md section 6.

LIMITATIONS (full list in docs/METHODOLOGY.md section 7)
---------------------------------------------------------
  * Only 12 candidate funds analysed (hand-picked, well-known schemes).
  * ~5-year analysis window (2019-2023).
  * Survivorship/selection effects exist.
  * Statistical significance != causality; Benjamini-Hochberg q-values are reported and required for confirmed strong evidence.
  * Where a named manager is recorded, that manager may not have run the scheme for the whole window
    (see attribution_note); the result then describes the scheme, not the manager.
  * Bootstrap inference is block-length sensitive.
  * Four-factor model may omit relevant risk exposures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.stats_utils import benjamini_hochberg

# ---------------------------------------------------------------------------
# VERSION
# ---------------------------------------------------------------------------
EVALUATION_VERSION: str = "2.3.0"

# ---------------------------------------------------------------------------
# CATEGORY LABELS
# ---------------------------------------------------------------------------
CAT_STRONG = "Strong Evidence of Skill"
CAT_PROMISING = "Strong Skill Candidate (Preliminary)"
CAT_POSITIVE_INCONCLUSIVE = "Positive but Inconclusive"
CAT_MODERATE = "Moderate Evidence"
CAT_WEAK = "Weak/Inconclusive"
CAT_STYLE_LUCK = "Likely Style/Luck Driven"

# ---------------------------------------------------------------------------
# THRESHOLDS  (documented; change via version bump)
# ---------------------------------------------------------------------------
ALPHA_PVALUE_THRESHOLD: float = 0.05        # significance level (HAC p-values)
BOOTSTRAP_PVALUE_THRESHOLD: float = 0.05    # CBB significance level
MARGINAL_PVALUE_THRESHOLD: float = 0.10     # "marginal" evidence (also gates the style/luck label)
STYLE_REDUCTION_THRESHOLD: float = 75.0     # pct reduction triggers style label
MATERIAL_REDUCTION_THRESHOLD: float = 50.0  # pct reduction = "materially reduced"
STRONG_ALPHA_PCT: float = 3.0               # >= 3% annual -> full perf score
MODERATE_ALPHA_PCT: float = 1.0             # >= 1% annual -> partial perf score
TENURE_OVERLAP_WARN_PCT: float = 90.0       # below this the named manager did not run the scheme all window

# ---------------------------------------------------------------------------
# INPUT FILE MAP
# ---------------------------------------------------------------------------
INPUT_FILES = {
    "capm":        Path("data/processed/capm_results.csv"),
    "bootstrap":   Path("data/processed/bootstrap_results.csv"),
    "factor":      Path("data/processed/factor_model_results.csv"),
    "persistence": Path("data/processed/persistence_summary.csv"),
}
# Optional context files (used only for the manager-attribution caveat columns)
OPTIONAL_SCREENING = Path("data/raw/screening_metadata.json")
OPTIONAL_PERIODS = Path("data/processed/persistence_period_results.csv")

OUTPUT_CSV  = Path("data/processed/final_evaluation.csv")
OUTPUT_JSON = Path("data/processed/final_evaluation.json")

# ---------------------------------------------------------------------------
# COLUMN MAPPINGS  (source column -> internal name)
# Upstream column renames: update only this section.
# ---------------------------------------------------------------------------
CAPM_COL_MAP: dict[str, str] = {
    "symbol":                      "symbol",
    "n_observations":              "capm_n",
    "alpha_daily":                 "capm_alpha_daily",
    "beta":                        "capm_beta",
    "alpha_pvalue_hac":            "capm_alpha_pvalue",       # Newey-West HAC p-value (used for decisions)
    "alpha_pvalue":                "capm_alpha_pvalue_ols",   # classical OLS p-value (reported)
    "r_squared":                   "capm_r_squared",
    "alpha_daily_ci95_lower":      "capm_alpha_ci95_lower",
    "alpha_daily_ci95_upper":      "capm_alpha_ci95_upper",
    "alpha_annual_pct":            "capm_alpha_annual_pct",
    "alpha_annual_pct_ci95_lower": "capm_alpha_annual_ci95_lower",
    "alpha_annual_pct_ci95_upper": "capm_alpha_annual_ci95_upper",
}

BOOTSTRAP_COL_MAP: dict[str, str] = {
    "symbol":                      "symbol",
    "p_value":                     "bootstrap_p_value",
    "p_value_one_sided":           "bootstrap_p_value_one_sided",
    "p_value_bh":                  "bootstrap_p_value_bh",
    "significantly_positive_5pct": "bootstrap_significant",             # two-sided p < 5% AND alpha > 0
    "significant_at_5pct":         "bootstrap_two_sided_significant",
    "ci_lower":                    "bootstrap_ci_lower",
    "ci_median":                   "bootstrap_ci_median",
    "ci_upper":                    "bootstrap_ci_upper",
    "observed_alpha_annual_pct":   "bootstrap_annual_alpha_pct",
    "ci_annual_lower_pct":         "bootstrap_annual_ci_lower_pct",
    "ci_annual_upper_pct":         "bootstrap_annual_ci_upper_pct",
    "block_length":                "bootstrap_block_length",
    "n_bootstrap":                 "bootstrap_B",
}

FACTOR_COL_MAP: dict[str, str] = {
    "symbol":                     "symbol",
    "alpha_daily":                "factor_alpha_daily",
    "alpha_annual_pct":           "factor_alpha_annual_pct",
    "alpha_pvalue_hac":           "factor_alpha_pvalue",       # Newey-West HAC p-value (used for decisions)
    "alpha_pvalue":               "factor_alpha_pvalue_ols",
    "alpha_significant_hac":      "factor_alpha_significant",
    "beta_market":                "factor_market_beta",
    "beta_smb":                   "factor_smb_beta",
    "beta_hml":                   "factor_hml_beta",
    "beta_momentum":              "factor_momentum_beta",
    "r_squared":                  "factor_r_squared",
    "alpha_reduction_pct":        "alpha_reduction_pct",
    "alpha_reduction_pts_annual": "alpha_reduction_pts_annual",
}

PERSISTENCE_COL_MAP: dict[str, str] = {
    "symbol":                                      "symbol",
    "positive_alpha_persistence_rate":             "positive_alpha_persistence_rate",
    "significant_positive_alpha_persistence_rate": "significant_positive_alpha_persistence_rate",
    "alpha_mean_annual_pct":                       "alpha_mean_annual_pct",
    "alpha_median_annual_pct":                     "alpha_median_annual_pct",
    "alpha_std_annual_pct":                        "alpha_std_annual_pct",
    "alpha_positive_all_periods":                  "alpha_positive_all_periods",
    "alpha_positive_majority":                     "alpha_positive_majority",
}


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Run the upstream analytical modules before the evaluator."
        )


def _assert_columns_present(
    df: pd.DataFrame, required: list[str], label: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] Missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )


def _assert_unique_symbols(df: pd.DataFrame, label: str) -> None:
    dupes = df["symbol"][df["symbol"].duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"[{label}] Duplicate symbols: {dupes}. "
            "Each CSV must have exactly one row per fund symbol."
        )


def _assert_no_missing_numerics(
    df: pd.DataFrame, cols: list[str], label: str
) -> None:
    for col in cols:
        if col in df.columns and df[col].isna().any():
            bad = df[df[col].isna()]["symbol"].tolist()
            raise ValueError(
                f"[{label}] Missing values in '{col}' for funds: {bad}"
            )


def _assert_pvalue_range(
    df: pd.DataFrame, cols: list[str], label: str
) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        out = df[(df[col] < 0) | (df[col] > 1)]["symbol"].tolist()
        if out:
            raise ValueError(
                f"[{label}] '{col}' out of [0,1] for funds: {out}"
            )


def _assert_rate_range(
    df: pd.DataFrame, cols: list[str], label: str
) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        out = df[(df[col] < 0) | (df[col] > 1)]["symbol"].tolist()
        if out:
            raise ValueError(
                f"[{label}] '{col}' out of [0,1] for funds: {out}"
            )


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def _upgrade_bootstrap_schema(boot: pd.DataFrame) -> pd.DataFrame:
    """
    The USA bootstrap engine pre-dates three columns that the shared rule set reads. Derive them
    exactly as the India engine defines them (nothing is re-estimated; the saved p-values are used):

      significantly_positive_5pct  two-sided p < 0.05 AND observed alpha > 0
      p_value_bh                   Benjamini-Hochberg q-value of the two-sided p-values (fund universe)
      p_value_one_sided            not recoverable from the saved summary (needs the resample draws): NaN
    """
    boot = boot.copy()
    if "significantly_positive_5pct" not in boot.columns:
        boot["significantly_positive_5pct"] = (
            boot["significant_at_5pct"].astype(bool) & (boot["observed_alpha_daily"] > 0)
        )
    if "p_value_bh" not in boot.columns:
        boot["p_value_bh"] = benjamini_hochberg(pd.to_numeric(boot["p_value"], errors="coerce").fillna(1.0).to_numpy())
    if "p_value_one_sided" not in boot.columns:
        boot["p_value_one_sided"] = np.nan
    return boot


def _upgrade_factor_schema(fact: pd.DataFrame) -> pd.DataFrame:
    """Derive the two factor-model columns the shared rule set reads, as the India engine defines them."""
    fact = fact.copy()
    if "alpha_significant_hac" not in fact.columns:
        fact["alpha_significant_hac"] = fact["alpha_pvalue_hac"] < ALPHA_PVALUE_THRESHOLD
    if "alpha_reduction_pts_annual" not in fact.columns:
        fact["alpha_reduction_pts_annual"] = fact["capm_alpha_annual_pct"] - fact["alpha_annual_pct"]
    return fact


def load_and_validate_inputs(
    project_root: "str | Path" = ".",
    return_source_hashes: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """
    Load all four result CSVs, rename columns, validate, and join.

    Parameters
    ----------
    project_root : path-like
        Root of the project (parent of data/).
    return_source_hashes : bool
        If True, return (merged_df, hashes_dict) for immutability tests.

    Returns
    -------
    pd.DataFrame  (or tuple if return_source_hashes=True)

    Raises
    ------
    FileNotFoundError
        Any input CSV is missing.
    ValueError
        Validation failures: missing columns, duplicates, bad joins, NaN
        numerics, out-of-range p-values or rates.
    """
    root = Path(project_root)

    # 1. Existence
    for key, rel_path in INPUT_FILES.items():
        _assert_file_exists(root / rel_path)

    # 2. Load
    capm_raw = pd.read_csv(root / INPUT_FILES["capm"])
    boot_raw = pd.read_csv(root / INPUT_FILES["bootstrap"])
    fact_raw = pd.read_csv(root / INPUT_FILES["factor"])
    pers_raw = pd.read_csv(root / INPUT_FILES["persistence"])

    for df_item in [capm_raw, boot_raw, fact_raw, pers_raw]:
        if "symbol" in df_item.columns:
            df_item["symbol"] = df_item["symbol"].astype(str)

    # Hash source files for immutability testing
    source_hashes = {
        "capm":        hash(capm_raw.to_string()),
        "bootstrap":   hash(boot_raw.to_string()),
        "factor":      hash(fact_raw.to_string()),
        "persistence": hash(pers_raw.to_string()),
    }

    # 2b. Derive the columns this engine's older outputs lack (see the helpers above)
    boot_raw = _upgrade_bootstrap_schema(boot_raw)
    fact_raw = _upgrade_factor_schema(fact_raw)

    # 3. Column validation
    _assert_columns_present(capm_raw, list(CAPM_COL_MAP),        "capm_results.csv")
    _assert_columns_present(boot_raw, list(BOOTSTRAP_COL_MAP),   "bootstrap_results.csv")
    _assert_columns_present(fact_raw, list(FACTOR_COL_MAP),      "factor_model_results.csv")
    _assert_columns_present(pers_raw, list(PERSISTENCE_COL_MAP), "persistence_summary.csv")

    # 4. Select and rename
    capm = capm_raw[list(CAPM_COL_MAP)].rename(columns=CAPM_COL_MAP)
    boot = boot_raw[list(BOOTSTRAP_COL_MAP)].rename(columns=BOOTSTRAP_COL_MAP)
    fact = fact_raw[list(FACTOR_COL_MAP)].rename(columns=FACTOR_COL_MAP)
    pers = pers_raw[list(PERSISTENCE_COL_MAP)].rename(columns=PERSISTENCE_COL_MAP)

    # 5. Uniqueness
    for df, label in [
        (capm, "CAPM"), (boot, "Bootstrap"),
        (fact, "Factor"), (pers, "Persistence"),
    ]:
        _assert_unique_symbols(df, label)

    # 6. Merge outer (catches missing funds across files)
    merged = capm.merge(boot, on="symbol", how="outer")
    merged = merged.merge(fact, on="symbol", how="outer")
    merged = merged.merge(pers, on="symbol", how="outer")

    # 7. Join integrity check
    join_fail_mask = (
        merged["capm_alpha_daily"].isna() |
        merged["bootstrap_p_value"].isna() |
        merged["factor_alpha_daily"].isna() |
        merged["positive_alpha_persistence_rate"].isna()
    )
    if join_fail_mask.any():
        bad = merged.loc[join_fail_mask, "symbol"].tolist()
        raise ValueError(
            f"Join failure: funds missing from >=1 source file: {bad}\n"
            "Ensure all four CSVs contain entries for every fund."
        )

    merged = merged.reset_index(drop=True)

    # Cross-sectional false-discovery-rate q-values, computed separately for
    # each family of fund-level tests within this market.
    for p_col, q_col in [
        ("capm_alpha_pvalue", "capm_alpha_pvalue_bh"),
        ("bootstrap_p_value", "bootstrap_p_value_bh_computed"),
        ("factor_alpha_pvalue", "factor_alpha_pvalue_bh"),
    ]:
        if p_col in merged.columns:
            merged[q_col] = benjamini_hochberg(pd.to_numeric(merged[p_col], errors="coerce").fillna(1.0).to_numpy())

    # 8. Numeric / range checks
    _assert_no_missing_numerics(merged, [
        "capm_alpha_daily", "capm_alpha_annual_pct", "capm_alpha_pvalue", "capm_alpha_pvalue_ols",
        "capm_beta", "capm_r_squared",
        "bootstrap_p_value", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "factor_alpha_daily", "factor_alpha_annual_pct", "factor_alpha_pvalue", "factor_alpha_pvalue_ols",
        "positive_alpha_persistence_rate",
        "significant_positive_alpha_persistence_rate",
    ], "merged")

    _assert_pvalue_range(merged, [
        "capm_alpha_pvalue", "capm_alpha_pvalue_ols", "bootstrap_p_value", "bootstrap_p_value_one_sided",
        "bootstrap_p_value_bh", "factor_alpha_pvalue", "factor_alpha_pvalue_ols",
    ], "merged")

    _assert_rate_range(merged, [
        "positive_alpha_persistence_rate",
        "significant_positive_alpha_persistence_rate",
    ], "merged")

    if return_source_hashes:
        return merged, source_hashes
    return merged


def load_attribution_context(project_root: "str | Path" = ".") -> pd.DataFrame:
    """
    Optional context for the manager-attribution caveat: named manager, tenure start, and the share of
    the analysis window during which that manager was in charge. Empty frame if metadata is absent.
    """
    root = Path(project_root)
    screening_path = root / OPTIONAL_SCREENING
    periods_path = root / OPTIONAL_PERIODS
    if not screening_path.exists():
        return pd.DataFrame(columns=["symbol"])

    with open(screening_path, "r", encoding="utf-8") as fh:
        screening = json.load(fh)

    # The US registry records no named managers or tenure dates: there is nothing to attribute.
    if not any(rec.get("tenure_start") for rec in screening):
        return pd.DataFrame(columns=["symbol"])

    sample_start = sample_end = None
    if periods_path.exists():
        periods = pd.read_csv(periods_path)
        sample_start = pd.Timestamp(periods["start_date"].min())
        sample_end = pd.Timestamp(periods["end_date"].max())

    rows = []
    for rec in screening:
        symbol = str(rec.get("ticker"))
        tenure_start = pd.to_datetime(rec.get("tenure_start"), errors="coerce")
        overlap_pct = np.nan
        if sample_start is not None and pd.notna(tenure_start):
            window_days = max((sample_end - sample_start).days, 1)
            covered_start = max(tenure_start, sample_start)
            overlap_pct = 100.0 * max((sample_end - covered_start).days, 0) / window_days
        note = ""
        if pd.notna(overlap_pct) and overlap_pct < TENURE_OVERLAP_WARN_PCT:
            note = (f"Named manager ({rec.get('lead_manager')}) in charge for only {overlap_pct:.0f}% of the "
                    "sample window; the result describes the scheme, not that manager.")
        if rec.get("notes"):
            note = (note + " " + str(rec["notes"])).strip()
        rows.append({
            "symbol": symbol,
            "lead_manager": rec.get("lead_manager", ""),
            "tenure_start": rec.get("tenure_start", ""),
            "tenure_overlap_pct": overlap_pct,
            "attribution_note": note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EVIDENCE FLAGS
# ---------------------------------------------------------------------------

def compute_evidence_flags(row: pd.Series) -> dict[str, bool]:
    """
    Derive all Boolean evidence flags for a single fund row.

    Rules
    -----
    capm_positive
        CAPM annual alpha > 0

    capm_significant
        capm_positive AND CAPM alpha p-value (Newey-West HAC) < ALPHA_PVALUE_THRESHOLD (0.05)
        i.e. alpha is significantly POSITIVE.  (Significantly negative alpha is reported
        separately as capm_significantly_negative and earns no credit.)

    capm_marginally_significant
        capm_positive AND p-value < MARGINAL_PVALUE_THRESHOLD (0.10)

    capm_significantly_negative
        CAPM annual alpha < 0 AND p-value < 0.05  (evidence of under-performance)

    bootstrap_significant
        bootstrap flag "significantly positive at 5%" (two-sided p < 0.05 AND alpha > 0)
        OR (bootstrap p_value < 0.05 AND capm_positive)
        (belt-and-suspenders guard that is direction-aware)

    bootstrap_ci_positive
        bootstrap CI lower bound > 0
        (the entire 95% CI lies above zero)

    factor_alpha_positive
        four-factor annual alpha > 0

    factor_alpha_significant
        factor model alpha_significant flag (HAC) is True
        OR factor alpha p-value < ALPHA_PVALUE_THRESHOLD

    factor_alpha_survives
        factor_alpha_positive AND factor_alpha_significant
        (alpha remains positive AND statistically significant after
         controlling for market, size, value, and momentum exposures)

    alpha_materially_reduced
        alpha_reduction_pct >= MATERIAL_REDUCTION_THRESHOLD (50%)
        (factor adjustment removed >= 50% of raw CAPM alpha magnitude)

    style_explanation
        capm_marginally_significant AND alpha_materially_reduced
        (there is a noteworthy positive CAPM alpha, and factor exposures explain much of it)

    persistent_positive_majority
        alpha_positive_majority flag from persistence engine (STRICT majority of subperiods)

    persistent_all_periods
        alpha_positive_all_periods flag from persistence engine
    """
    red_pct = float(row["alpha_reduction_pct"]) if not pd.isna(row["alpha_reduction_pct"]) else 0.0

    capm_positive      = bool(row["capm_alpha_annual_pct"] > 0)
    capm_p             = float(row["capm_alpha_pvalue"])
    capm_significant   = bool(capm_positive and capm_p < ALPHA_PVALUE_THRESHOLD)
    capm_marginal      = bool(capm_positive and capm_p < MARGINAL_PVALUE_THRESHOLD)
    capm_sig_negative  = bool((not capm_positive) and row["capm_alpha_annual_pct"] < 0
                              and capm_p < ALPHA_PVALUE_THRESHOLD)

    bootstrap_significant = bool(
        capm_positive and (
            row["bootstrap_significant"]
            or row["bootstrap_p_value"] < BOOTSTRAP_PVALUE_THRESHOLD
        )
    )
    bootstrap_ci_positive = bool(row["bootstrap_ci_lower"] > 0)

    factor_alpha_positive    = bool(row["factor_alpha_annual_pct"] > 0)
    factor_alpha_significant = bool(
        row["factor_alpha_significant"]
        or row["factor_alpha_pvalue"] < ALPHA_PVALUE_THRESHOLD
    )
    factor_alpha_survives    = bool(factor_alpha_positive and factor_alpha_significant)

    alpha_materially_reduced = bool(red_pct >= MATERIAL_REDUCTION_THRESHOLD)
    style_explanation        = bool(capm_marginal and alpha_materially_reduced)

    positive_alpha_majority    = bool(row["alpha_positive_majority"])
    positive_alpha_all_periods = bool(row["alpha_positive_all_periods"])

    return {
        "capm_positive":                capm_positive,
        "capm_significant":             capm_significant,
        "capm_marginally_significant":  capm_marginal,
        "capm_significantly_negative":  capm_sig_negative,
        "bootstrap_significant":        bootstrap_significant,
        "bootstrap_ci_positive":        bootstrap_ci_positive,
        "factor_alpha_positive":        factor_alpha_positive,
        "factor_alpha_significant":     factor_alpha_significant,
        "factor_alpha_survives":        factor_alpha_survives,
        "alpha_materially_reduced":     alpha_materially_reduced,
        "style_explanation":            style_explanation,
        "persistent_positive_majority": positive_alpha_majority,
        "persistent_all_periods":       positive_alpha_all_periods,
    }


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def compute_scores(
    row: pd.Series,
    flags: dict[str, bool],
) -> dict[str, float]:
    """
    Compute transparent evidence scores (total 0-100).

    All thresholds and weights are documented below and in
    docs/METHODOLOGY.md section 6.

    PERFORMANCE SCORE (0-20)
    ------------------------
    Rewards magnitude AND sign of CAPM annualised alpha.
      alpha <= 0          -> 0
      0 < alpha < 1%      -> 10
      1% <= alpha < 3%    -> 15
      alpha >= 3%         -> 20

    SIGNIFICANCE SCORE (0-25)   (only POSITIVE alpha earns credit)
    -------------------------
    HAC component (0-15):  [0 unless CAPM alpha > 0]
      p < 0.01  -> 15
      p < 0.05  -> 10
      p < 0.10  -> 5
      else      -> 0
    Bootstrap component (0-10):
      both bootstrap_significant AND bootstrap_ci_positive -> 10
      one of the above                                    -> 5
      neither                                             -> 0

    FACTOR SCORE (0-25)
    -------------------
    factor_alpha_survives (positive AND significant) -> 25
    factor_alpha_positive only                       -> 10
    else                                             -> 0

    PERSISTENCE SCORE (0-20)
    ------------------------
    alpha_positive_all_periods  -> 20
    alpha_positive_majority     -> 12     (strict majority of subperiods)
    pos_persistence_rate >= 0.5 -> 6      (exactly half)
    else                        -> 0
    Bonus: +3 if sig_persistence_rate >= 0.5 (capped at 20)

    STYLE/LUCK ADJUSTMENT (-10 to +10)
    -----------------------------------
    Applies only when CAPM alpha is positive and marginally significant (p < 0.10):
      red_pct >= 75%  -> -10
      red_pct >= 50%  -> -5
      red_pct >= 25%  -> -2
    Otherwise: red_pct < 25% AND factor_survives -> +5, else 0

    TOTAL = clamp(sum, 0, 100)
    """
    ann_alpha     = float(row["capm_alpha_annual_pct"])
    capm_p        = float(row["capm_alpha_pvalue"])
    red_pct       = float(row["alpha_reduction_pct"]) if not pd.isna(row["alpha_reduction_pct"]) else 0.0
    pos_pers_rate = float(row["positive_alpha_persistence_rate"])
    sig_pers_rate = float(row["significant_positive_alpha_persistence_rate"])

    # performance
    if ann_alpha <= 0:
        perf = 0.0
    elif ann_alpha < MODERATE_ALPHA_PCT:
        perf = 10.0
    elif ann_alpha < STRONG_ALPHA_PCT:
        perf = 15.0
    else:
        perf = 20.0

    # significance (direction-aware: no credit for a significantly negative alpha)
    if not flags["capm_positive"]:
        ols = 0.0
    elif capm_p < 0.01:
        ols = 15.0
    elif capm_p < 0.05:
        ols = 10.0
    elif capm_p < 0.10:
        ols = 5.0
    else:
        ols = 0.0

    if flags["bootstrap_significant"] and flags["bootstrap_ci_positive"]:
        boot = 10.0
    elif flags["bootstrap_significant"] or flags["bootstrap_ci_positive"]:
        boot = 5.0
    else:
        boot = 0.0

    sig = ols + boot  # max 25

    # factor
    if flags["factor_alpha_survives"]:
        fac = 25.0
    elif flags["factor_alpha_positive"]:
        fac = 10.0
    else:
        fac = 0.0

    # persistence
    if flags["persistent_all_periods"]:
        pers = 20.0
    elif flags["persistent_positive_majority"]:
        pers = 12.0
    elif pos_pers_rate >= 0.5:
        pers = 6.0
    else:
        pers = 0.0
    if sig_pers_rate >= 0.5:
        pers = min(20.0, pers + 3.0)

    # style/luck adjustment
    eligible = flags["capm_marginally_significant"]
    if eligible and red_pct >= 75.0:
        sla = -10.0
    elif eligible and red_pct >= 50.0:
        sla = -5.0
    elif eligible and red_pct >= 25.0:
        sla = -2.0
    elif red_pct < 25.0 and flags["factor_alpha_survives"]:
        sla = 5.0
    else:
        sla = 0.0

    total = float(np.clip(perf + sig + fac + pers + sla, 0.0, 100.0))

    return {
        "performance_score":     round(perf, 4),
        "significance_score":    round(sig, 4),
        "factor_score":          round(fac, 4),
        "persistence_score":     round(pers, 4),
        "style_luck_adjustment": round(sla, 4),
        "total_score":           round(total, 4),
    }


# ---------------------------------------------------------------------------
# CATEGORY CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_fund(
    flags: dict[str, bool],
    scores: dict[str, float],
    row: pd.Series,
) -> str:
    """
    Assign one of six mutually exclusive evidence categories (first match wins; see the module
    docstring for the precedence order and the plain-language definition of each).

    "Strong Skill Candidate (Preliminary)" is a screening tier, not proof of skill. It
    requires positive CAPM alpha with marginal HAC evidence, positive and
    statistically significant four-factor alpha, alpha in a strict majority
    of subperiods, and a bootstrap p-value below 0.10. The strict Strong
    Evidence category remains unchanged and requires all five core criteria.
    """
    red_pct = float(row["alpha_reduction_pct"]) if not pd.isna(row["alpha_reduction_pct"]) else 0.0
    bootstrap_p = float(row["bootstrap_p_value"]) if not pd.isna(row["bootstrap_p_value"]) else 1.0

    # Style/luck override: there must be positive outperformance to explain.
    if (flags["capm_marginally_significant"]
            and red_pct >= STYLE_REDUCTION_THRESHOLD
            and not (flags["factor_alpha_positive"] and flags["factor_alpha_significant"])):
        return CAT_STYLE_LUCK

    # Strict confirmed-evidence tier. Criteria are unchanged from v2.0.0.
    if (flags["capm_positive"]
            and flags["capm_significant"]
            and flags["bootstrap_significant"]
            and flags["factor_alpha_survives"]
            and flags["persistent_positive_majority"]
            and float(row.get("capm_alpha_pvalue_bh", 1.0)) < 0.10
            and float(row.get("factor_alpha_pvalue_bh", 1.0)) < 0.10
            and float(row.get("bootstrap_p_value_bh_computed", row.get("bootstrap_p_value_bh", 1.0))) < 0.10):
        return CAT_STRONG

    # Screening tier for a coherent but not fully robust signal.
    # Bootstrap p < 0.10 is deliberately weaker than the strict 5% test;
    # the label explicitly says "Potential Candidate" and does not imply proof.
    if (flags["capm_positive"]
            and flags["capm_marginally_significant"]
            and flags["factor_alpha_survives"]
            and flags["persistent_positive_majority"]
            and bootstrap_p < 0.10):
        return CAT_PROMISING

    # Moderate evidence: one or more material positive signals, without the
    # complete convergence required for either higher tier.
    if (
        (flags["capm_positive"] and flags["capm_significant"])
        or flags["factor_alpha_survives"]
        or (flags["persistent_positive_majority"] and flags["capm_marginally_significant"])
        or (flags["capm_positive"] and flags["bootstrap_significant"])
    ):
        return CAT_MODERATE

    # Directionally positive signal, explicitly not statistically conclusive:
    # both full-sample alpha estimates are positive and CAPM alpha is positive
    # in a strict majority of subperiods. This is descriptive, not "skill".
    if (flags["capm_positive"]
            and flags["factor_alpha_positive"]
            and flags["persistent_positive_majority"]):
        return CAT_POSITIVE_INCONCLUSIVE

    return CAT_WEAK


def _primary_strengths(flags: dict[str, bool]) -> list[str]:
    """Return list of evidence strengths (machine-readable sentences)."""
    s = []
    if flags["capm_positive"]:
        s.append("Positive CAPM alpha observed.")
    if flags["capm_significant"]:
        s.append("CAPM alpha is significantly positive (Newey-West HAC p < 0.05).")
    if flags["bootstrap_significant"]:
        s.append("Bootstrap inference supports significance of alpha (CBB p < 0.05, alpha > 0).")
    if flags["bootstrap_ci_positive"]:
        s.append("Bootstrap 95% CI for alpha excludes zero from below.")
    if flags["factor_alpha_survives"]:
        s.append(
            "Four-factor alpha remains positive and statistically significant "
            "after controlling for market, size, value, and momentum exposures."
        )
    elif flags["factor_alpha_positive"]:
        s.append("Four-factor alpha is positive, though not statistically significant.")
    if flags["persistent_all_periods"]:
        s.append("Alpha is positive in all four non-overlapping subperiods.")
    elif flags["persistent_positive_majority"]:
        s.append("Alpha is positive in a majority (3 of 4) of non-overlapping subperiods.")
    return s


def _primary_weaknesses(flags: dict[str, bool]) -> list[str]:
    """Return list of evidence gaps (machine-readable sentences)."""
    w = []
    if flags["capm_significantly_negative"]:
        w.append("CAPM alpha is significantly negative (evidence of under-performance).")
    elif not flags["capm_positive"]:
        w.append("CAPM alpha is non-positive.")
    if not flags["capm_significant"] and not flags["capm_significantly_negative"]:
        w.append("CAPM alpha is not statistically significant at the 5% level.")
    if not flags["bootstrap_significant"]:
        w.append("Bootstrap inference does not support significantly positive alpha.")
    if not flags["bootstrap_ci_positive"]:
        w.append("Bootstrap 95% CI for alpha includes or is below zero.")
    if not flags["factor_alpha_positive"]:
        w.append("Four-factor alpha is negative.")
    elif not flags["factor_alpha_significant"]:
        w.append("Four-factor alpha is positive but not statistically significant.")
    if flags["style_explanation"]:
        w.append(
            "Factor adjustment materially reduces CAPM alpha, suggesting systematic "
            "style exposures partially explain apparent outperformance."
        )
    if not flags["persistent_positive_majority"]:
        w.append(
            "Alpha is not positive in a majority (at least 3 of 4) of non-overlapping subperiods."
        )
    return w


def generate_explanation(
    category: str,
    flags: dict[str, bool],
    scores: dict[str, float],
    row: pd.Series,
) -> str:
    """
    Generate a deterministic rule-based explanation string.

    Language policy:
      - Evidence-oriented, not causal.
      - Does NOT state "the manager has skill" or "caused by the manager".
      - Identical flags always produce identical text.
    """
    ann_alpha = float(row["capm_alpha_annual_pct"])
    fac_alpha = float(row["factor_alpha_annual_pct"])
    red_pct   = float(row["alpha_reduction_pct"]) if not pd.isna(row["alpha_reduction_pct"]) else 0.0
    a_str     = f"{ann_alpha:+.2f}%"
    f_str     = f"{fac_alpha:+.2f}%"
    pts_str   = f"{ann_alpha - fac_alpha:.2f} percentage points"
    r_str     = f"{red_pct:.1f}%"

    if category == CAT_STYLE_LUCK:
        factor_status = (
            "positive but not statistically significant"
            if flags["factor_alpha_positive"] else "negative"
        )
        return (
            f"CAPM annual alpha of {a_str} falls to {f_str} after four-factor adjustment "
            f"(a loss of {pts_str}, i.e. {r_str} of the CAPM alpha), leaving the factor-adjusted alpha "
            f"{factor_status}. "
            "The observed outperformance is more consistent with systematic factor "
            "exposures (market, size, value, or momentum tilts) than with "
            "independent, persistent manager-specific alpha. "
            "Classification: Likely Style/Luck Driven."
        )

    if category == CAT_STRONG:
        return (
            f"CAPM annual alpha of {a_str} is statistically significant "
            "(HAC and bootstrap). "
            "Four-factor analysis confirms a positive, statistically significant "
            "alpha after controlling for systematic exposures, and positive alpha "
            "is observed in a majority of non-overlapping subperiods. "
            "All five core evidence criteria are satisfied. "
            "Classification: Strong Evidence of Skill. "
            "Note: this classification reflects the strength of statistical "
            "evidence; it does not establish that skill caused the outperformance."
        )

    if category == CAT_PROMISING:
        if float(row["bootstrap_p_value"]) >= BOOTSTRAP_PVALUE_THRESHOLD:
            boot_text = "block bootstrap p-value is below 0.10 but not below the strict 0.05 threshold. "
        else:
            boot_text = ("block bootstrap p-value is below 0.05, but at least one strict criterion for "
                         "Strong Evidence (HAC p < 0.05 or BH q < 0.10) is not met. ")
        return (
            f"CAPM annual alpha of {a_str} is positive with marginal HAC evidence; "
            f"four-factor alpha remains positive and statistically significant at {f_str}; "
            "alpha is positive in at least three of four subperiods, and the circular "
            + boot_text +
            "This convergence makes the fund a promising candidate for further review, "
            "not a confirmed demonstration of manager skill. "
            "Classification: Strong Skill Candidate (Preliminary)."
        )

    if category == CAT_POSITIVE_INCONCLUSIVE:
        return (
            f"CAPM annual alpha of {a_str} and four-factor alpha of {f_str} are both positive, "
            "and CAPM alpha is positive in at least three of four subperiods. "
            "However, neither the HAC/ bootstrap evidence nor factor-adjusted inference "
            "meets the higher-tier criteria. This is a directionally positive signal, "
            "not statistically established manager skill. "
            "Classification: Positive but Inconclusive."
        )

    if category == CAT_MODERATE:
        parts: list[str] = []
        if flags["capm_positive"] and flags["capm_significant"]:
            parts.append(f"CAPM alpha of {a_str} is positive and statistically significant.")
        elif flags["capm_positive"]:
            parts.append(f"CAPM alpha of {a_str} is positive but not statistically significant at the 5% level.")
        else:
            parts.append(f"CAPM alpha of {a_str} is not positive.")
        if flags["factor_alpha_survives"]:
            parts.append(
                "Four-factor alpha remains positive and significant after factor adjustment."
            )
        elif flags["factor_alpha_positive"]:
            parts.append(
                "Four-factor alpha is positive but does not reach statistical significance."
            )
        else:
            parts.append(
                f"Four-factor adjustment reduces CAPM alpha by {r_str}; "
                "the factor-adjusted alpha is not positive."
            )
        if flags["persistent_positive_majority"]:
            parts.append("Alpha is positive in a majority of subperiods.")
        else:
            parts.append("Persistence evidence is limited or inconclusive.")
        if flags["bootstrap_significant"]:
            parts.append("Bootstrap inference supports alpha significance.")
        else:
            parts.append("Bootstrap inference does not support alpha significance.")
        return (
            " ".join(parts) +
            " One or more major evidence criteria are incomplete. "
            "Classification: Moderate Evidence."
        )

    # WEAK / INCONCLUSIVE (default)
    if flags["capm_significantly_negative"]:
        intro = (
            f"CAPM alpha of {a_str} is significantly negative: the evidence points to "
            "under-performance relative to the market risk premium, not outperformance."
        )
    elif flags["capm_positive"]:
        intro = (
            f"CAPM alpha of {a_str} is positive but statistical support is "
            "insufficient across the HAC, bootstrap, factor adjustment, and "
            "persistence dimensions."
        )
    else:
        intro = (
            f"CAPM alpha of {a_str} is non-positive. "
            "Evidence does not support outperformance relative to the market "
            "risk premium."
        )
    if flags["style_explanation"]:
        intro += (
            f" Factor adjustment reduces CAPM alpha by {r_str}, "
            "suggesting systematic style exposures are a significant contributor."
        )
    return (
        intro
        + " Evidence is insufficient to distinguish this performance from "
        "random variation or market/style exposures. "
        "Classification: Weak/Inconclusive."
    )


# ---------------------------------------------------------------------------
# SINGLE-FUND EVALUATION
# ---------------------------------------------------------------------------

def _opt(row: pd.Series, key: str, default: float = float("nan")) -> float:
    """Optional numeric field (absent in synthetic test rows)."""
    value = row.get(key, default)
    return float(value) if not pd.isna(value) else float("nan")


def evaluate_fund(row: pd.Series) -> dict[str, Any]:
    """
    Run the full evaluation pipeline for a single fund row.

    Computes flags -> scores -> category -> explanation.
    Validates score bounds before returning.

    Parameters
    ----------
    row : pd.Series
        A single fund row from the merged DataFrame produced by
        load_and_validate_inputs().

    Returns
    -------
    dict
        All output fields for final_evaluation.csv / .json.
    """
    flags       = compute_evidence_flags(row)
    scores      = compute_scores(row, flags)
    category    = classify_fund(flags, scores, row)
    strengths   = _primary_strengths(flags)
    weaknesses  = _primary_weaknesses(flags)
    explanation = generate_explanation(category, flags, scores, row)

    # Score bound validation
    bounds = {
        "performance_score":     (0, 20),
        "significance_score":    (0, 25),
        "factor_score":          (0, 25),
        "persistence_score":     (0, 20),
        "style_luck_adjustment": (-10, 10),
        "total_score":           (0, 100),
    }
    for key, (lo, hi) in bounds.items():
        val = scores[key]
        if not (lo <= val <= hi):
            raise ValueError(
                f"[Score bounds] {key}={val} outside [{lo},{hi}] "
                f"for fund {row['symbol']}."
            )

    red_safe = (
        float(row["alpha_reduction_pct"])
        if not pd.isna(row["alpha_reduction_pct"])
        else float("nan")
    )

    return {
        # identifiers
        "symbol":             str(row["symbol"]),
        "evaluation_version": EVALUATION_VERSION,
        # CAPM
        "capm_alpha_daily":      float(row["capm_alpha_daily"]),
        "capm_alpha_annual_pct": float(row["capm_alpha_annual_pct"]),
        "capm_beta":             float(row["capm_beta"]),
        "capm_r_squared":        float(row["capm_r_squared"]),
        "capm_alpha_pvalue":     float(row["capm_alpha_pvalue"]),
        "capm_alpha_pvalue_bh":  _opt(row, "capm_alpha_pvalue_bh"),
        "capm_alpha_pvalue_ols": _opt(row, "capm_alpha_pvalue_ols"),
        # bootstrap
        "bootstrap_p_value":     float(row["bootstrap_p_value"]),
        "bootstrap_p_value_one_sided": _opt(row, "bootstrap_p_value_one_sided"),
        "bootstrap_p_value_bh":  _opt(row, "bootstrap_p_value_bh"),
        "bootstrap_p_value_bh_computed": _opt(row, "bootstrap_p_value_bh_computed"),
        "bootstrap_ci_lower":    float(row["bootstrap_ci_lower"]),
        "bootstrap_ci_upper":    float(row["bootstrap_ci_upper"]),
        "bootstrap_significant": bool(row["bootstrap_significant"]),
        # factor model
        "factor_alpha_daily":      float(row["factor_alpha_daily"]),
        "factor_alpha_annual_pct": float(row["factor_alpha_annual_pct"]),
        "factor_alpha_pvalue":     float(row["factor_alpha_pvalue"]),
        "factor_alpha_pvalue_bh":  _opt(row, "factor_alpha_pvalue_bh"),
        "factor_alpha_pvalue_ols": _opt(row, "factor_alpha_pvalue_ols"),
        "factor_alpha_significant": bool(row["factor_alpha_significant"]),
        "alpha_reduction_pct":     red_safe,
        "alpha_reduction_pts_annual": float(row["capm_alpha_annual_pct"]) - float(row["factor_alpha_annual_pct"]),
        "factor_r_squared":        float(row["factor_r_squared"]),
        # persistence
        "positive_alpha_persistence_rate":             float(row["positive_alpha_persistence_rate"]),
        "significant_positive_alpha_persistence_rate": float(row["significant_positive_alpha_persistence_rate"]),
        "alpha_mean_annual_pct":   float(row["alpha_mean_annual_pct"]),
        "alpha_median_annual_pct": float(row["alpha_median_annual_pct"]),
        "alpha_std_annual_pct":    float(row["alpha_std_annual_pct"]),
        "alpha_positive_majority": bool(row["alpha_positive_majority"]),
        "alpha_positive_all_periods": bool(row["alpha_positive_all_periods"]),
        # evidence flags
        "capm_positive":                 flags["capm_positive"],
        "capm_significant":              flags["capm_significant"],
        "capm_marginally_significant":   flags["capm_marginally_significant"],
        "capm_significantly_negative":   flags["capm_significantly_negative"],
        "bootstrap_significant_flag":    flags["bootstrap_significant"],
        "bootstrap_ci_positive":         flags["bootstrap_ci_positive"],
        "factor_alpha_positive":         flags["factor_alpha_positive"],
        "factor_alpha_significant_flag": flags["factor_alpha_significant"],
        "factor_alpha_survives":         flags["factor_alpha_survives"],
        "alpha_materially_reduced":      flags["alpha_materially_reduced"],
        "style_explanation":             flags["style_explanation"],
        "persistent_positive_majority":  flags["persistent_positive_majority"],
        "persistent_all_periods":        flags["persistent_all_periods"],
        # scores
        "performance_score":     scores["performance_score"],
        "significance_score":    scores["significance_score"],
        "factor_score":          scores["factor_score"],
        "persistence_score":     scores["persistence_score"],
        "style_luck_adjustment": scores["style_luck_adjustment"],
        "total_score":           scores["total_score"],
        # final verdict
        "final_category":    category,
        "primary_strengths": "; ".join(strengths)  if strengths  else "None identified.",
        "primary_weaknesses": "; ".join(weaknesses) if weaknesses else "None identified.",
        "final_explanation": explanation,
    }


# ---------------------------------------------------------------------------
# FULL EVALUATION RUN
# ---------------------------------------------------------------------------

def run_evaluation(project_root: "str | Path" = ".") -> pd.DataFrame:
    """
    Run the complete evaluation pipeline for all funds.

    Steps:
    1. Load and validate all four input CSVs.
    2. Compute flags, scores, and category per fund.
    3. Validate output constraints (one row/fund, score bounds, no nulls).
    4. Append manager-attribution caveat columns (if screening metadata is available).
    5. Write final_evaluation.csv and final_evaluation.json.
    6. Return results DataFrame.

    Parameters
    ----------
    project_root : path-like
        Root directory of the project.

    Returns
    -------
    pd.DataFrame  (one row per fund, all evaluation columns).
    """
    root   = Path(project_root)
    merged = load_and_validate_inputs(root)

    records: list[dict[str, Any]] = [
        evaluate_fund(row) for _, row in merged.iterrows()
    ]
    results = pd.DataFrame(records)

    context = load_attribution_context(root)
    if not context.empty:
        results = results.merge(context, on="symbol", how="left")
        results["attribution_note"] = results["attribution_note"].fillna("")

    # Output invariants
    assert len(results) == len(merged), "Row count mismatch after evaluation."
    assert results["symbol"].nunique() == len(results), "Duplicate symbols in output."
    assert results["final_category"].notna().all(), "Missing final_category."
    assert results["final_explanation"].notna().all(), "Missing final_explanation."
    assert (
        (results["total_score"] >= 0) & (results["total_score"] <= 100)
    ).all(), "total_score outside [0,100]."

    # Write CSV
    out_csv  = root / OUTPUT_CSV
    out_json = root / OUTPUT_JSON
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)

    # Write JSON (numpy-type-safe)
    def _safe(obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    json_records = [{k: _safe(v) for k, v in rec.items()} for rec in results.to_dict("records")]
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(json_records, fh, indent=2, ensure_ascii=False)

    return results


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame) -> None:
    """Print a concise evaluation summary to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  US FUND MANAGER SKILL DETECTOR -- Integrated Evaluation Summary")
    print(f"  Evaluation Version: {EVALUATION_VERSION}")
    print(sep)

    cat_order  = [CAT_STRONG, CAT_PROMISING, CAT_MODERATE, CAT_POSITIVE_INCONCLUSIVE, CAT_WEAK, CAT_STYLE_LUCK]
    cat_counts = results["final_category"].value_counts()

    print(f"\n  Funds evaluated : {len(results)}")
    print(
        f"  Score range     : "
        f"{results['total_score'].min():.1f} - {results['total_score'].max():.1f}"
    )
    print("\n  Category distribution:")
    for cat in cat_order:
        print(f"    {cat:<35}: {cat_counts.get(cat, 0)}")

    print("\n  Per-fund results (by total_score desc):")
    for _, r in results.sort_values("total_score", ascending=False).iterrows():
        print(
            f"    {r['symbol']:<8} | {r['final_category']:<35} | "
            f"score={r['total_score']:.1f}"
        )

    print("\n  METHODOLOGICAL WARNINGS:")
    for w in [
        "[W1] Only 12 hand-picked funds analysed; results may not generalise.",
        "[W2] ~5-year window (2019-2023); long-run patterns may differ.",
        "[W3] Survivorship/selection effects exist; BH q-values are reported and gate only the confirmed-strong tier.",
        "[W4] Statistical significance != causality.",
        "[W5] Three of the twelve funds (AMECX, PRHSX, PAGRX) are exploratory non-peers for a broad US equity universe.",
        "[W6] Bootstrap inference is sensitive to block length (30 days used; see sensitivity table).",
        "[W7] Four-factor model may omit relevant risk exposures.",
    ]:
        print(f"    {w}")
    print(f"{sep}\n")


if __name__ == "__main__":
    import sys
    _root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    print(f"[evaluator] Project root: {_root.resolve()}")
    _results = run_evaluation(_root)
    print_summary(_results)
    print("[evaluator] Outputs written:")
    print(f"  {(_root / OUTPUT_CSV).resolve()}")
    print(f"  {(_root / OUTPUT_JSON).resolve()}")
