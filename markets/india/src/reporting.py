"""
Results reporting: summary tables (results/tables) and figures (results/figures)
Indian Fund Manager Skill Detector

Everything here is derived from the already-computed files in data/processed/; no statistical
estimation happens in this module.
"""

import json
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORY_COLORS = {
    "Strong Evidence of Skill": "#15803D",
    "Strong Skill Candidate (Preliminary)": "#047857",
    "Moderate Evidence": "#1D4ED8",
    "Positive but Inconclusive": "#0369A1",
    "Weak/Inconclusive": "#B45309",
    "Likely Style/Luck Driven": "#B91C1C",
}


def _read(root: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root, "data", "processed", name), dtype={"symbol": str})


def _short_names(root: str) -> Dict[str, str]:
    path = os.path.join(root, "data", "raw", "screening_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return {str(r["ticker"]): r.get("short_name", str(r["ticker"])) for r in json.load(fh)}


def export_results_tables(root: str) -> Dict[str, str]:
    """Write the leaderboard and alpha-comparison tables to results/tables/."""
    out_dir = os.path.join(root, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    names = _short_names(root)

    final = _read(root, "final_evaluation.csv")
    capm = _read(root, "capm_results.csv")
    factor = _read(root, "factor_model_results.csv")
    boot = _read(root, "bootstrap_results.csv")

    board = final.copy()
    board.insert(1, "fund", board["symbol"].map(names).fillna(board["symbol"]))
    board_cols = ["symbol", "fund", "final_category", "total_score",
                  "capm_alpha_annual_pct", "capm_alpha_pvalue", "bootstrap_p_value", "bootstrap_p_value_bh",
                  "factor_alpha_annual_pct", "factor_alpha_pvalue", "alpha_reduction_pts_annual",
                  "positive_alpha_persistence_rate", "tenure_overlap_pct", "attribution_note"]
    board = board[[c for c in board_cols if c in board.columns]].sort_values("total_score", ascending=False)
    p_board = os.path.join(out_dir, "skill_leaderboard.csv")
    board.to_csv(p_board, index=False)

    comp = capm[["symbol", "alpha_annual_pct", "alpha_pvalue", "alpha_pvalue_hac", "beta", "r_squared", "durbin_watson"]].rename(
        columns={"alpha_annual_pct": "capm_alpha_annual_pct", "alpha_pvalue": "capm_p_ols", "alpha_pvalue_hac": "capm_p_hac",
                 "beta": "capm_beta", "r_squared": "capm_r2", "durbin_watson": "capm_dw"})
    comp = comp.merge(
        factor[["symbol", "alpha_annual_pct", "alpha_pvalue", "alpha_pvalue_hac", "beta_market", "beta_smb", "beta_hml",
                "beta_momentum", "r_squared"]].rename(columns={
                    "alpha_annual_pct": "factor_alpha_annual_pct", "alpha_pvalue": "factor_p_ols",
                    "alpha_pvalue_hac": "factor_p_hac", "r_squared": "factor_r2"}),
        on="symbol")
    comp = comp.merge(boot[["symbol", "p_value", "p_value_bh", "ci_annual_lower_pct", "ci_annual_upper_pct"]].rename(
        columns={"p_value": "bootstrap_p_two_sided", "p_value_bh": "bootstrap_p_bh"}), on="symbol")
    comp.insert(1, "fund", comp["symbol"].map(names).fillna(comp["symbol"]))
    p_comp = os.path.join(out_dir, "alpha_comparison.csv")
    comp.to_csv(p_comp, index=False)

    return {"leaderboard": p_board, "alpha_comparison": p_comp}


def export_figures(root: str) -> Dict[str, str]:
    """Write static PNG figures to results/figures/."""
    out_dir = os.path.join(root, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    names = _short_names(root)
    paths: Dict[str, str] = {}

    final = _read(root, "final_evaluation.csv")
    capm = _read(root, "capm_results.csv")
    factor = _read(root, "factor_model_results.csv")
    periods = _read(root, "persistence_period_results.csv")
    label = lambda s: names.get(s, s)  # noqa: E731

    # --- Figure 1: evidence index, coloured by category -------------------------------------
    f = final.sort_values("total_score")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([label(s) for s in f["symbol"]], f["total_score"],
            color=[CATEGORY_COLORS.get(c, "#6b7280") for c in f["final_category"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Evidence index (0-100)")
    ax.set_title("Skill-evidence index by fund (v%s rules)" % final["evaluation_version"].iloc[0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, list(CATEGORY_COLORS.keys()), loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    paths["evidence_index"] = os.path.join(out_dir, "evidence_index.png")
    fig.savefig(paths["evidence_index"], dpi=150)
    plt.close(fig)

    # --- Figure 2: CAPM vs 4-factor alpha with HAC 95% intervals ----------------------------
    m = capm.merge(factor, on="symbol", suffixes=("_c", "_f"))
    m = m.sort_values("alpha_annual_pct_f")
    y = np.arange(len(m))

    def ann(a):
        return ((1.0 + a) ** 252 - 1.0) * 100.0

    c_lo = ann(m["alpha_daily_c"] - 1.96 * m["alpha_stderr_hac"]); c_hi = ann(m["alpha_daily_c"] + 1.96 * m["alpha_stderr_hac"])
    f_lo = ann(m["alpha_daily_f"] - 1.96 * m["alpha_se_hac"]);     f_hi = ann(m["alpha_daily_f"] + 1.96 * m["alpha_se_hac"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(m["alpha_annual_pct_c"], y + 0.15, xerr=[m["alpha_annual_pct_c"] - c_lo, c_hi - m["alpha_annual_pct_c"]],
                fmt="o", color="#1D4ED8", capsize=3, label="CAPM alpha (HAC 95% CI)")
    ax.errorbar(m["alpha_annual_pct_f"], y - 0.15, xerr=[m["alpha_annual_pct_f"] - f_lo, f_hi - m["alpha_annual_pct_f"]],
                fmt="s", color="#B45309", capsize=3, label="4-factor alpha (HAC 95% CI)")
    ax.axvline(0, color="#374151", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(s) for s in m["symbol"]])
    ax.set_xlabel("Annualised alpha (%)")
    ax.set_title("Alpha before and after factor adjustment")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths["alpha_comparison"] = os.path.join(out_dir, "alpha_comparison.png")
    fig.savefig(paths["alpha_comparison"], dpi=150)
    plt.close(fig)

    # --- Figure 3: subperiod alpha heat-map -------------------------------------------------
    pivot = periods.pivot(index="symbol", columns="period_id", values="alpha_annual_pct")
    pivot.index = [label(s) for s in pivot.index]
    vmax = float(np.nanmax(np.abs(pivot.values)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Annualised CAPM alpha (%)")
    ax.set_title("Subperiod alpha persistence")
    fig.tight_layout()
    paths["subperiod_alpha"] = os.path.join(out_dir, "subperiod_alpha.png")
    fig.savefig(paths["subperiod_alpha"], dpi=150)
    plt.close(fig)

    # --- Figure 4: block-length sensitivity (if available) ----------------------------------
    sens_path = os.path.join(root, "data", "processed", "bootstrap_block_sensitivity.csv")
    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path, dtype={"symbol": str})
        fig, ax = plt.subplots(figsize=(8, 5))
        for sym, g in sens.groupby("symbol"):
            ax.plot(g["block_length"], g["p_value_two_sided"], marker="o", ms=3, lw=1, label=label(sym))
        ax.axhline(0.05, color="#B91C1C", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Bootstrap block length b (b=1: i.i.d.)"); ax.set_ylabel("Two-sided bootstrap p-value")
        ax.set_title("Sensitivity of bootstrap significance to block length")
        ax.legend(fontsize=6, ncol=2, frameon=False)
        fig.tight_layout()
        paths["block_sensitivity"] = os.path.join(out_dir, "block_length_sensitivity.png")
        fig.savefig(paths["block_sensitivity"], dpi=150)
        plt.close(fig)

    return paths


README_RESULTS_MARKERS = ("<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->")
README_SENSITIVITY_MARKERS = ("<!-- SENSITIVITY_TABLE_START -->", "<!-- SENSITIVITY_TABLE_END -->")
CATEGORY_ORDER = ["Strong Evidence of Skill", "Strong Skill Candidate (Preliminary)", "Moderate Evidence", "Positive but Inconclusive", "Weak/Inconclusive", "Likely Style/Luck Driven"]


def _replace_block(text: str, markers, body: str):
    start, end = markers
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda _m: f"{start}\n{body}\n{end}", text), True


def build_readme_results_table(root: str) -> str:
    """Markdown results table generated from final_evaluation.csv (single source of truth)."""
    names = _short_names(root)
    final = _read(root, "final_evaluation.csv").sort_values("total_score", ascending=False)
    lines = [
        "| Scheme Code | Fund | Named Manager | Classification | Score (0-100) | CAPM alpha (HAC p) | 4-factor alpha (HAC p) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in final.iterrows():
        manager = str(r.get("lead_manager", "") or "")
        if pd.notna(r.get("tenure_overlap_pct", np.nan)) and r["tenure_overlap_pct"] < 90:
            manager += " †"
        lines.append(
            f"| `{r['symbol']}` | {names.get(r['symbol'], r['symbol'])} | {manager} | **{r['final_category']}** | "
            f"**{r['total_score']:.1f}** | {r['capm_alpha_annual_pct']:+.2f}% (p={r['capm_alpha_pvalue']:.3f}) | "
            f"{r['factor_alpha_annual_pct']:+.2f}% (p={r['factor_alpha_pvalue']:.3f}) |")
    lines.append("")
    lines.append("† the named manager was in charge of the scheme for less than 90% of the sample window "
                 "(see `attribution_note` in `final_evaluation.csv`).")
    return "\n".join(lines)


def build_readme_sensitivity_table(root: str) -> Optional[str]:
    """Category counts under each market proxy, from benchmark_sensitivity.csv."""
    path = os.path.join(root, "data", "processed", "benchmark_sensitivity.csv")
    if not os.path.exists(path):
        return None
    sens = pd.read_csv(path, dtype={"symbol": str})
    counts = sens.groupby(["market_proxy", "final_category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CATEGORY_ORDER, fill_value=0)
    header = "| Market proxy | " + " | ".join(CATEGORY_ORDER) + " |"
    lines = [header, "|---|" + "---|" * len(CATEGORY_ORDER)]
    for proxy, row in counts.iterrows():
        lines.append(f"| `{proxy}` | " + " | ".join(str(int(row[c])) for c in CATEGORY_ORDER) + " |")
    return "\n".join(lines)


def update_readme(root: str) -> bool:
    """Rewrite the generated blocks of README.md (between the HTML markers). No-op if README is absent."""
    readme = os.path.join(root, "README.md")
    if not os.path.exists(readme):
        return False
    with open(readme, "r", encoding="utf-8") as fh:
        text = fh.read()
    text, ok1 = _replace_block(text, README_RESULTS_MARKERS, build_readme_results_table(root))
    sens = build_readme_sensitivity_table(root)
    ok2 = False
    if sens is not None:
        text, ok2 = _replace_block(text, README_SENSITIVITY_MARKERS, sens)
    with open(readme, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return ok1 or ok2


def export_all(root: str) -> Dict[str, str]:
    paths = {}
    paths.update(export_results_tables(root))
    paths.update(export_figures(root))
    if update_readme(root):
        paths["readme"] = os.path.join(root, "README.md")
    for key, p in paths.items():
        print(f"[REPORTING] {key}: {p}")
    return paths
