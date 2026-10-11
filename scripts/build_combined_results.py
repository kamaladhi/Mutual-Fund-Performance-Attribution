"""Regenerate data/combined_results.csv from the two markets' saved final_evaluation.csv files.

    python scripts/build_combined_results.py            # write the file
    python scripts/build_combined_results.py --check    # exit 1 if the committed file is out of date

The file is a convenience export: the dashboard reads the market files directly. Nothing is re-estimated.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import combined_data as cd  # noqa: E402


def main(argv: list[str]) -> int:
    target = ROOT / "data" / "combined_results.csv"
    if "--check" in argv:
        fresh_df, _, issues = cd.all_results()
        if issues:
            print("Data issues:", *issues, sep="\n  ")
            return 1
        if not target.exists():
            print(f"{target} is missing; run without --check to create it.")
            return 1
        stored = pd.read_csv(target, dtype={"symbol": str})
        fresh = pd.read_csv(io.StringIO(fresh_df.to_csv(index=False)), dtype={"symbol": str})
        same = stored.shape == fresh.shape and list(stored.columns) == list(fresh.columns) and stored.equals(fresh)
        if not same:
            numeric = stored.select_dtypes("number").columns
            same = (stored.shape == fresh.shape and list(stored.columns) == list(fresh.columns)
                    and stored.drop(columns=numeric).fillna("").equals(fresh.drop(columns=numeric).fillna(""))
                    and (stored[numeric].fillna(-9e99) - fresh[numeric].fillna(-9e99)).abs().max().max() < 1e-12)
        print("combined_results.csv is up to date." if same else "combined_results.csv is OUT OF DATE.")
        return 0 if same else 1

    out = cd.export_combined_csv(target)
    df = pd.read_csv(out, dtype={"symbol": str})
    print(f"Wrote {out} ({len(df)} funds).")
    print(cd.category_counts(df).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
