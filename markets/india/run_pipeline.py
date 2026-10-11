"""
run_pipeline.py -- one-command, end-to-end reproduction of every analytical output.

    python run_pipeline.py

Steps (each also runnable on its own with ``python -m src.<module>``):
  1. data_loader   -> data/processed/aligned_analysis_dataset.csv (+ data_validation_report.json)
  2. capm          -> capm_results.csv/json (+ capm_sensitivity_nifty500_price.csv)
  3. bootstrap     -> bootstrap_results.csv/json (+ bootstrap_block_sensitivity.csv)
  4. factor_model  -> factor_model_results.csv/json
  5. persistence   -> persistence_period_results / persistence_summary
  6. descriptive   -> descriptive_statistics.csv
  7. evaluator     -> final_evaluation.csv/json
  8. sensitivity   -> benchmark_sensitivity.csv (whole stack re-run under alternative market proxies)
  9. reporting     -> results/tables/*.csv, results/figures/*.png

Raw data under data/raw/ is never re-downloaded if it already exists (idempotent acquisition).
"""

import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import (data_loader, capm, bootstrap, factor_model, persistence, descriptive,  # noqa: E402
                 evaluator, reporting, sensitivity)


def main(root: str = ROOT, n_bootstrap: int = 1000, block_length: int = 30, seed: int = 42) -> pd.DataFrame:
    t0 = time.time()
    processed = os.path.join(root, "data", "processed")

    # 1. data
    summary = data_loader.run_data_pipeline(root)
    aligned = pd.read_csv(os.path.join(processed, "aligned_analysis_dataset.csv"), dtype={"symbol": str})

    # 2. CAPM (primary market proxy) + labelled sensitivity against the NIFTY 500 price index
    capm_df, capm_dict, _ = capm.run_capm_analysis(aligned)
    capm.save_capm_results(capm_df, capm_dict, processed)
    alt = aligned.copy()
    alt["excess_market_return"] = alt["excess_nifty500_price_return"]
    alt_df, _, _ = capm.run_capm_analysis(alt)
    alt_df.insert(1, "market_proxy", "nifty500_price_index (excludes dividends)")
    alt_df.to_csv(os.path.join(processed, "capm_sensitivity_nifty500_price.csv"), index=False)

    # 3. bootstrap
    boot_df, boot_dict, _ = bootstrap.run_bootstrap_analysis(
        aligned, capm_dict, block_length=block_length, n_bootstrap=n_bootstrap, seed=seed)
    bootstrap.save_bootstrap_results(boot_df, boot_dict, processed)
    sens = bootstrap.run_block_length_sensitivity(aligned, n_bootstrap=n_bootstrap, seed=seed)
    sens.to_csv(os.path.join(processed, "bootstrap_block_sensitivity.csv"), index=False)

    # 4. Carhart 4-factor
    fac_df, fac_dict, _ = factor_model.run_four_factor_analysis(aligned, capm_dict)
    factor_model.save_factor_model_results(fac_df, fac_dict, processed)

    # 5. persistence
    period_df, _ = persistence.run_period_capm_analysis(aligned, num_periods=4)
    pers_summary = persistence.calculate_persistence_metrics(period_df, capm_df)
    persistence.save_persistence_results(period_df, pers_summary, processed)

    # 6. descriptive statistics
    descriptive.save_descriptive_statistics(
        descriptive.run_descriptive_statistics(aligned, raw_dir=os.path.join(root, "data", "raw")), processed)

    # 7. integrated evaluation
    results = evaluator.run_evaluation(root)
    evaluator.print_summary(results)

    # 8. benchmark-proxy sensitivity (re-runs the whole stack under alternative market proxies)
    sensitivity.save_benchmark_sensitivity(
        sensitivity.run_benchmark_sensitivity(aligned, n_bootstrap=n_bootstrap, block_length=block_length, seed=seed),
        processed)

    # 9. tables + figures
    reporting.export_all(root)

    # 10. run manifest (read by the dashboard sidebar; documents exactly how the outputs were produced)
    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "evaluation_version": evaluator.EVALUATION_VERSION,
        "market_proxy": summary["market_proxy"],
        "sample_start": summary["date_range"][0],
        "sample_end": summary["date_range"][1],
        "distinct_dates": summary["unique_dates"],
        "observations_per_fund": [summary["obs_per_fund_min"], summary["obs_per_fund_max"]],
        "asynchronous_observations_dropped": summary["n_dropped_async"],
        "bootstrap": {"method": "circular block, paired resampling, null-centred", "block_length": block_length,
                      "n_bootstrap": n_bootstrap, "seed": seed},
        "hac": "Newey-West, Bartlett kernel, lags = floor(4*(n/100)^(2/9)), small-sample n/(n-k)",
        "assumed_nifty500_dividend_yield_for_sensitivity": sensitivity.ASSUMED_NIFTY500_DIVIDEND_YIELD,
    }
    with open(os.path.join(processed, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[run_pipeline] done in {time.time() - t0:.1f}s | data summary: {summary}")
    return results


if __name__ == "__main__":
    main()
