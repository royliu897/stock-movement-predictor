from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import (
    recovered_healthcare_notebook_config,
    run_recovered_healthcare_notebook_search,
)


def configured_data_dir() -> Path:
    configured = os.environ.get("HEALTHCARE_DATA_DIR")
    if configured:
        return Path(configured)
    raise SystemExit("Set HEALTHCARE_DATA_DIR to the directory containing healthcareSmallcap.parquet, healthcareMidcap.parquet, and healthcareLargecap.parquet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover the original notebook-style healthcare RF search path without forcing the full expensive run by default."
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing healthcareSmallcap/Midcap/Largecap parquet files")
    parser.add_argument("--run-search", action="store_true", help="Execute the full RandomizedSearchCV -> GridSearchCV pipeline")
    parser.add_argument("--n-jobs", type=int, default=1, help="Job count for sklearn search objects when --run-search is used")
    args = parser.parse_args()
    data_dir = args.data_dir or configured_data_dir()

    config = recovered_healthcare_notebook_config()
    payload = {
        "recovered_from": "Untitled--1.ipynb",
        "datasets": list(config.dataset_names),
        "data_dir": str(data_dir),
        "target_definition": "close[t+7] > close[t]",
        "drop_columns_before_training": ["close", "timestamp", "symbol", "target"],
        "random_search_n_iter": config.random_search_n_iter,
        "random_search_cv": config.random_search_cv,
        "grid_search_cv": config.grid_search_cv,
        "scoring": config.scoring,
        "random_search_distributions": {
            "n_estimators": "randint(50, 200)",
            "max_depth": [None, 10, 20],
            "min_samples_split": "randint(2, 10)",
            "min_samples_leaf": "randint(1, 5)",
        },
        "grid_search_grid": config.grid_search_grid,
        "run_search": args.run_search,
    }

    # Default mode is documentation, not training. Reviewers can inspect the recovered path without paying the runtime cost.
    if not args.run_search:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    result = run_recovered_healthcare_notebook_search(data_dir, n_jobs=args.n_jobs)
    payload["row_count"] = result.row_count
    payload["feature_count"] = result.feature_count
    payload["random_search_best_params"] = result.random_search_best_params
    payload["random_search_best_score"] = result.random_search_best_score
    payload["grid_search_best_params"] = result.grid_search_best_params
    payload["grid_search_best_score"] = result.grid_search_best_score

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "healthcare_notebook_reconstruction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
