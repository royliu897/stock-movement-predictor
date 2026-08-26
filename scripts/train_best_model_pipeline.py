from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import run_best_model_training_pipeline


def configured_data_dir() -> Path:
    configured = os.environ.get("HEALTHCARE_DATA_DIR")
    if configured:
        return Path(configured)
    raise SystemExit("Set HEALTHCARE_DATA_DIR to the directory containing healthcareSmallcap.parquet, healthcareMidcap.parquet, and healthcareLargecap.parquet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display or run the full training pipeline used for the tuned healthcare random forest."
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing healthcareSmallcap/Midcap/Largecap parquet files")
    parser.add_argument("--run-training", action="store_true", help="Execute the full feature-selection and 5-fold search workflow")
    parser.add_argument("--n-jobs", type=int, default=1, help="Job count for sklearn estimators and searches")
    args = parser.parse_args()
    data_dir = args.data_dir or configured_data_dir()

    payload = {
        "datasets": [
            "healthcareSmallcap.parquet",
            "healthcareMidcap.parquet",
            "healthcareLargecap.parquet",
        ],
        "data_dir": str(data_dir),
        "target_definition": "close[t+7] > close[t]",
        "outer_split": "chronological 80/20 holdout",
        "feature_selection": "RandomForest feature importance, top 13 features",
        "baseline_model": {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
        },
        "search_workflow": {
            "random_search": {
                "n_iter": 10,
                "cv": 5,
                "scoring": "accuracy",
                "space": {
                    "n_estimators": "randint(50, 200)",
                    "max_depth": [None, 10, 20],
                    "min_samples_split": "randint(2, 10)",
                    "min_samples_leaf": "randint(1, 5)",
                },
            },
            "grid_search": {
                "cv": 5,
                "scoring": "accuracy",
                "grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
        },
        "run_training": args.run_training,
    }

    # Default mode is documentation: show the full training path without making reviewers wait on a heavy run.
    if not args.run_training:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    result = run_best_model_training_pipeline(data_dir, n_jobs=args.n_jobs)
    payload["selected_features"] = result.selected_features
    payload["train_rows"] = result.train_rows
    payload["feature_count"] = result.feature_count
    payload["random_search_best_params"] = result.random_search_best_params
    payload["random_search_best_score"] = result.random_search_best_score
    payload["grid_search_best_params"] = result.grid_search_best_params
    payload["grid_search_best_score"] = result.grid_search_best_score

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "best_model_training_pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
