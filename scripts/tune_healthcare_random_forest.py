from __future__ import annotations

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import run_healthcare_tuning_pipeline


def main() -> None:
    dataset_path = REPO_ROOT / "data" / "healthcare_market_data.parquet"
    # This script is the public version of the original random-search then grid-search loop.
    result = run_healthcare_tuning_pipeline(dataset_path)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "healthcare_tuning_summary.txt"

    payload = {
        "selected_features": result.selected_features,
        "random_search_best_params": result.random_search_best_params,
        "random_search_best_score": result.random_search_best_score,
        "grid_search_best_params": result.grid_search_best_params,
        "grid_search_best_score": result.grid_search_best_score,
        "final_test_accuracy": result.final_test_accuracy,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
