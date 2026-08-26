from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import (
    evaluate_saved_model_on_time_split,
    run_healthcare_time_split_benchmark,
)


def default_model_path() -> Path:
    configured = os.environ.get("MODEL_PATH")
    if configured:
        return Path(configured)
    return REPO_ROOT / "artifacts" / "best_model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-leakage healthcare benchmark on baselines and the trained model artifact.")
    parser.add_argument("--model-path", type=Path, default=default_model_path(), help="Path to best_model.pkl")
    parser.add_argument(
        "--benchmark-model",
        choices=("baseline", "trained", "all"),
        default="all",
        help="Choose whether to run the untuned baselines, the trained artifact, or both.",
    )
    args = parser.parse_args()

    dataset_path = REPO_ROOT / "data" / "healthcare_market_data.parquet"
    # The chronological split is shared so every benchmark mode evaluates on the same forward holdout.
    results, metadata = run_healthcare_time_split_benchmark(dataset_path)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "healthcare_saved_model_comparison.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Healthcare saved-model comparison\n")
        handle.write(f"rows={int(metadata['rows'])}\n")
        handle.write(f"feature_count={int(metadata['feature_count'])}\n")
        handle.write(f"train_rows={int(metadata['train_rows'])}\n")
        handle.write(f"test_rows={int(metadata['test_rows'])}\n")
        handle.write(f"train_end={metadata['train_end']}\n")
        handle.write(f"test_start={metadata['test_start']}\n")
        handle.write(f"benchmark_model={args.benchmark_model}\n")

        if args.benchmark_model in {"baseline", "all"}:
            for result in results:
                handle.write(f"{result.model_name}: accuracy={result.accuracy:.4f}\n")

        if args.benchmark_model in {"trained", "all"}:
            if args.model_path.exists():
                saved_result, saved_metadata = evaluate_saved_model_on_time_split(dataset_path, args.model_path)
                handle.write(f"{saved_result.model_name}: accuracy={saved_result.accuracy:.4f}\n")
                handle.write(f"saved_model_positive_rate={saved_metadata['positive_rate']:.4f}\n")
            else:
                handle.write(f"best_model.pkl: missing ({args.model_path})\n")
                handle.write("Set MODEL_PATH or place the artifact at artifacts/best_model.pkl.\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
