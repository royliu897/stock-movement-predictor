from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import run_healthcare_holdout_benchmark


def main() -> None:
    repo_root = REPO_ROOT
    dataset_path = repo_root / "data" / "healthcare_direction_holdout.parquet"
    results, metadata = run_healthcare_holdout_benchmark(dataset_path)

    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "healthcare_holdout_summary.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Healthcare direction benchmark\n")
        handle.write(f"rows={int(metadata['rows'])}\n")
        handle.write(f"feature_count={int(metadata['feature_count'])}\n")
        handle.write(f"positive_rate={metadata['positive_rate']:.4f}\n")
        handle.write(f"test_size={metadata['test_size']:.2f}\n")
        handle.write(f"random_state={int(metadata['random_state'])}\n")
        for result in results:
            handle.write(f"{result.model_name}: accuracy={result.accuracy:.4f}\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
