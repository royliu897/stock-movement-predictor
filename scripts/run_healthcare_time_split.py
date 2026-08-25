from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import run_healthcare_time_split_benchmark


def main() -> None:
    dataset_path = REPO_ROOT / "data" / "healthcare_market_data.parquet"
    # This benchmark answers the harder question: what survives when test rows come strictly later in time?
    results, metadata = run_healthcare_time_split_benchmark(dataset_path)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "healthcare_time_split_summary.txt"

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Healthcare time-split benchmark\n")
        handle.write(f"rows={int(metadata['rows'])}\n")
        handle.write(f"feature_count={int(metadata['feature_count'])}\n")
        handle.write(f"positive_rate={metadata['positive_rate']:.4f}\n")
        handle.write(f"train_rows={int(metadata['train_rows'])}\n")
        handle.write(f"test_rows={int(metadata['test_rows'])}\n")
        handle.write(f"train_end={metadata['train_end']}\n")
        handle.write(f"test_start={metadata['test_start']}\n")
        for result in results:
            handle.write(f"{result.model_name}: accuracy={result.accuracy:.4f}\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
