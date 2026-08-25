from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from stock_movement_predictor.backtesting import run_backtest
from stock_movement_predictor.data import load_market_data
from stock_movement_predictor.feature_engineering import build_features
from stock_movement_predictor.models import build_model_suite


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "sample_stock_data.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    market_data = load_market_data(data_path)
    features = build_features(market_data, horizon=7)
    model_specs = build_model_suite()
    results = run_backtest(features, model_specs, splits=5)

    summary_path = results_dir / "sample_backtest_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(f"{result.model_name}: mean_accuracy={result.mean_accuracy:.4f} folds={result.fold_accuracies}\n")

    plt.figure(figsize=(8, 5))
    plt.bar([result.model_name for result in results], [result.mean_accuracy for result in results], color=["#7a8c5f", "#e59f71", "#375a7f"])
    plt.ylabel("Mean time-series CV accuracy")
    plt.title("Sample backtest model comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "sample_backtest_accuracy.png", dpi=150)


if __name__ == "__main__":
    main()

