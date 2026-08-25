from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stock_movement_predictor.healthcare_benchmark import analyze_feature_selection


def main() -> None:
    dataset_path = REPO_ROOT / "data" / "healthcare_market_data.parquet"
    correlations, importances = analyze_feature_selection(dataset_path)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    corr_path = results_dir / "healthcare_feature_correlations.png"
    imp_path = results_dir / "healthcare_feature_importances.png"
    summary_path = results_dir / "healthcare_feature_selection_summary.txt"

    # Correlation comes first because it shows which indicators are telling roughly the same story.
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlations.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(correlations.index)))
    ax.set_yticklabels(correlations.index)
    ax.set_title("Healthcare Feature Correlation Matrix (Spearman)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(corr_path, dpi=150)
    plt.close(fig)

    # Importance comes next because it shows which engineered columns the forest actually uses.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(importances.index, importances.values, color="#3d6d80")
    ax.set_ylabel("Feature importance")
    ax.set_title("Random Forest Feature Importance on Training Window")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(imp_path, dpi=150)
    plt.close(fig)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Feature selection analysis\n")
        handle.write("Preserved healthcare schema includes 13 engineered features.\n")
        handle.write("Correlation matrix and random-forest importances are computed on the training window only.\n")
        handle.write("\nTop importances:\n")
        for feature, value in importances.items():
            handle.write(f"{feature}: {value:.6f}\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
