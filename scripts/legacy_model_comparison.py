from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


FEATURES_13 = [
    "high",
    "low",
    "trade_count",
    "open",
    "volume",
    "vwap",
    "RSI_14",
    "MA_10",
    "MA_50",
    "MA_200",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
]


def add_future_targets(frame: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    df = frame.sort_values(["symbol", "timestamp"]).copy()
    df["future_close"] = df.groupby("symbol")["close"].shift(-horizon_days)
    df["target01"] = (df["future_close"] > df["close"]).astype(int)
    return df.dropna(subset=["future_close"] + FEATURES_13)


def evaluate_classifier(model, dataset_name: str, frame: pd.DataFrame) -> str:
    X = frame[FEATURES_13]
    y = frame["target01"]
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    positive_rate = float(np.mean(predictions))
    return f"{dataset_name:28s} classifier_accuracy={accuracy:.6f} predicted_positive_rate={positive_rate:.6f} rows={len(frame)}"


def evaluate_regressor(model, dataset_name: str, frame: pd.DataFrame) -> str:
    X = frame[FEATURES_13]
    future_close = frame["future_close"]
    close_now = frame["close"].to_numpy()
    predicted_future_close = model.predict(X)
    predicted_direction = (predicted_future_close > close_now).astype(int)
    direction_accuracy = accuracy_score(frame["target01"], predicted_direction)
    mae = mean_absolute_error(future_close, predicted_future_close)
    r2 = r2_score(future_close, predicted_future_close)
    return (
        f"{dataset_name:28s} directional_accuracy={direction_accuracy:.6f} "
        f"mae={mae:.6f} r2={r2:.6f} rows={len(frame)}"
    )


def main() -> None:
    misc = Path("/home/royl/Misc")
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    healthcare_sets = {
        "healthcare_smallcap": add_future_targets(pd.read_parquet(misc / "healthcareSmallcap.parquet")),
        "healthcare_midcap": add_future_targets(pd.read_parquet(misc / "healthcareMidcap.parquet")),
        "healthcare_largecap": add_future_targets(pd.read_parquet(misc / "healthcareLargecap.parquet")),
    }
    tech_sets = {
        "tech_smallcap": add_future_targets(pd.read_parquet(misc / "techSmallcap.parquet")),
        "tech_midcap": add_future_targets(pd.read_parquet(misc / "techMidcap.parquet")),
        "tech_largecap": add_future_targets(pd.read_parquet(misc / "techLargecap.parquet")),
    }

    classifier = joblib.load(misc / "best_model.pkl")
    reg_small = joblib.load(misc / "small_cap_model.pkl")
    reg_mid = joblib.load(misc / "mid_cap_model.pkl")
    reg_large = joblib.load(misc / "large_cap_model.pkl")

    lines: list[str] = []
    lines.append("Legacy classifier path")
    for name, frame in healthcare_sets.items():
        lines.append(evaluate_classifier(classifier, name, frame))
    healthcare_combo = pd.concat(list(healthcare_sets.values()), ignore_index=True)
    tech_combo = pd.concat(list(tech_sets.values()), ignore_index=True)
    lines.append(evaluate_classifier(classifier, "healthcare_combined", healthcare_combo))
    lines.append(evaluate_classifier(classifier, "tech_combined", tech_combo))
    lines.append("")
    lines.append("Legacy regressor path")
    lines.append(evaluate_regressor(reg_small, "healthcare_smallcap", healthcare_sets["healthcare_smallcap"]))
    lines.append(evaluate_regressor(reg_mid, "healthcare_midcap", healthcare_sets["healthcare_midcap"]))
    lines.append(evaluate_regressor(reg_large, "healthcare_largecap", healthcare_sets["healthcare_largecap"]))

    output = "\n".join(lines)
    output_path = results_dir / "legacy_model_comparison.txt"
    output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()

