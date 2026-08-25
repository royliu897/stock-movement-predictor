from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_market_data(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Could not find market data at {source}")

    if source.suffix == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    required = {"close", "high", "low", "open", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    if "symbol" not in frame.columns:
        frame["symbol"] = "UNKNOWN"

    return frame.sort_values(["symbol", "timestamp"] if "timestamp" in frame.columns else ["symbol"]).reset_index(drop=True)


def load_feature_matrix(features_path: str | Path, labels_path: str | Path, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    if label_column not in labels.columns:
        raise ValueError(f"Expected label column {label_column!r} in {labels_path}")
    return features, labels[label_column]

