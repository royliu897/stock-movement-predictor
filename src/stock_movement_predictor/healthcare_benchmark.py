from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .data import load_labeled_matrix


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    accuracy: float


def run_healthcare_holdout_benchmark(
    dataset_path: str | Path,
    label_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[BenchmarkResult], dict[str, float]]:
    features, labels = load_labeled_matrix(dataset_path, label_column=label_column)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    models = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        (
            "random_forest_100",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]

    results: list[BenchmarkResult] = []
    for model_name, estimator in models:
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        results.append(BenchmarkResult(model_name=model_name, accuracy=accuracy_score(y_test, predictions)))

    metadata = {
        "rows": float(len(features)),
        "feature_count": float(features.shape[1]),
        "positive_rate": float(labels.mean()),
        "test_size": test_size,
        "random_state": float(random_state),
    }
    return results, metadata
