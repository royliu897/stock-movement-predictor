from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# Preserved feature set from the healthcare direction branch that produced the saved model.
PRESERVED_FEATURE_COLUMNS = [
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


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    accuracy: float


@dataclass(frozen=True)
class TuningResult:
    selected_features: list[str]
    feature_importances: dict[str, float]
    random_search_best_params: dict[str, object]
    random_search_best_score: float
    grid_search_best_params: dict[str, object]
    grid_search_best_score: float
    final_test_accuracy: float


def load_preserved_healthcare_frame(path: str | Path, horizon: int = 7) -> pd.DataFrame:
    """Load the healthcare benchmark frame and rebuild the 7-day direction label."""
    frame = pd.read_parquet(path).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    frame["target"] = (frame.groupby("symbol")["close"].shift(-horizon) > frame["close"]).astype(int)
    return frame.dropna(subset=["target"]).reset_index(drop=True)


def _date_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")


def chronological_holdout_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use an all-past versus all-future split to check whether the signal survives forward in time."""
    dates = _date_series(frame)
    unique_dates = pd.Index(sorted(dates.unique()))
    cutoff = max(1, int(len(unique_dates) * (1.0 - test_fraction)))
    cutoff = min(cutoff, len(unique_dates) - 1)

    train_dates = set(unique_dates[:cutoff])
    test_dates = set(unique_dates[cutoff:])
    train_frame = frame.loc[dates.isin(train_dates)].reset_index(drop=True)
    test_frame = frame.loc[dates.isin(test_dates)].reset_index(drop=True)
    return train_frame, test_frame


def build_date_blocked_splits(frame: pd.DataFrame, n_splits: int = 3) -> list[tuple[list[int], list[int]]]:
    """Build expanding-window folds so tuning can stay chronological instead of shuffling rows."""
    dates = _date_series(frame)
    unique_dates = pd.Index(sorted(dates.unique()))
    folds: list[tuple[list[int], list[int]]] = []

    fold_size = len(unique_dates) // (n_splits + 1)
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2) if fold < n_splits - 1 else len(unique_dates)
        train_dates = set(unique_dates[:train_end])
        test_dates = set(unique_dates[train_end:test_end])
        train_rows = frame.index[dates.isin(train_dates)].tolist()
        test_rows = frame.index[dates.isin(test_dates)].tolist()
        folds.append((train_rows, test_rows))

    return folds


def _split_xy(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    columns = feature_columns or PRESERVED_FEATURE_COLUMNS
    return frame[columns], frame["target"]


def run_healthcare_random_holdout_benchmark(
    dataset_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[BenchmarkResult], dict[str, float]]:
    """Reproduce the fast random holdout path that is closest to the saved-model training loop."""
    frame = load_preserved_healthcare_frame(dataset_path)
    features, labels = _split_xy(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
    )

    models = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        ("random_forest_100", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
    ]
    results: list[BenchmarkResult] = []
    for model_name, estimator in models:
        estimator.fit(X_train, y_train)
        results.append(BenchmarkResult(model_name, accuracy_score(y_test, estimator.predict(X_test))))

    metadata = {
        "rows": float(len(frame)),
        "feature_count": float(len(PRESERVED_FEATURE_COLUMNS)),
        "positive_rate": float(labels.mean()),
        "test_size": test_size,
        "random_state": float(random_state),
    }
    return results, metadata


def run_healthcare_time_split_benchmark(
    dataset_path: str | Path,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[BenchmarkResult], dict[str, float | str]]:
    """Measure how simple fresh baselines behave once the split is made forward-looking."""
    frame = load_preserved_healthcare_frame(dataset_path)
    train_frame, test_frame = chronological_holdout_split(frame, test_fraction=test_fraction)
    X_train, y_train = _split_xy(train_frame)
    X_test, y_test = _split_xy(test_frame)

    models = [
        ("dummy_most_frequent", DummyClassifier(strategy="most_frequent")),
        ("random_forest_100", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
        (
            "random_forest_300_depth10",
            RandomForestClassifier(n_estimators=300, max_depth=10, random_state=random_state, n_jobs=-1),
        ),
    ]
    results: list[BenchmarkResult] = []
    for model_name, estimator in models:
        estimator.fit(X_train, y_train)
        results.append(BenchmarkResult(model_name, accuracy_score(y_test, estimator.predict(X_test))))

    metadata = {
        "rows": float(len(frame)),
        "feature_count": float(len(PRESERVED_FEATURE_COLUMNS)),
        "positive_rate": float(frame["target"].mean()),
        "train_rows": float(len(train_frame)),
        "test_rows": float(len(test_frame)),
        "train_end": str(pd.to_datetime(train_frame["timestamp"], utc=True).max().date()),
        "test_start": str(pd.to_datetime(test_frame["timestamp"], utc=True).min().date()),
    }
    return results, metadata


def evaluate_saved_model_on_time_split(
    dataset_path: str | Path,
    model_path: str | Path,
    test_fraction: float = 0.2,
) -> tuple[BenchmarkResult, dict[str, float | str]]:
    """Score the preserved trained artifact on the same forward split as the fresh baselines."""
    frame = load_preserved_healthcare_frame(dataset_path)
    train_frame, test_frame = chronological_holdout_split(frame, test_fraction=test_fraction)
    X_test, y_test = _split_xy(test_frame)
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    metadata = {
        "rows": float(len(frame)),
        "test_rows": float(len(test_frame)),
        "train_end": str(pd.to_datetime(train_frame["timestamp"], utc=True).max().date()),
        "test_start": str(pd.to_datetime(test_frame["timestamp"], utc=True).min().date()),
        "positive_rate": float(y_test.mean()),
    }
    return BenchmarkResult("best_model.pkl", accuracy_score(y_test, predictions)), metadata


def analyze_feature_selection(
    dataset_path: str | Path,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Expose the two checks that drove feature narrowing: correlation and tree importance."""
    frame = load_preserved_healthcare_frame(dataset_path)
    train_frame, _ = chronological_holdout_split(frame, test_fraction=test_fraction)
    X_train, y_train = _split_xy(train_frame)

    correlations = X_train.corr(method="spearman")
    forest = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = pd.Series(forest.feature_importances_, index=PRESERVED_FEATURE_COLUMNS).sort_values(ascending=False)
    return correlations, importances


def run_healthcare_tuning_pipeline(
    dataset_path: str | Path,
    test_fraction: float = 0.2,
    n_splits: int = 3,
    top_k: int = 8,
    random_state: int = 42,
) -> TuningResult:
    """Mirror the original workflow: rank features, trim the set, then tune the forest on blocked folds."""
    frame = load_preserved_healthcare_frame(dataset_path)
    train_frame, test_frame = chronological_holdout_split(frame, test_fraction=test_fraction)
    X_train, y_train = _split_xy(train_frame)
    X_test, y_test = _split_xy(test_frame)

    # Start with a broad forest so the feature subset is chosen from model behavior, not guesswork.
    selector = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=PRESERVED_FEATURE_COLUMNS).sort_values(ascending=False)
    selected_features = importances.head(top_k).index.tolist()

    # Keep tuning chronological; random CV would inflate scores on nearby rows.
    cv_splits = build_date_blocked_splits(train_frame, n_splits=n_splits)
    base_estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions={
            "n_estimators": [100, 150, 200],
            "max_depth": [6, 8, 10, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2", None],
        },
        n_iter=5,
        scoring="accuracy",
        cv=cv_splits,
        random_state=random_state,
        n_jobs=1,
    )
    random_search.fit(X_train[selected_features], y_train)
    random_best = random_search.best_params_

    max_depth_value = random_best["max_depth"]
    if max_depth_value is None:
        max_depth_grid: list[int | None] = [None]
    else:
        max_depth_grid = sorted({max(4, int(max_depth_value) - 2), int(max_depth_value), int(max_depth_value) + 2})

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_grid={
            "n_estimators": sorted({max(50, int(random_best["n_estimators"]) - 50), int(random_best["n_estimators"]), int(random_best["n_estimators"]) + 50}),
            "max_depth": max_depth_grid,
            "min_samples_split": sorted({max(2, int(random_best["min_samples_split"]) - 2), int(random_best["min_samples_split"]), int(random_best["min_samples_split"]) + 2}),
            "min_samples_leaf": sorted({max(1, int(random_best["min_samples_leaf"]) - 1), int(random_best["min_samples_leaf"]), int(random_best["min_samples_leaf"]) + 1}),
            "max_features": [random_best["max_features"]],
        },
        scoring="accuracy",
        cv=cv_splits,
        n_jobs=1,
    )
    grid_search.fit(X_train[selected_features], y_train)

    # Final score is held out from the tuning folds so the summary reads like a real pipeline checkpoint.
    final_model = RandomForestClassifier(**grid_search.best_params_, random_state=random_state, n_jobs=-1)
    final_model.fit(X_train[selected_features], y_train)
    final_accuracy = accuracy_score(y_test, final_model.predict(X_test[selected_features]))

    return TuningResult(
        selected_features=selected_features,
        feature_importances={k: float(v) for k, v in importances.items()},
        random_search_best_params=random_search.best_params_,
        random_search_best_score=float(random_search.best_score_),
        grid_search_best_params=grid_search.best_params_,
        grid_search_best_score=float(grid_search.best_score_),
        final_test_accuracy=float(final_accuracy),
    )
