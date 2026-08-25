from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from .feature_engineering import FEATURE_COLUMNS
from .models import ModelSpec


@dataclass(frozen=True)
class BacktestResult:
    model_name: str
    fold_accuracies: list[float]

    @property
    def mean_accuracy(self) -> float:
        return sum(self.fold_accuracies) / len(self.fold_accuracies)


def run_backtest(frame: pd.DataFrame, model_specs: list[ModelSpec], splits: int = 5) -> list[BacktestResult]:
    X = frame[FEATURE_COLUMNS]
    y = frame["target"]
    splitter = TimeSeriesSplit(n_splits=splits)
    results: list[BacktestResult] = []

    for spec in model_specs:
        scores: list[float] = []
        for train_idx, test_idx in splitter.split(X):
            estimator = clone(spec.estimator)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = estimator.predict(X.iloc[test_idx])
            scores.append(float(accuracy_score(y.iloc[test_idx], predictions)))
        results.append(BacktestResult(spec.name, scores))

    return results

