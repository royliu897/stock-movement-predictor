from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier


class LinearRegressionDirectionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self._regressor = LinearRegression()

    def fit(self, X, y):
        signed = np.where(np.asarray(y) > 0, 1.0, -1.0)
        self._regressor.fit(X, signed)
        return self

    def predict(self, X):
        scores = self._regressor.predict(X)
        return (scores >= 0.0).astype(int)

    def predict_proba(self, X):
        scores = self._regressor.predict(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs, probs])


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator


def build_model_suite() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="linear_regression_baseline",
            estimator=LinearRegressionDirectionClassifier(),
        ),
        ModelSpec(
            name="random_forest_baseline",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="xgboost_final",
            estimator=XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
            ),
        ),
    ]

