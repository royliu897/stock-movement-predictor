from pathlib import Path

from stock_movement_predictor.backtesting import run_backtest
from stock_movement_predictor.data import load_labeled_matrix, load_market_data
from stock_movement_predictor.feature_engineering import FEATURE_COLUMNS, build_features
from stock_movement_predictor.healthcare_benchmark import run_healthcare_holdout_benchmark
from stock_movement_predictor.models import build_model_suite


def test_feature_builder_produces_expected_columns():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_stock_data.csv"
    market_data = load_market_data(data_path)
    features = build_features(market_data, horizon=7)

    for column in FEATURE_COLUMNS + ["target", "future_return"]:
        assert column in features.columns
    assert not features.empty


def test_backtest_runs_on_sample_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_stock_data.csv"
    market_data = load_market_data(data_path)
    features = build_features(market_data, horizon=7)
    results = run_backtest(features, build_model_suite(), splits=3)
    assert len(results) == 3
    for result in results:
        assert 0.0 <= result.mean_accuracy <= 1.0


def test_labeled_holdout_matrix_loads():
    data_path = Path(__file__).resolve().parents[1] / "data" / "healthcare_direction_holdout.parquet"
    features, labels = load_labeled_matrix(data_path, label_column="target")
    assert not features.empty
    assert len(features) == len(labels)
    assert "target" not in features.columns


def test_healthcare_benchmark_shows_lift_over_baseline():
    data_path = Path(__file__).resolve().parents[1] / "data" / "healthcare_direction_holdout.parquet"
    results, metadata = run_healthcare_holdout_benchmark(data_path)
    metrics = {result.model_name: result.accuracy for result in results}

    assert metadata["rows"] > 300_000
    assert 0.65 <= metrics["dummy_most_frequent"] <= 0.70
    assert metrics["random_forest_100"] >= 0.85
    assert metrics["random_forest_100"] > metrics["dummy_most_frequent"] + 0.15
