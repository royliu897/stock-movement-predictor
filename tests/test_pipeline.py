from pathlib import Path

from stock_movement_predictor.backtesting import run_backtest
from stock_movement_predictor.data import load_market_data
from stock_movement_predictor.feature_engineering import FEATURE_COLUMNS, build_features
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

