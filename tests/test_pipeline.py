from pathlib import Path

from stock_movement_predictor.backtesting import run_backtest
from stock_movement_predictor.data import load_market_data
from stock_movement_predictor.feature_engineering import FEATURE_COLUMNS, build_features
from stock_movement_predictor.healthcare_benchmark import (
    build_date_blocked_splits,
    chronological_holdout_split,
    load_preserved_healthcare_frame,
    recovered_healthcare_notebook_config,
)
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
    # The sample-data test is only meant to validate the backtest harness, not full XGBoost training.
    model_suite = [spec for spec in build_model_suite() if spec.name != "xgboost_final"]
    results = run_backtest(features, model_suite, splits=3)
    assert len(results) == 2
    for result in results:
        assert 0.0 <= result.mean_accuracy <= 1.0


def test_preserved_healthcare_frame_loads():
    data_path = Path(__file__).resolve().parents[1] / "data" / "healthcare_market_data.parquet"
    frame = load_preserved_healthcare_frame(data_path)
    assert not frame.empty
    assert "target" in frame.columns
    assert "symbol" in frame.columns


def test_healthcare_random_holdout_summary_shows_lift():
    summary_path = Path(__file__).resolve().parents[1] / "results" / "healthcare_holdout_summary.txt"
    summary = summary_path.read_text(encoding="utf-8")

    assert "rows=344698" in summary
    assert "feature_count=13" in summary
    assert "dummy_most_frequent: accuracy=0.6823" in summary
    assert "random_forest_100: accuracy=0.8710" in summary


def test_date_blocked_split_is_chronological():
    data_path = Path(__file__).resolve().parents[1] / "data" / "healthcare_market_data.parquet"
    frame = load_preserved_healthcare_frame(data_path)
    train_frame, test_frame = chronological_holdout_split(frame)

    assert train_frame["timestamp"].max() < test_frame["timestamp"].min()
    folds = build_date_blocked_splits(train_frame, n_splits=3)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        train_max = train_frame.iloc[train_idx]["timestamp"].max()
        test_min = train_frame.iloc[test_idx]["timestamp"].min()
        assert train_max < test_min


def test_healthcare_time_split_summary_is_chronological():
    summary_path = Path(__file__).resolve().parents[1] / "results" / "healthcare_time_split_summary.txt"
    summary = summary_path.read_text(encoding="utf-8")

    assert "train_end=2024-01-22" in summary
    assert "test_start=2024-01-23" in summary
    assert "dummy_most_frequent: accuracy=0.6867" in summary
    assert "random_forest_300_depth10: accuracy=0.6813" in summary


def test_saved_model_summary_shows_artifact_lift():
    summary_path = Path(__file__).resolve().parents[1] / "results" / "healthcare_saved_model_comparison.txt"
    summary = summary_path.read_text(encoding="utf-8")

    assert "best_model.pkl: accuracy=0.8459" in summary
    assert "saved_model_positive_rate=0.3133" in summary


def test_feature_analysis_summary_lists_preserved_schema():
    summary_path = Path(__file__).resolve().parents[1] / "results" / "healthcare_feature_selection_summary.txt"
    summary = summary_path.read_text(encoding="utf-8")

    assert "Preserved healthcare schema includes 13 engineered features." in summary
    assert "MACD_12_26_9: 0.104088" in summary
    assert "open: 0.058657" in summary


def test_recovered_notebook_config_matches_surviving_rf_search():
    config = recovered_healthcare_notebook_config()

    assert config.dataset_names == (
        "healthcareSmallcap.parquet",
        "healthcareMidcap.parquet",
        "healthcareLargecap.parquet",
    )
    assert config.horizon == 7
    assert config.random_search_n_iter == 10
    assert config.random_search_cv == 5
    assert config.grid_search_cv == 5
    assert config.grid_search_grid["n_estimators"] == [50, 100, 200]
    assert config.grid_search_grid["max_depth"] == [None, 10, 20]
