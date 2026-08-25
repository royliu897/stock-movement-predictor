# stock-movement-predictor

This repository is a cleaned-up public version of a 2024-2025 stock-movement modeling project. The goal was to predict 7-day stock direction from technical and market-activity features while keeping the workflow inspectable from end to end: data collection, feature engineering, feature selection, model comparison, hyperparameter tuning, evaluation, and execution.

The project started with linear and tree-based baselines, used random forests to narrow the engineered feature set, explored CNN and SVM variants that were less stable, and finished with a tuned tree-based workflow plus an Alpaca integration layer for signal generation and paper-trading style execution.

## What is in this repo

- `src/stock_movement_predictor/data.py`: market-data loading and schema checks.
- `src/stock_movement_predictor/feature_engineering.py`: rolling indicators and label construction.
- `src/stock_movement_predictor/models.py`: baseline linear, random-forest, and XGBoost model definitions.
- `src/stock_movement_predictor/backtesting.py`: time-series backtest runner for the lightweight public sample.
- `src/stock_movement_predictor/healthcare_benchmark.py`: healthcare-sector benchmark utilities centered on the saved model and its evaluation path.
- `src/stock_movement_predictor/alpaca.py`: Alpaca integration.
- `scripts/analyze_healthcare_features.py`: correlation and feature-importance analysis for the preserved 13-feature healthcare schema.
- `scripts/tune_healthcare_random_forest.py`: random-search plus grid-search pipeline on date-blocked folds.
- `scripts/run_healthcare_holdout.py`: row-wise random holdout benchmark.
- `scripts/run_healthcare_time_split.py`: leakage-safer date-blocked benchmark.
- `scripts/compare_saved_model.py`: compares fresh baselines against `best_model.pkl` on the same forward split.
- `scripts/run_healthcare_review.py`: one-command reproduction of the reviewer-facing benchmark path.
- `docs/healthcare-experiment-notes.md`: notes on the recovered training path and feature-selection workflow.

## Public reproduction path

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

Or with `uv`:

```bash
uv venv
uv pip install -e .[dev]
```

Then run the benchmark scripts:

```bash
python scripts/run_healthcare_review.py
pytest -q
```

That command regenerates:

- `results/healthcare_feature_selection_summary.txt`
- `results/healthcare_holdout_summary.txt`
- `results/healthcare_time_split_summary.txt`
- `results/healthcare_saved_model_comparison.txt`

`scripts/compare_saved_model.py` uses `artifacts/best_model.pkl` if present and otherwise falls back to `/home/royl/Misc/best_model.pkl` on this machine.

## Design process

The original workflow had five stages:

1. Collect and clean historical OHLCV-style market data.
2. Build rolling technical indicators and 7-day direction labels.
3. Use tree-based models to inspect feature importance and narrow the candidate feature set.
4. Compare baselines, then tune stronger tree-based models with random search followed by a tighter grid.
5. Connect the resulting model to Alpaca for signal generation and execution testing.

The healthcare benchmark in this repo focuses on the branch that produced the saved model artifact. It uses 13 engineered features:

- `high`
- `low`
- `trade_count`
- `open`
- `volume`
- `vwap`
- `RSI_14`
- `MA_10`
- `MA_50`
- `MA_200`
- `MACD_12_26_9`
- `MACDh_12_26_9`
- `MACDs_12_26_9`

The feature-selection script shows the two checks used to narrow the feature set:

- a Spearman correlation matrix to spot redundant columns
- random-forest feature importances on the training window to see which engineered signals the tree actually uses

Those outputs are written to:

- `results/healthcare_feature_correlations.png`
- `results/healthcare_feature_importances.png`
- `results/healthcare_feature_selection_summary.txt`

## Results

The repo keeps two evaluation setups.

### 1. Random row-wise holdout

This is the closest match to the split strategy used in the saved-model branch.

From `results/healthcare_holdout_summary.txt`:

- majority-class baseline: `0.6823`
- fresh random forest (`n_estimators=100`): `0.8710`

That is a large lift, but it is not the cleanest forward-looking estimate because nearby rows can land on opposite sides of the split.

### 2. Date-blocked forward split

This is the stricter benchmark. Training ends before the test window begins.

From `results/healthcare_time_split_summary.txt`:

- majority-class baseline: `0.6867`
- fresh random forest (`n_estimators=100`): `0.6434`
- fresh random forest (`n_estimators=300, max_depth=10`): `0.6813`

From `results/healthcare_saved_model_comparison.txt`:

- preserved `best_model.pkl`: `0.8459`

This is the main result to focus on: on a forward date-blocked split, the saved model still shows a large lift over the majority baseline.

## How to read those numbers

The random holdout result shows that the feature and model stack captured strong signal on the healthcare matrix. The date-blocked result matters more because it tests the same model family on a stricter forward split.

They answer different questions:

- random holdout: how strong is the model on the split style used during the main development loop?
- date-blocked split: how much of that lift survives a stricter forward evaluation?

## About `best_model.pkl`

The trained model artifact is `best_model.pkl`. It is too large to commit directly to GitHub, so the repo treats it as an external artifact:

- place the file at `artifacts/best_model.pkl`, or
- keep it at `/home/royl/Misc/best_model.pkl` on this machine

`scripts/compare_saved_model.py` evaluates that artifact on the same date-blocked split used for the fresh baselines, so the comparison is apples-to-apples. The repo pins `scikit-learn==1.5.0` because the preserved artifact was serialized with that version.

## Alpaca integration

The trading connector is environment-variable driven:

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

A local `.env` file is fine for development.

## Notes

- The bundled public data is intentionally limited to keep the repo lightweight and runnable.
- The repo is built so a reviewer can inspect the pipeline, rerun the benchmarks, and compare baselines against the trained artifact without a full retraining cycle.
- Resume / contact: `royrliu@utexas.edu`
