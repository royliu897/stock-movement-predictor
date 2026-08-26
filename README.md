# stock-movement-predictor

This repository is a cleaned-up public version of a 2024-2025 stock-movement modeling project. It focuses on predicting 7-day stock direction from technical and market-activity features, with the code organized around the full workflow from feature engineering and feature selection through tuning, model training, and broker-side signal execution.

The project compares linear and tree-based baselines, narrows the feature set with random forests, explores CNN and SVM variants, and finishes with a tuned tree-based workflow plus an Alpaca integration layer for signal generation and paper-trading style execution rather than live deployment.

## What is in this repo

- `src/stock_movement_predictor/data.py`: market-data loading and schema checks.
- `src/stock_movement_predictor/feature_engineering.py`: rolling indicators and label construction.
- `src/stock_movement_predictor/models.py`: baseline linear, random-forest, and XGBoost model definitions.
- `src/stock_movement_predictor/backtesting.py`: time-series backtest runner for the lightweight public sample.
- `src/stock_movement_predictor/healthcare_benchmark.py`: healthcare-sector benchmark utilities centered on the saved model and its evaluation path.
- `src/stock_movement_predictor/alpaca.py`: Alpaca integration.
- `scripts/analyze_healthcare_features.py`: correlation and feature-importance analysis for the preserved 13-feature healthcare schema.
- `scripts/tune_healthcare_random_forest.py`: random-search plus grid-search pipeline on date-blocked folds.
- `scripts/train_best_model_pipeline.py`: the full 5-fold training and tuning pipeline for the tuned healthcare random forest.
- `scripts/run_healthcare_holdout.py`: row-wise random holdout benchmark.
- `scripts/run_healthcare_time_split.py`: leakage-safer date-blocked benchmark.
- `scripts/compare_saved_model.py`: compares untuned baselines and `best_model.pkl` on the same forward split.
- `scripts/run_healthcare_review.py`: regenerates the comparison, the supporting baselines, and the feature-selection outputs.
- `docs/healthcare-experiment-notes.md`: notes on the recovered training path and feature-selection workflow.

## Project structure

The original workflow had five stages:

1. Collect and clean historical OHLCV-style market data.
2. Build rolling technical indicators and 7-day direction labels.
3. Use tree-based models to inspect feature importance and narrow the candidate feature set.
4. Compare baselines, then tune stronger tree-based models with random search followed by a tighter grid.
5. Connect the resulting model to Alpaca for signal generation and execution testing.

The training code is organized around the same pattern:

- feature ranking and narrowing
- untuned baseline training
- 5-fold hyperparameter search
- tighter grid refinement
- final model fit and artifact save

`scripts/train_best_model_pipeline.py` is the end-to-end training path for that workflow. The raw healthcare parquet inputs it expects are not bundled in the repo because they are too large for a lightweight public clone.

The healthcare benchmark in this repo focuses on the branch that produced the saved model artifact. It uses 13 engineered features: `high`, `low`, `trade_count`, `open`, `volume`, `vwap`, `RSI_14`, `MA_10`, `MA_50`, `MA_200`, `MACD_12_26_9`, `MACDh_12_26_9`, and `MACDs_12_26_9`.

The feature-selection script shows the two checks used to narrow the feature set:

- a Spearman correlation matrix to spot redundant columns
- random-forest feature importances on the training window to see which engineered signals the tree actually uses

Those outputs are written to:

- `results/healthcare_feature_correlations.png`
- `results/healthcare_feature_importances.png`
- `results/healthcare_feature_selection_summary.txt`

## Benchmark design

The repo keeps two evaluation setups:

### 1. Random row-wise holdout

This is the faster development split used to compare ordinary untuned models under a standard 80/20 partition.

From `results/healthcare_holdout_summary.txt`:

- majority-class baseline: `0.6823`
- fresh random forest (`n_estimators=100`): `0.8710`

That is a large lift, but it is not the cleanest forward-looking estimate because nearby rows can land on opposite sides of the split.

### 2. Date-blocked forward split

This is the stricter benchmark used for the main artifact-vs-baseline test. Training ends before the test window begins, so the model is always evaluated on later dates.

From `results/healthcare_time_split_summary.txt`:

- majority-class baseline: `0.6867`
- fresh random forest (`n_estimators=100`): `0.6434`
- fresh random forest (`n_estimators=300, max_depth=10`): `0.6813`

From `results/healthcare_saved_model_comparison.txt`:

- preserved `best_model.pkl`: `0.8459`

## How to read those numbers

The random holdout shows how strongly the feature stack can separate direction classes under a standard 80/20 split. The forward date-blocked split is the main benchmark because it evaluates both the baseline models and the trained artifact on a chronological holdout with no leakage across train and test windows.

## About `best_model.pkl`

The trained model artifact is `best_model.pkl`. It is too large to commit directly to GitHub, so the repo treats it as an external artifact:

- place the file at `artifacts/best_model.pkl`, or
- set `MODEL_PATH` to its location before running `scripts/compare_saved_model.py`

`scripts/compare_saved_model.py` evaluates that artifact on the same date-blocked split used for the fresh baselines. The repo pins `scikit-learn==1.5.0` because the preserved artifact was serialized with that version.

The original raw healthcare parquet files used by the training pipeline are not bundled in the repo. They lived as separate local files (`healthcareSmallcap.parquet`, `healthcareMidcap.parquet`, and `healthcareLargecap.parquet`) and are referenced through `HEALTHCARE_DATA_DIR` when you want to run the full training path.

## Alpaca integration

The trading connector is environment-variable driven:

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

A local `.env` file is fine for development.

## How to run the benchmark

Install the package:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

Run the full no-leakage comparison:

```bash
python scripts/compare_saved_model.py
```

Run only the untuned baselines:

```bash
python scripts/compare_saved_model.py --benchmark-model baseline
```

Run only the trained artifact:

```bash
python scripts/compare_saved_model.py --benchmark-model trained
```

By default, `scripts/compare_saved_model.py` looks for `artifacts/best_model.pkl`. To point at an external copy, set `MODEL_PATH=/path/to/best_model.pkl`.

To inspect the training path without launching the expensive run:

```bash
python scripts/train_best_model_pipeline.py
```

To regenerate the comparison together with the supporting feature-selection and baseline outputs:

```bash
python scripts/run_healthcare_review.py
pytest -q
```

## Notes

- The bundled public data is intentionally limited to keep the repo lightweight and runnable.
- The full healthcare training parquet files are intentionally left external because they are too large to bundle.
- Resume / contact: `royrliu@utexas.edu`
