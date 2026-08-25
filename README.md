# stock-movement-predictor

This repository packages my 2024-2025 stock modeling project into a cleaner public structure. The project started with a simple question: could short-term price movement be modeled in a way that was still interpretable enough to debug? That led to a workflow built around data collection, technical-indicator feature engineering, backtesting, and comparing several model families before settling on XGBoost as the final direction.

The progression was roughly:

- start with linear regression as a baseline,
- use random forests to sanity-check the feature set,
- test more complex models like SVMs and CNN-style experiments,
- move to XGBoost once the tree-based pipeline was working well,
- connect the prediction pipeline to Alpaca for paper-trading and live data plumbing.

This is an educational backtesting project, not financial advice, and it was not used for live trading with real funds.

## Project structure

- `src/stock_movement_predictor/data.py`: CSV/parquet loading and schema checks.
- `src/stock_movement_predictor/feature_engineering.py`: rolling indicators such as moving averages, RSI, MACD, and an RVI-style signal.
- `src/stock_movement_predictor/models.py`: linear regression and random forest baselines plus the final XGBoost path.
- `src/stock_movement_predictor/backtesting.py`: time-series cross-validation backtest runner.
- `src/stock_movement_predictor/alpaca.py`: optional Alpaca integration using environment variables only.
- `scripts/run_sample_backtest.py`: runnable public example on bundled sample data.
- `scripts/legacy_model_comparison.py`: comparison script for older saved model artifacts against the surviving local parquet datasets.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/run_sample_backtest.py
```

If you prefer `uv`:

```bash
uv venv
uv pip install -e .
uv run python scripts/run_sample_backtest.py
```

## Experimental workflow

### 1. Data collection and preprocessing

The original project used historical price bars collected through Alpaca plus larger locally stored CSV/parquet tables. From those bars, I built a technical-feature pipeline around:

- price and volume features,
- rolling moving averages,
- RSI,
- MACD and related components,
- simple volatility and momentum signals.

That feature pipeline ended up mattering more than trying increasingly exotic model classes.

### 2. Model comparison

I tried several model families over the course of the project:

- linear regression as a baseline,
- random forests for feature sanity checks and interpretability,
- SVM and CNN-style experiments,
- XGBoost as the final model family.

The main lesson was that the more complex models were not automatically better. In practice, SVM and CNN variants tended to overfit more easily on the data I had, while the tree-based pipeline was easier to tune and reason about.

### 3. Feature selection and final direction

Random forests were useful earlier in the project because they gave a relatively readable way to think about feature importance and whether the hand-built indicator set was doing anything useful at all. Once that workflow was stable, XGBoost became the final direction because it kept the same general feature-engineering structure while giving better predictive performance.

### 4. Execution and paper trading

I also wired the project into Alpaca for paper-trading experiments. That part was less about claiming a production trading system and more about proving that the data, feature, model, and execution layers could be connected into one pipeline.

## Replicated results

What I was able to rerun from the surviving artifacts splits into two categories.

### Publicly reproducible from this repo

Running `python scripts/run_sample_backtest.py` on the bundled sample data reproduced these mean time-series cross-validation accuracies:

- `linear_regression_baseline`: `0.5194`
- `random_forest_baseline`: `0.5039`
- `xgboost_final`: `0.5013`

Those numbers come from the small public sample included in this repository. A reviewer can reproduce them directly from a fresh clone.

### Reproduced locally from older saved artifacts

Using the surviving local parquet datasets and serialized models, I also reran the legacy comparison path in `scripts/legacy_model_comparison.py`. That produced:

- `best_model.pkl` on combined healthcare parquet tables: `0.950420` classifier accuracy
- `best_model.pkl` on combined tech parquet tables: `0.645539` classifier accuracy
- legacy cap-bucket regressors: roughly `0.505` directional accuracy despite high price-level `R^2`

Those results are genuine reruns, but they are not currently reproducible from a clean public clone because the larger parquet datasets and saved model pickle files are not bundled in this repository.

## What a reviewer can run

A reviewer can reproduce the public path end to end:

```bash
python scripts/run_sample_backtest.py
```

That will:

- load the bundled sample dataset,
- rebuild features,
- run the baseline and XGBoost models with time-series cross-validation,
- regenerate `results/sample_backtest_summary.txt`,
- regenerate `results/sample_backtest_accuracy.png`.

A reviewer cannot rerun the larger legacy artifact comparison without access to the original external parquet datasets and saved model files.

## Alpaca configuration

Use environment variables rather than hardcoded credentials:

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

You can place these in a local `.env` file for convenience, but real credentials should never be committed.

## Notes

- The public repo reflects the actual arc of the project: compare several models, use random forests to understand the feature set, then move to XGBoost for the final direction.
- The bundled sample data is intentionally small so the repo stays lightweight and runnable.
- Some larger historical datasets and older model artifacts stayed local and are therefore documented rather than shipped.

Contact: `royrliu@utexas.edu`
