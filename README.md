# stock-movement-predictor

This repository packages my 2024-2025 short-horizon stock modeling work into a clean, reproducible structure. The public path centers on gradient-boosted trees with XGBoost, using linear regression and random forest baselines and a time-series cross-validation backtest; earlier CNN and SVM experiments were tested separately and dropped because they overfit the available data.

The original project lived across notebooks and scratch scripts, so this repo consolidates the parts that were substantively used: Alpaca-based historical data ingestion, technical-indicator feature engineering, baseline model comparison, and the later boosted-tree direction. Notebook review also showed that several older serialized artifacts still matter, but they represent different modeling paths: `best_model.pkl` is a legacy random-forest classifier on sector-specific indicator tables, while the much larger cap-bucket models are price regressors rather than directional classifiers.

This is an educational backtesting project, not financial advice, and it was not used for live trading with real funds.

## What is in this repo

- `src/stock_movement_predictor/data.py`: CSV/parquet loading and schema checks.
- `src/stock_movement_predictor/feature_engineering.py`: rolling indicators derived from the original notebooks, including moving averages, RSI, MACD histogram, and a simple RVI-style signal.
- `src/stock_movement_predictor/models.py`: linear regression and random forest baselines plus the final XGBoost model definition.
- `src/stock_movement_predictor/backtesting.py`: time-series cross-validation backtest runner.
- `src/stock_movement_predictor/alpaca.py`: optional Alpaca fetcher driven by environment variables only.
- `scripts/run_sample_backtest.py`: runnable example on bundled sample data.
- `scripts/legacy_model_comparison.py`: direct comparison of the strongest legacy classifier path against the weaker legacy regressor path.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/run_sample_backtest.py
```

If you prefer `uv`, the same project works with:

```bash
uv venv
uv pip install -e .
uv run python scripts/run_sample_backtest.py
```

## Data

The full historical datasets used during development were too large and messy to publish directly, so this repo bundles only a small sample CSV for reproducibility. The code accepts either:

- raw market bars in CSV/parquet with OHLCV columns, or
- pre-engineered feature matrices if you want to reproduce the older notebook workflow.

The original private working directory included Alpaca-fetched price bars and multiple parquet snapshots segmented by market-cap or sector buckets. Those large datasets and old serialized model files are intentionally not committed here.

What I could verify from the surviving local artifacts:

- the bundled sample backtest path is modest and roughly near chance on its small public sample,
- the legacy healthcare-sector classifier artifact performs very strongly on its matching engineered parquet tables,
- the huge legacy cap-bucket models are regressors and do not show strong directional accuracy when evaluated as up/down models.

Those differences are intentional to expose in the public repo rather than hide: some legacy paths were genuinely useful, and others were mostly exploratory.

## Alpaca credentials

Create a local `.env` file or export environment variables:

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Real credentials never belong in the repository. The original notebooks contained hardcoded paper-trading keys; those were removed from the public version and should be rotated before any future use.

## Notes

- The public repo preserves the actual modeling direction of the original work, but normalizes broken notebook paths and inconsistent intermediate scripts into one coherent package.
- The original data inventory is uneven: long single-symbol minute-bar histories exist for `SPY` and `FXI`, while the broader multi-stock parquet feature tables cover roughly 2022-2024 and are the closest surviving match to the strongest legacy classifier artifact.
- Resume / contact: `royrliu@utexas.edu`
