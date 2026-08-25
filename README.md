# stock-movement-predictor

This repository is a cleaned-up public version of a 2024-2025 stock-movement modeling project. The core question was whether short-horizon direction could be predicted from technical features in a way that was still inspectable enough to debug and compare across model families. I tried linear baselines first, used random forests to check whether the engineered indicators were carrying signal, explored SVM and CNN variants, and used Alpaca paper-trading integration to test the full data-to-signal pipeline.

This is an educational modeling and backtesting project. It was not used for live trading with real funds, and nothing here is financial advice.

## What this public repo centers on

The most reproducible branch I was able to preserve cleanly is a healthcare-sector direction benchmark:

- label: whether the price is higher 7 steps ahead
- features: price/volume fields plus RSI, moving averages, and MACD-derived indicators
- split: random 80/20 holdout with `random_state=42`
- comparison: majority-class baseline versus random forest

On the bundled healthcare holdout dataset, that benchmark gives:

| Model | Accuracy |
|---|---|
| Majority-class baseline | 0.6780 |
| Random forest | 0.8710 |

That gap is the main result this public repo is built around. It is straightforward to rerun from a fresh clone and it reflects the broader lesson from the original project: the feature pipeline mattered more than trying to jump immediately to a more complicated model.

## Project structure

- `data/healthcare_direction_holdout.parquet`: bundled feature matrix for the reproducible healthcare benchmark.
- `data/sample_stock_data.csv`: smaller raw sample used by the feature-engineering and cross-validation example.
- `src/stock_movement_predictor/feature_engineering.py`: rolling indicators including moving averages, RSI, MACD, and an RVI-style signal.
- `src/stock_movement_predictor/healthcare_benchmark.py`: majority-baseline versus random-forest benchmark runner.
- `src/stock_movement_predictor/models.py`: baseline model suite plus the XGBoost path from the broader project.
- `src/stock_movement_predictor/backtesting.py`: time-series cross-validation runner for the smaller public sample workflow.
- `src/stock_movement_predictor/alpaca.py`: optional Alpaca integration using environment variables only.
- `scripts/run_healthcare_holdout.py`: one-command reproduction of the headline benchmark.
- `scripts/run_sample_backtest.py`: smaller public sample showing the end-to-end feature and backtest pipeline.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/run_healthcare_holdout.py
```

With `uv`:

```bash
uv venv
uv pip install -e .
uv run python scripts/run_healthcare_holdout.py
```

## Reproducing the main result

```bash
python scripts/run_healthcare_holdout.py
```

This writes `results/healthcare_holdout_summary.txt` and prints a summary like:

```text
Healthcare direction benchmark
rows=344698
feature_count=13
positive_rate=0.3217
test_size=0.20
random_state=42
dummy_most_frequent: accuracy=0.6780
random_forest_100: accuracy=0.8710
```

## Broader modeling process

This repo reflects the structure of the larger project even though not every old artifact is bundled here.

- Linear regression was the starting baseline.
- Random forests were useful both as a classifier and as a quick way to sanity-check which indicators were pulling their weight.
- XGBoost was part of the later model-comparison work and remains in the public codebase.
- SVM and CNN experiments were explored but tended to overfit relative to the simpler tree-based paths.
- Alpaca integration was used for paper-trading and signal-generation experiments, mainly to verify the whole loop from market data to order logic.

## Secondary example

The repo also includes a smaller sample-based cross-validation path:

```bash
python scripts/run_sample_backtest.py
```

That example is useful for understanding the feature engineering and evaluation flow on raw sample market data, but the benchmark above is the cleaner public result to focus on.

## Alpaca configuration

```bash
export ALPACA_API_KEY=...
export ALPACA_API_SECRET=...
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Use a local `.env` file if you want, but never commit real credentials.

Contact: `royrliu@utexas.edu`
