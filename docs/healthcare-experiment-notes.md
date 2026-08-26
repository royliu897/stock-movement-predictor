# Healthcare Experiment Notes

This file records what survived from the original local stock-modeling project and how the public benchmark path was reconstructed.

## Surviving artifact paths

- local `train_model.py`
- local `generate_recommendations.py`
- local `Untitled--1.ipynb`
- local `xgbooost.ipynb`
- local `best_model.pkl`

## What `best_model.pkl` is tied to

The saved `best_model.pkl` artifact matches the healthcare random-forest branch used in `train_model.py`:

- label: 7-day direction, `close[t+7] > close[t]`
- dataset: merged healthcare small-, mid-, and large-cap parquet files
- features: 13 engineered columns
- estimator family: `RandomForestClassifier`
- split style in the surviving script: random row-wise `train_test_split`

The file itself does not match the heavier random-search / grid-search notebook hyperparameters exactly. The cleanest reading is that more than one experiment branch existed locally and the saved artifact and tuning notebooks came from adjacent parts of the same project.

## What survived from tuning

Two separate tuning paths still exist in notebook form:

- `Untitled--1.ipynb`
  - random search over random-forest hyperparameters
  - grid search around the random-search result
  - cross-validation scoring
- `xgbooost.ipynb`
  - Optuna-based XGBoost tuning
  - stratified cross-validation

Those notebooks are not clean enough to publish directly as the public pipeline, but they are enough to justify the structure of the refactored scripts:

- `scripts/tune_healthcare_random_forest.py`
- `scripts/reconstruct_healthcare_notebook_pipeline.py`
- `src/stock_movement_predictor/healthcare_benchmark.py`

The recovered notebook script mirrors the surviving RF notebook more directly:

- merged raw healthcare small/mid/large parquet files
- target: `close[t+7] > close[t]`
- random forest
- `RandomizedSearchCV(..., n_iter=10, cv=5, scoring="accuracy")`
- followed by `GridSearchCV(..., cv=5, scoring="accuracy")`

That path is kept separate from the lightweight public benchmark because it is closer to the original long-running search loop and depends on raw local parquet files that are not bundled in the repo.

## Why the repo shows two evaluation setups

The original branch that produced the strongest headline number used a random row-wise split. That is informative, but it can overstate performance on panel time-series data because rows from nearby dates are highly correlated.

For that reason the public repo also includes a stricter date-blocked benchmark:

- all training rows come from earlier dates
- all test rows come from later dates
- the preserved saved model is evaluated on that same forward split

That gives the more defensible retraining path to foreground in the repo, even though the preserved saved artifact remains the strongest surviving single model result.

## Why the artifact result is presented separately

The saved artifact scores much higher on the date-blocked comparison than the fresh blocked retrains in the public pipeline. At minimum, that means the current public retraining path is not a full reproduction of the original saved-model branch.

The two most plausible explanations are:

- the original artifact was trained under a different split regime, most likely the random row-wise split visible in the surviving script
- the original branch and the public retrain path differ in feature timing, preprocessing, or both

For that reason the repo treats `best_model.pkl` as a preserved benchmark artifact and keeps it separate from the fresh blocked retrains.

## Feature-selection reconstruction

The public feature-selection script reproduces the two checks that could be clearly justified from the surviving project:

1. rank pairwise feature relationships with a Spearman correlation matrix
2. fit a random forest on the training window and inspect feature importances

This does not reproduce every exploratory notebook exactly. It does reproduce the logic of how the final feature subset was narrowed down:

- remove obviously redundant or weak columns
- keep technical indicators that consistently carry signal
- verify that the tree actually uses the engineered features being added

## Public-repo boundary

This repo is meant to show the engineering process and the preserved result path without forcing a full retraining cycle. Large model files are kept optional, and the scripts are designed so a reviewer can still:

- inspect the feature engineering
- inspect the feature-selection workflow
- run fresh baselines
- compare those baselines against the preserved saved model if the artifact is available
