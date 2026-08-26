Place optional large model artifacts here when you want to run comparisons that use the preserved trained model.

Supported path:

- `artifacts/best_model.pkl`

You can also keep the artifact elsewhere and point to it with `MODEL_PATH=/path/to/best_model.pkl`.

The public repo does not commit the file because it is too large for normal GitHub storage.

Recommended hosting: attach `best_model.pkl` to a GitHub Release and download it into `artifacts/` before running the benchmark.
