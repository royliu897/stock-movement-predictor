from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=REPO_ROOT)


def main() -> None:
    # This is the shortest path a reviewer can run to see feature selection, baselines, and the saved-model lift.
    for script_name in [
        "analyze_healthcare_features.py",
        "run_healthcare_holdout.py",
        "run_healthcare_time_split.py",
        "compare_saved_model.py",
    ]:
        run(script_name)


if __name__ == "__main__":
    main()
