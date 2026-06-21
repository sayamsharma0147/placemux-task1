import csv
import os
from datetime import datetime

LOG_PATH = "experiments.csv"


def log_experiment(run_id: str, params: dict, metrics: dict, log_path: str = LOG_PATH) -> None:
    """Append one row per run. Falls back to a timestamped file if the
    primary log is locked/unwritable (e.g. open in Excel)."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        **params,
        **metrics,
    }

    try:
        _write_row(row, log_path)
        print(f"Logged run '{run_id}' to {log_path}")
    except PermissionError:
        fallback_path = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"WARNING: '{log_path}' is locked (maybe open in Excel?). "
              f"Writing to '{fallback_path}' instead.")
        try:
            _write_row(row, fallback_path)
            print(f"Logged run '{run_id}' to {fallback_path}")
        except OSError as e:
            print(f"ERROR: could not write experiment log anywhere: {e}")
    except OSError as e:
        print(f"ERROR: could not write experiment log to '{log_path}': {e}")


def _write_row(row: dict, log_path: str) -> None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)