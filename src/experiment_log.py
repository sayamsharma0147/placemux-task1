import csv
import os
from datetime import datetime

LOG_PATH = "experiments.csv"


def log_experiment(run_id: str, params: dict, metrics: dict, log_path: str = LOG_PATH) -> None:
    """Append one row per run. Creates the file with headers on first run."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        **params,
        **metrics,
    }
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged run '{run_id}' to {log_path}")
