"""Create an E1-E4 validation summary from completed primary run manifests."""

import argparse
import csv
import json
from pathlib import Path

from config import RESULTS_DIR
from experiments.datasets import DATASETS
from experiments.primary_protocol import PRIMARY_CONFIGS, protocol_fingerprint

EXPERIMENTS = ("E1", "E2", "E3", "E4")


def latest_manifest(dataset, experiment):
    root = Path(RESULTS_DIR) / "primary" / dataset / experiment
    candidates = sorted(root.glob("*/run_manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No completed primary run found for {dataset}/{experiment}")
    path = candidates[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = PRIMARY_CONFIGS[experiment]
    if data.get("status") != "completed" or data.get("epochs_completed") != expected["epochs"]:
        raise ValueError(f"Invalid primary manifest: {path}")
    if data.get("is_pilot"):
        raise ValueError(f"Pilot manifest found in primary results: {path}")
    if data.get("experiment") != experiment or data.get("dataset") != dataset:
        raise ValueError(f"Manifest identity mismatch: {path}")
    if data.get("seed") != expected["seed"]:
        raise ValueError(f"Manifest seed mismatch: {path}")
    if data.get("protocol_fingerprint") != protocol_fingerprint(expected):
        raise ValueError(f"Manifest protocol mismatch: {path}")
    return path, data


def build_summary(dataset):
    rows = []
    for exp in EXPERIMENTS:
        path, item = latest_manifest(dataset, exp)
        rows.append({
            "experiment": exp,
            "epochs_completed": item["epochs_completed"],
            "best_checkpoint_epoch": item["best_checkpoint_epoch"],
            "best_validation_loss": item["best_validation_loss"],
            "validation_accuracy": item["validation_accuracy"],
            "validation_macro_f1": item["validation_macro_f1"],
            "optimizer_updates": item["optimizer_updates"],
            "training_time_seconds": item["training_time_seconds"],
            "manifest_path": str(path),
            "checkpoint_path": item["checkpoint_path"],
        })

    out = Path(RESULTS_DIR) / "primary" / dataset
    json_path = out / "primary_validation_summary.json"
    csv_path = out / "primary_validation_summary.csv"
    json_path.write_text(json.dumps({"dataset": dataset, "experiments": rows}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    args = parser.parse_args()
    build_summary(args.dataset)


if __name__ == "__main__":
    main()
