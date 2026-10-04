"""Final test evaluation for frozen 40-epoch E1-E4 primary runs.

All four completed validation-selected runs are verified before the official
CIFAR test loader is created. Do not use test results to retune the protocol.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from config import DEVICE, RESULTS_DIR, set_seed
from evaluation.metrics import evaluate_model
from experiments.datasets import DATASETS, dataset_settings
from experiments.primary_protocol import PRIMARY_CONFIGS, validate_primary_protocol
from experiments.summarize_primary import latest_manifest
from models.custom_cnn import CustomCNN

EXPERIMENTS = ("E1", "E2", "E3", "E4")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_runs(dataset):
    resolved = []
    for exp in EXPERIMENTS:
        manifest_path, manifest = latest_manifest(dataset, exp)
        checkpoint_path = Path(manifest["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint["epoch"]) != int(manifest["best_checkpoint_epoch"]):
            raise ValueError(f"{exp}: checkpoint epoch does not match validation manifest")
        if manifest["epochs_completed"] != PRIMARY_CONFIGS[exp]["epochs"]:
            raise ValueError(f"{exp}: run did not complete the frozen 40-epoch budget")
        resolved.append((exp, manifest_path, manifest, checkpoint_path, sha256_file(checkpoint_path)))
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--allow-repeat", action="store_true", help="Allow another final-test output folder if one already exists.")
    args = parser.parse_args()

    validate_primary_protocol()
    set_seed(42)
    resolved = resolve_runs(args.dataset)

    final_root = Path(RESULTS_DIR) / "final_test" / args.dataset
    existing = list(final_root.glob("*/final_test_summary.json")) if final_root.exists() else []
    if existing and not args.allow_repeat:
        raise RuntimeError(
            "A final-test evaluation already exists for this dataset. "
            "Use --allow-repeat only for a documented technical rerun."
        )

    # Test data is loaded only after every E1-E4 run passes preflight checks.
    data_cls, num_classes, dataset_label = dataset_settings(args.dataset)
    data = data_cls()
    class_names = tuple(data.class_names)
    test_loader = data.get_test_loader(batch_size=256)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = final_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    detail = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label,
        "dataset_key": args.dataset,
        "split": "official test",
        "protocol_epochs": 40,
        "seed": 42,
        "experiments": {},
    }

    for exp, manifest_path, manifest, checkpoint_path, checksum in resolved:
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model = CustomCNN(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate_model(model, test_loader, DEVICE, class_names=class_names)

        row = {
            "experiment": exp,
            "epochs_completed": manifest["epochs_completed"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_accuracy": manifest["validation_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "validation_macro_f1": manifest["validation_macro_f1"],
            "test_macro_f1": metrics["macro_f1"],
            "test_loss": metrics["loss"],
            "optimizer_updates": manifest["optimizer_updates"],
            "training_time_seconds": manifest["training_time_seconds"],
        }
        rows.append(row)
        detail["experiments"][exp] = {
            **row,
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checksum,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "per_class": metrics["per_class"],
        }

    csv_path = output_dir / "final_test_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "final_test_summary.json"
    json_path.write_text(json.dumps(detail, indent=2, allow_nan=False), encoding="utf-8")

    print("Final test evaluation complete. Do not retune E1-E4 from these results.")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
