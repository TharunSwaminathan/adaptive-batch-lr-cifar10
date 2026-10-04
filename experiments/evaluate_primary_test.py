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

import numpy as np
import torch
from torch.torch_version import TorchVersion

from config import DEVICE, RESULTS_DIR, set_seed
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import evaluate_model, generalization_gap
from experiments.datasets import DATASETS, dataset_settings
from experiments.primary_protocol import (
    PRIMARY_CONFIGS,
    protocol_fingerprint,
    validate_primary_protocol,
)
from experiments.summarize_primary import latest_manifest
from models.custom_cnn import CustomCNN

EXPERIMENTS = ("E1", "E2", "E3", "E4")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_metadata(checkpoint):
    metadata = checkpoint.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing run_metadata")
    return metadata


def resolve_runs(dataset):
    resolved = []
    commits = set()

    for exp in EXPERIMENTS:
        manifest_path, manifest = latest_manifest(dataset, exp)
        expected = PRIMARY_CONFIGS[exp]
        expected_fingerprint = protocol_fingerprint(expected)

        checkpoint_path = Path(manifest["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        metadata = _checkpoint_metadata(checkpoint)
        if int(checkpoint["epoch"]) != int(manifest["best_checkpoint_epoch"]):
            raise ValueError(f"{exp}: checkpoint epoch does not match validation manifest")
        if manifest["epochs_completed"] != expected["epochs"]:
            raise ValueError(f"{exp}: run did not complete the frozen epoch budget")
        if manifest.get("seed") != expected["seed"]:
            raise ValueError(f"{exp}: manifest seed does not match frozen protocol")
        if manifest.get("protocol_fingerprint") != expected_fingerprint:
            raise ValueError(f"{exp}: manifest protocol fingerprint mismatch")
        if metadata.get("experiment") != exp or metadata.get("dataset_key") != dataset:
            raise ValueError(f"{exp}: checkpoint metadata identity mismatch")
        if metadata.get("seed") != expected["seed"]:
            raise ValueError(f"{exp}: checkpoint seed mismatch")
        if metadata.get("protocol_fingerprint") != expected_fingerprint:
            raise ValueError(f"{exp}: checkpoint protocol fingerprint mismatch")

        commit = manifest.get("git_commit")
        if not commit or metadata.get("git_commit") != commit:
            raise ValueError(f"{exp}: missing or inconsistent git commit provenance")
        commits.add(commit)

        resolved.append(
            (exp, manifest_path, manifest, checkpoint_path, sha256_file(checkpoint_path))
        )

    if len(commits) != 1:
        raise ValueError(
            "Primary E1-E4 runs were produced from different Git commits. "
            "Use one frozen code revision for the final comparison."
        )

    return resolved, next(iter(commits))


def _serializable_metrics(metrics):
    return {key: value for key, value in metrics.items() if key not in ("y_true", "y_pred")}


def _save_per_class_csv(per_class, path):
    rows = []
    for class_name, values in per_class.items():
        rows.append({"class": class_name, **values})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("class", "precision", "recall", "f1", "support"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Allow another final-test output folder if one already exists.",
    )
    args = parser.parse_args()

    validate_primary_protocol()
    set_seed(42)
    resolved, frozen_git_commit = resolve_runs(args.dataset)

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
    train_eval_loader = data.get_train_eval_loader(batch_size=256)
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
        "frozen_git_commit": frozen_git_commit,
        "class_names": list(class_names),
        "experiments": {},
    }

    for exp, manifest_path, manifest, checkpoint_path, checksum in resolved:
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model = CustomCNN(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        train_metrics = evaluate_model(
            model, train_eval_loader, DEVICE, class_names=class_names
        )
        test_metrics = evaluate_model(
            model, test_loader, DEVICE, class_names=class_names
        )
        gap_pp = generalization_gap(
            train_metrics["accuracy"], test_metrics["accuracy"]
        )

        exp_dir = output_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=False)

        raw_matrix = plot_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            exp_dir / "test_confusion_matrix.png",
            normalize=False,
            class_names=class_names,
            title=f"{exp} official-test confusion matrix",
        )
        normalized_matrix = plot_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            exp_dir / "test_confusion_matrix_normalized.png",
            normalize=True,
            class_names=class_names,
            title=f"{exp} official-test confusion matrix (normalized)",
        )
        np.savetxt(exp_dir / "test_confusion_matrix.csv", raw_matrix, delimiter=",", fmt="%d")
        np.savetxt(
            exp_dir / "test_confusion_matrix_normalized.csv",
            normalized_matrix,
            delimiter=",",
            fmt="%.8f",
        )
        _save_per_class_csv(test_metrics["per_class"], exp_dir / "test_per_class_metrics.csv")

        test_metrics_path = exp_dir / "test_metrics.json"
        test_metrics_path.write_text(
            json.dumps(
                {
                    "experiment": exp,
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "train_eval_accuracy": train_metrics["accuracy"],
                    "generalization_gap_percentage_points": gap_pp,
                    **_serializable_metrics(test_metrics),
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        row = {
            "experiment": exp,
            "epochs_completed": manifest["epochs_completed"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_accuracy": manifest["validation_accuracy"],
            "train_eval_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "generalization_gap_percentage_points": gap_pp,
            "validation_macro_f1": manifest["validation_macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_loss": test_metrics["loss"],
            "optimizer_updates": manifest["optimizer_updates"],
            "training_time_seconds": manifest["training_time_seconds"],
        }
        rows.append(row)
        detail["experiments"][exp] = {
            **row,
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checksum,
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "per_class_metrics_path": str(exp_dir / "test_per_class_metrics.csv"),
            "confusion_matrix_path": str(exp_dir / "test_confusion_matrix.png"),
            "normalized_confusion_matrix_path": str(
                exp_dir / "test_confusion_matrix_normalized.png"
            ),
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
