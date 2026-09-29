"""Final locked test-set evaluation for E1-E4.

This script evaluates the already-selected best-validation checkpoint from
each primary experiment on the untouched CIFAR-10 test set.

IMPORTANT:
- No training is performed.
- No controller decisions are made.
- No checkpoint is selected using test metrics.
- Do not retune hyperparameters after inspecting these results.

Run from the repository root:
    python -m experiments.evaluate_final_test
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    INITIAL_LEARNING_RATE,
    RESULTS_DIR,
    SEED,
    set_seed,
)
from data.cifar10 import CIFAR10DataModule
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import evaluate_model
from models.custom_cnn import CustomCNN


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_completed_timestamp_run(run_name):
    """Find the newest completed primary timestamped run."""
    root = Path(RESULTS_DIR) / run_name

    if not root.exists():
        raise FileNotFoundError(
            f"Could not find primary run directory: {root}"
        )

    candidates = []

    for run_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        reverse=True,
    ):
        metadata_path = run_dir / f"{run_name}_metadata.json"
        checkpoint_path = run_dir / f"{run_name}_best.pt"

        if not metadata_path.exists() or not checkpoint_path.exists():
            continue

        try:
            metadata = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            continue

        if metadata.get("status") != "completed":
            continue

        candidates.append(
            {
                "run_dir": run_dir,
                "metadata_path": metadata_path,
                "checkpoint_path": checkpoint_path,
                "metadata": metadata,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"No completed primary run found under: {root}"
        )

    return candidates[0]


def resolve_primary_runs():
    """Resolve the four frozen primary experiments."""

    e1_name = (
        f"fixed_fixed_batch32"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )

    e2_name = (
        f"adaptive_batch_fixed"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )

    e3_name = (
        f"adaptive_lr_batch32"
        f"_refLR{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )

    e4_name = (
        f"adaptive_batch_adaptive_lr"
        f"_refLR{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )

    # E1 and E2 store checkpoints in the shared checkpoints directory.
    e1_metadata_path = require_file(
        Path(RESULTS_DIR) / f"{e1_name}_metadata.json",
        "E1 metadata",
    )
    e1_checkpoint_path = require_file(
        Path(CHECKPOINT_DIR) / f"{e1_name}_best.pt",
        "E1 best checkpoint",
    )

    e2_metadata_path = require_file(
        Path(RESULTS_DIR) / f"{e2_name}_metadata.json",
        "E2 metadata",
    )
    e2_checkpoint_path = require_file(
        Path(CHECKPOINT_DIR) / f"{e2_name}_best.pt",
        "E2 best checkpoint",
    )

    e1_metadata = load_json(e1_metadata_path)
    e2_metadata = load_json(e2_metadata_path)

    if e1_metadata.get("status") != "completed":
        raise ValueError("E1 metadata is not marked completed.")
    if e2_metadata.get("status") != "completed":
        raise ValueError("E2 metadata is not marked completed.")

    # E3 and E4 use timestamped run directories.
    e3 = find_completed_timestamp_run(e3_name)
    e4 = find_completed_timestamp_run(e4_name)

    return [
        {
            "id": "E1",
            "label": "Fixed batch + fixed LR",
            "run_name": e1_name,
            "checkpoint_path": e1_checkpoint_path,
            "metadata_path": e1_metadata_path,
            "metadata": e1_metadata,
        },
        {
            "id": "E2",
            "label": "Adaptive batch + fixed LR",
            "run_name": e2_name,
            "checkpoint_path": e2_checkpoint_path,
            "metadata_path": e2_metadata_path,
            "metadata": e2_metadata,
        },
        {
            "id": "E3",
            "label": "Fixed batch + adaptive LR",
            "run_name": e3_name,
            "checkpoint_path": e3["checkpoint_path"],
            "metadata_path": e3["metadata_path"],
            "metadata": e3["metadata"],
        },
        {
            "id": "E4",
            "label": "Adaptive batch + adaptive LR",
            "run_name": e4_name,
            "checkpoint_path": e4["checkpoint_path"],
            "metadata_path": e4["metadata_path"],
            "metadata": e4["metadata"],
        },
    ]


def load_checkpoint(model, checkpoint_path):
    """Load only the already-selected best checkpoint weights."""
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint lacks model_state_dict: {checkpoint_path}"
        )

    if "epoch" not in checkpoint:
        raise ValueError(
            f"Checkpoint lacks epoch: {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return checkpoint


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def save_comparison_csv(rows, path):
    fields = [
        "experiment",
        "label",
        "checkpoint_epoch",
        "test_loss",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "num_samples",
        "training_time_seconds",
        "optimizer_updates",
        "checkpoint_path",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(SEED)

    print("=" * 78)
    print("FINAL LOCKED CIFAR-10 TEST EVALUATION")
    print("=" * 78)
    print("No training will be performed.")
    print("No controller will be updated.")
    print("Best checkpoints were selected using validation data only.")
    print("Do not retune E1-E4 after inspecting these test results.")
    print("=" * 78)

    runs = resolve_primary_runs()

    print("\nResolved primary checkpoints:")
    for run in runs:
        print(
            f"  {run['id']}: "
            f"{run['checkpoint_path']}"
        )

    # Create the test loader only now, after E1-E4 are frozen.
    data = CIFAR10DataModule()
    test_loader = data.get_test_loader(batch_size=256)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output_dir = (
        Path(RESULTS_DIR)
        / "final_test_evaluation"
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    comparison_rows = []
    combined_json = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "dataset": "CIFAR-10 official test set",
        "num_test_samples": len(test_loader.dataset),
        "seed": SEED,
        "selection_policy": (
            "Each checkpoint was selected using validation loss before "
            "test-set evaluation. Test results are evaluation-only."
        ),
        "experiments": {},
    }

    for run in runs:
        print("\n" + "-" * 78)
        print(
            f"{run['id']} | {run['label']}"
        )
        print("-" * 78)

        model = CustomCNN().to(DEVICE)

        checkpoint = load_checkpoint(
            model,
            run["checkpoint_path"],
        )

        metrics = evaluate_model(
            model,
            test_loader,
            DEVICE,
        )

        experiment_dir = output_dir / run["id"]
        experiment_dir.mkdir(parents=True, exist_ok=False)

        # Preserve detailed metrics, including per-class scores.
        metrics_for_json = {
            key: value
            for key, value in metrics.items()
            if key not in ("y_true", "y_pred")
        }

        detail = {
            "experiment": run["id"],
            "label": run["label"],
            "run_name": run["run_name"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_path": str(run["checkpoint_path"]),
            "metadata_path": str(run["metadata_path"]),
            "metrics": metrics_for_json,
        }

        save_json(
            detail,
            experiment_dir / "test_metrics.json",
        )

        plot_confusion_matrix(
            metrics["y_true"],
            metrics["y_pred"],
            experiment_dir / "test_confusion_matrix.png",
            normalize=False,
            title=f"{run['id']} test confusion matrix",
        )

        plot_confusion_matrix(
            metrics["y_true"],
            metrics["y_pred"],
            experiment_dir / "test_confusion_matrix_normalized.png",
            normalize=True,
            title=f"{run['id']} normalized test confusion matrix",
        )

        training_time = run["metadata"].get(
            "total_training_time_seconds"
        )
        optimizer_updates = run["metadata"].get(
            "total_optimizer_updates"
        )

        row = {
            "experiment": run["id"],
            "label": run["label"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "test_loss": float(metrics["loss"]),
            "test_accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "num_samples": int(metrics["num_samples"]),
            "training_time_seconds": training_time,
            "optimizer_updates": optimizer_updates,
            "checkpoint_path": str(run["checkpoint_path"]),
        }

        comparison_rows.append(row)
        combined_json["experiments"][run["id"]] = detail

        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Test loss:        {metrics['loss']:.4f}")
        print(f"Test accuracy:    {metrics['accuracy']:.2%}")
        print(f"Macro precision:  {metrics['macro_precision']:.4f}")
        print(f"Macro recall:     {metrics['macro_recall']:.4f}")
        print(f"Macro F1:         {metrics['macro_f1']:.4f}")

    save_comparison_csv(
        comparison_rows,
        output_dir / "final_test_comparison.csv",
    )
    save_json(
        combined_json,
        output_dir / "final_test_results.json",
    )

    print("\n" + "=" * 78)
    print("FINAL TEST COMPARISON")
    print("=" * 78)

    for row in comparison_rows:
        print(
            f"{row['experiment']} | "
            f"Acc: {row['test_accuracy']:.2%} | "
            f"Loss: {row['test_loss']:.4f} | "
            f"Macro F1: {row['macro_f1']:.4f} | "
            f"Checkpoint epoch: {row['checkpoint_epoch']}"
        )

    print("=" * 78)
    print(f"Final test outputs saved to: {output_dir}")
    print(
        "The test set is now considered opened for final reporting. "
        "Do not use these results to retune E1-E4."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
