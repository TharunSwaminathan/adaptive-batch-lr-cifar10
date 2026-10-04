"""Final locked CIFAR-100 test-set evaluation for E1-E4.

This script evaluates the already-selected best-validation-loss checkpoint
from each frozen CIFAR-100 experiment on the official 10,000-image test set.

IMPORTANT:
- No training is performed.
- No controller decisions are made.
- No checkpoint is selected using test metrics.
- Do not retune CIFAR-100 E1-E4 after inspecting these results.

Run from the repository root:
    python -m experiments.evaluate_cifar100_final_test
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.torch_version import TorchVersion

from config import DEVICE, RESULTS_DIR, set_seed
from data.cifar100 import CIFAR100DataModule
from evaluation.metrics import evaluate_model
from models.custom_cnn import CustomCNN


SEED = 42
TEST_BATCH_SIZE = 256

RUNS = [
    {
        "id": "E1",
        "label": "Fixed batch + fixed LR",
        "run_name": "cifar100_E1_fixed_fixed_batch32_lr0.01_seed42",
        "checkpoint": Path("checkpoints/cifar100/E1")
        / "cifar100_E1_fixed_fixed_batch32_lr0.01_seed42_best.pt",
        "expected_epoch": 17,
        "validation_accuracy": 0.3000,
        "validation_macro_f1": 0.2847,
        "optimizer_updates": 28140,
        "training_time_seconds": 444.38,
    },
    {
        "id": "E2",
        "label": "Adaptive batch + fixed LR",
        "run_name": "cifar100_E2_adaptive_batch_fixed_lr0.01_seed42",
        "checkpoint": Path("checkpoints/cifar100/E2")
        / "cifar100_E2_adaptive_batch_fixed_lr0.01_seed42_best.pt",
        "expected_epoch": 17,
        "validation_accuracy": 0.3096,
        "validation_macro_f1": 0.2890,
        "optimizer_updates": 23219,
        "training_time_seconds": 436.16,
    },
    {
        "id": "E3",
        "label": "Fixed batch + adaptive LR",
        "run_name": "cifar100_E3_fixed_batch_adaptive_lr_seed42",
        "checkpoint": Path("checkpoints/cifar100/E3")
        / "cifar100_E3_fixed_batch_adaptive_lr_seed42_best.pt",
        "expected_epoch": 20,
        "validation_accuracy": 0.3332,
        "validation_macro_f1": 0.3152,
        "optimizer_updates": 28140,
        "training_time_seconds": 457.03,
    },
    {
        "id": "E4",
        "label": "Adaptive batch + adaptive LR",
        "run_name": "cifar100_E4_adaptive_batch_adaptive_lr_seed42",
        "checkpoint": Path("checkpoints/cifar100/E4")
        / "cifar100_E4_adaptive_batch_adaptive_lr_seed42_best.pt",
        "expected_epoch": 17,
        "validation_accuracy": 0.3182,
        "validation_macro_f1": 0.2974,
        "optimizer_updates": 23219,
        "training_time_seconds": 442.02,
    },
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkpoint(run):
    path = Path(run["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {run['id']} checkpoint: {path}"
        )
    return path


def load_checkpoint(model, path, expected_epoch):
    """Load the already-selected best-validation-loss checkpoint."""
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint has no model_state_dict: {path}"
        )

    if "epoch" not in checkpoint:
        raise ValueError(
            f"Checkpoint has no epoch: {path}"
        )

    checkpoint_epoch = int(checkpoint["epoch"])

    if checkpoint_epoch != expected_epoch:
        raise ValueError(
            f"{path} is epoch {checkpoint_epoch}, but the frozen "
            f"validation-selected checkpoint should be epoch "
            f"{expected_epoch}. Refusing to open the test set with "
            "an unexpected checkpoint."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return checkpoint


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def save_per_class_csv(per_class, path):
    fields = [
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for class_name, values in per_class.items():
            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "support": values["support"],
                }
            )


def save_confusion_artifacts(y_true, y_pred, class_names, output_dir):
    labels = np.arange(len(class_names))

    raw = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    totals = raw.sum(axis=1, keepdims=True)

    normalized = np.divide(
        raw,
        totals,
        out=np.zeros_like(raw, dtype=float),
        where=totals != 0,
    )

    np.save(
        output_dir / "test_confusion_matrix.npy",
        raw,
    )

    np.savetxt(
        output_dir / "test_confusion_matrix.csv",
        raw,
        delimiter=",",
        fmt="%d",
    )

    np.save(
        output_dir / "test_confusion_matrix_normalized.npy",
        normalized,
    )

    np.savetxt(
        output_dir / "test_confusion_matrix_normalized.csv",
        normalized,
        delimiter=",",
        fmt="%.8f",
    )

    # A 100-class confusion matrix is too dense for per-cell text labels.
    # Save a clean heatmap with class indices instead.
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Fraction within true class",
    )

    tick_positions = np.arange(0, len(class_names), 10)

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title("Normalized CIFAR-100 test confusion matrix")

    fig.tight_layout()
    fig.savefig(
        output_dir / "test_confusion_matrix_normalized.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    class_map = [
        {
            "class_index": index,
            "class_name": name,
        }
        for index, name in enumerate(class_names)
    ]

    save_json(
        class_map,
        output_dir / "cifar100_class_index_map.json",
    )


def main():
    set_seed(SEED)

    print("=" * 78)
    print("FINAL LOCKED CIFAR-100 TEST EVALUATION")
    print("=" * 78)
    print("No training will be performed.")
    print("No controller will be updated.")
    print("Checkpoints were selected using validation loss only.")
    print("CIFAR-100 E1-E4 are frozen.")
    print("Do not retune them after inspecting these test results.")
    print("=" * 78)

    # Verify all four checkpoints before constructing the test loader.
    resolved = []

    print("\nVerifying frozen checkpoints:")

    for run in RUNS:
        checkpoint_path = require_checkpoint(run)

        resolved.append(
            {
                **run,
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": sha256_file(
                    checkpoint_path
                ),
            }
        )

        print(
            f"  {run['id']}: "
            f"{checkpoint_path} "
            f"(expected epoch {run['expected_epoch']})"
        )

    # This is the point where the previously untouched official
    # CIFAR-100 test set is actually constructed for final evaluation.
    data = CIFAR100DataModule()

    class_names = tuple(
        data.test_dataset.classes
    )

    if len(class_names) != 100:
        raise ValueError(
            f"Expected 100 CIFAR-100 classes, found {len(class_names)}."
        )

    test_loader = data.get_test_loader(
        batch_size=TEST_BATCH_SIZE
    )

    if len(test_loader.dataset) != 10000:
        raise ValueError(
            "Expected the official 10,000-image CIFAR-100 test set."
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    output_dir = (
        Path(RESULTS_DIR)
        / "cifar100"
        / "final_test_evaluation"
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    comparison_rows = []

    combined = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CIFAR-100",
        "split": "official test",
        "num_test_samples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "seed": SEED,
        "selection_policy": (
            "All four checkpoints were selected using validation loss "
            "before the official CIFAR-100 test set was opened. "
            "Test metrics are evaluation-only and must not be used "
            "to retune E1-E4."
        ),
        "experiments": {},
    }

    for run in resolved:
        print("\n" + "-" * 78)
        print(f"{run['id']} | {run['label']}")
        print("-" * 78)

        model = CustomCNN(
            num_classes=100
        ).to(DEVICE)

        checkpoint = load_checkpoint(
            model,
            run["checkpoint"],
            run["expected_epoch"],
        )

        metrics = evaluate_model(
            model,
            test_loader,
            DEVICE,
            class_names=class_names,
        )

        experiment_dir = (
            output_dir / run["id"]
        )

        experiment_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        metrics_without_predictions = {
            key: value
            for key, value in metrics.items()
            if key not in ("y_true", "y_pred")
        }

        detail = {
            "experiment": run["id"],
            "label": run["label"],
            "run_name": run["run_name"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_path": str(run["checkpoint"]),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "validation_accuracy": run["validation_accuracy"],
            "validation_macro_f1": run["validation_macro_f1"],
            "optimizer_updates": run["optimizer_updates"],
            "training_time_seconds": run["training_time_seconds"],
            "test_metrics": metrics_without_predictions,
        }

        save_json(
            detail,
            experiment_dir / "test_metrics.json",
        )

        save_per_class_csv(
            metrics["per_class"],
            experiment_dir / "test_per_class_metrics.csv",
        )

        save_confusion_artifacts(
            metrics["y_true"],
            metrics["y_pred"],
            class_names,
            experiment_dir,
        )

        row = {
            "experiment": run["id"],
            "label": run["label"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_accuracy": run["validation_accuracy"],
            "test_accuracy": float(metrics["accuracy"]),
            "validation_macro_f1": run["validation_macro_f1"],
            "test_macro_f1": float(metrics["macro_f1"]),
            "test_loss": float(metrics["loss"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "num_test_samples": int(metrics["num_samples"]),
            "optimizer_updates": run["optimizer_updates"],
            "training_time_seconds": run["training_time_seconds"],
            "checkpoint_sha256": run["checkpoint_sha256"],
        }

        comparison_rows.append(row)
        combined["experiments"][run["id"]] = detail

        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Test loss:        {metrics['loss']:.4f}")
        print(f"Test accuracy:    {metrics['accuracy']:.2%}")
        print(f"Macro precision:  {metrics['macro_precision']:.4f}")
        print(f"Macro recall:     {metrics['macro_recall']:.4f}")
        print(f"Macro F1:         {metrics['macro_f1']:.4f}")

    comparison_fields = [
        "experiment",
        "label",
        "checkpoint_epoch",
        "validation_accuracy",
        "test_accuracy",
        "validation_macro_f1",
        "test_macro_f1",
        "test_loss",
        "macro_precision",
        "macro_recall",
        "num_test_samples",
        "optimizer_updates",
        "training_time_seconds",
        "checkpoint_sha256",
    ]

    with open(
        output_dir / "cifar100_final_test_comparison.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_fields,
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    save_json(
        combined,
        output_dir / "cifar100_final_test_results.json",
    )

    print("\n" + "=" * 78)
    print("FINAL CIFAR-100 TEST COMPARISON")
    print("=" * 78)

    for row in comparison_rows:
        gap = (
            100.0
            * (
                row["validation_accuracy"]
                - row["test_accuracy"]
            )
        )

        print(
            f"{row['experiment']} | "
            f"Val Acc: {row['validation_accuracy']:.2%} | "
            f"Test Acc: {row['test_accuracy']:.2%} | "
            f"Test F1: {row['test_macro_f1']:.4f} | "
            f"Val-Test: {gap:+.2f} pp | "
            f"Epoch: {row['checkpoint_epoch']}"
        )

    print("=" * 78)
    print(f"Final outputs saved to: {output_dir}")
    print(
        "The CIFAR-100 test set is now opened for final reporting. "
        "Do not use these results to retune E1-E4."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
