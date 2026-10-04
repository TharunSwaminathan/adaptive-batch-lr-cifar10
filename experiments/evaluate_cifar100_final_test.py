"""Final locked CIFAR-100 test evaluation for reconstructed E1-E4 runs.

The original CIFAR-100 checkpoints were lost. Therefore, this evaluator
does NOT hardcode historical best epochs from the old report. Instead it
verifies each NEW frozen rerun by requiring:

1. completed run metadata,
2. a best-validation checkpoint,
3. checkpoint epoch == metadata best_epoch,
4. validation metrics from that same rerun.

Only after all four runs pass those checks is the official CIFAR-100 test
loader created.

Run exactly once after the official reruns are complete:
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
    },
    {
        "id": "E2",
        "label": "Adaptive batch + fixed LR",
        "run_name": "cifar100_E2_adaptive_batch_fixed_lr0.01_seed42",
    },
    {
        "id": "E3",
        "label": "Fixed batch + adaptive LR",
        "run_name": "cifar100_E3_fixed_batch_adaptive_lr_seed42",
    },
    {
        "id": "E4",
        "label": "Adaptive batch + adaptive LR",
        "run_name": "cifar100_E4_adaptive_batch_adaptive_lr_seed42",
    },
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_frozen_run(run):
    exp = run["id"]
    run_name = run["run_name"]

    results_dir = Path("results/cifar100") / exp
    checkpoint_dir = Path("checkpoints/cifar100") / exp

    metadata_path = (
        results_dir
        / f"{run_name}_metadata.json"
    )

    validation_path = (
        results_dir
        / "validation_metrics.json"
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{run_name}_best.pt"
    )

    for path, label in (
        (metadata_path, "metadata"),
        (validation_path, "validation metrics"),
        (checkpoint_path, "best checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {exp} {label}: {path}"
            )

    metadata = load_json(
        metadata_path
    )

    validation = load_json(
        validation_path
    )

    if metadata.get("status") != "completed":
        raise ValueError(
            f"{exp} is not marked completed."
        )

    if metadata.get("dataset") != "CIFAR-100":
        raise ValueError(
            f"{exp} metadata dataset mismatch."
        )

    if int(metadata.get("seed", -1)) != SEED:
        raise ValueError(
            f"{exp} seed mismatch."
        )

    if int(metadata.get("epochs", -1)) != 20:
        raise ValueError(
            f"{exp} epoch-budget mismatch."
        )

    if int(metadata.get("num_classes", -1)) != 100:
        raise ValueError(
            f"{exp} num_classes mismatch."
        )

    if metadata.get(
        "test_set_used_during_training"
    ) is not False:
        raise ValueError(
            f"{exp} metadata does not confirm "
            "test-set isolation."
        )

    best_epoch = metadata.get(
        "best_epoch"
    )

    if best_epoch is None:
        raise ValueError(
            f"{exp} metadata has no best_epoch."
        )

    with torch.serialization.safe_globals(
        [TorchVersion]
    ):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    if int(checkpoint_epoch) != int(best_epoch):
        raise ValueError(
            f"{exp} checkpoint epoch {checkpoint_epoch} "
            f"does not match metadata best_epoch "
            f"{best_epoch}."
        )

    if validation.get("split") != "validation":
        raise ValueError(
            f"{exp} validation metrics file is not "
            "marked validation."
        )

    if int(
        validation.get(
            "checkpoint_epoch",
            -1,
        )
    ) != int(best_epoch):
        raise ValueError(
            f"{exp} validation metrics checkpoint epoch "
            "does not match metadata."
        )

    return {
        **run,
        "results_dir": results_dir,
        "metadata_path": metadata_path,
        "validation_path": validation_path,
        "checkpoint_path": checkpoint_path,
        "metadata": metadata,
        "validation": validation,
        "best_epoch": int(best_epoch),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
    }


def load_checkpoint(model, run):
    with torch.serialization.safe_globals(
        [TorchVersion]
    ):
        checkpoint = torch.load(
            run["checkpoint_path"],
            map_location=DEVICE,
            weights_only=True,
        )

    if int(checkpoint["epoch"]) != run["best_epoch"]:
        raise ValueError(
            f"{run['id']} checkpoint changed after "
            "preflight verification."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    return checkpoint


def save_per_class_csv(per_class, path):
    fields = [
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
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


def save_confusion_artifacts(
    y_true,
    y_pred,
    class_names,
    output_dir,
):
    labels = np.arange(
        len(class_names)
    )

    raw = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    totals = raw.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        raw,
        totals,
        out=np.zeros_like(
            raw,
            dtype=float,
        ),
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
        output_dir
        / "test_confusion_matrix_normalized.npy",
        normalized,
    )

    np.savetxt(
        output_dir
        / "test_confusion_matrix_normalized.csv",
        normalized,
        delimiter=",",
        fmt="%.8f",
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

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

    tick_positions = np.arange(
        0,
        len(class_names),
        10,
    )

    ax.set_xticks(
        tick_positions
    )

    ax.set_yticks(
        tick_positions
    )

    ax.set_xlabel(
        "Predicted class index"
    )

    ax.set_ylabel(
        "True class index"
    )

    ax.set_title(
        "Normalized CIFAR-100 test confusion matrix"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "test_confusion_matrix_normalized.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    save_json(
        [
            {
                "class_index": index,
                "class_name": name,
            }
            for index, name
            in enumerate(class_names)
        ],
        output_dir
        / "cifar100_class_index_map.json",
    )


def main():
    set_seed(SEED)

    print("=" * 78)
    print("FINAL LOCKED CIFAR-100 TEST EVALUATION")
    print("=" * 78)
    print(
        "Original checkpoints were lost; using the "
        "completed frozen-settings reconstruction runs."
    )
    print("No training will be performed.")
    print("No controller will be updated.")
    print(
        "All checkpoints must match their NEW "
        "validation-selected best epochs."
    )
    print(
        "Do not retune E1-E4 after inspecting "
        "these test results."
    )
    print("=" * 78)

    # Critical preflight: validate EVERY run before test-loader creation.
    resolved = []

    print(
        "\nPreflight verification of reconstructed runs:"
    )

    for run in RUNS:
        verified = resolve_frozen_run(
            run
        )

        resolved.append(
            verified
        )

        print(
            f"  {verified['id']}: "
            f"epoch {verified['best_epoch']} | "
            f"{verified['checkpoint_path']}"
        )

    print(
        "\nAll four frozen reruns verified."
    )
    print(
        "Creating the official CIFAR-100 test loader now."
    )

    # Official test set is opened only after all preflight checks pass.
    data = CIFAR100DataModule()

    class_names = tuple(
        data.test_dataset.classes
    )

    if len(class_names) != 100:
        raise ValueError(
            f"Expected 100 classes, found "
            f"{len(class_names)}."
        )

    test_loader = data.get_test_loader(
        batch_size=TEST_BATCH_SIZE
    )

    if len(test_loader.dataset) != 10000:
        raise ValueError(
            "Expected official 10,000-image "
            "CIFAR-100 test set."
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
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

    rows = []

    combined = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "CIFAR-100",
        "split": "official test",
        "num_test_samples": len(
            test_loader.dataset
        ),
        "num_classes": 100,
        "seed": SEED,
        "checkpoint_source": (
            "frozen-settings reconstruction runs "
            "performed because original checkpoint "
            "files were lost before test evaluation"
        ),
        "selection_policy": (
            "Each reconstruction checkpoint was selected "
            "using validation loss before the official "
            "CIFAR-100 test set was evaluated."
        ),
        "experiments": {},
    }

    for run in resolved:
        print("\n" + "-" * 78)
        print(
            f"{run['id']} | {run['label']}"
        )
        print("-" * 78)

        model = CustomCNN(
            num_classes=100
        ).to(DEVICE)

        checkpoint = load_checkpoint(
            model,
            run,
        )

        metrics = evaluate_model(
            model,
            test_loader,
            DEVICE,
            class_names=class_names,
        )

        exp_dir = (
            output_dir / run["id"]
        )

        exp_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        metrics_json = {
            key: value
            for key, value
            in metrics.items()
            if key not in (
                "y_true",
                "y_pred",
            )
        }

        detail = {
            "experiment": run["id"],
            "label": run["label"],
            "run_name": run["run_name"],
            "checkpoint_epoch": int(
                checkpoint["epoch"]
            ),
            "checkpoint_path": str(
                run["checkpoint_path"]
            ),
            "checkpoint_sha256": (
                run["checkpoint_sha256"]
            ),
            "validation_accuracy": float(
                run["validation"]["accuracy"]
            ),
            "validation_macro_f1": float(
                run["validation"]["macro_f1"]
            ),
            "optimizer_updates": int(
                run["metadata"][
                    "total_optimizer_updates"
                ]
            ),
            "training_time_seconds": float(
                run["metadata"][
                    "total_training_time_seconds"
                ]
            ),
            "test_metrics": metrics_json,
        }

        save_json(
            detail,
            exp_dir / "test_metrics.json",
        )

        save_per_class_csv(
            metrics["per_class"],
            exp_dir
            / "test_per_class_metrics.csv",
        )

        save_confusion_artifacts(
            metrics["y_true"],
            metrics["y_pred"],
            class_names,
            exp_dir,
        )

        row = {
            "experiment": run["id"],
            "label": run["label"],
            "checkpoint_epoch": int(
                checkpoint["epoch"]
            ),
            "validation_accuracy": float(
                run["validation"]["accuracy"]
            ),
            "test_accuracy": float(
                metrics["accuracy"]
            ),
            "validation_macro_f1": float(
                run["validation"]["macro_f1"]
            ),
            "test_macro_f1": float(
                metrics["macro_f1"]
            ),
            "test_loss": float(
                metrics["loss"]
            ),
            "macro_precision": float(
                metrics["macro_precision"]
            ),
            "macro_recall": float(
                metrics["macro_recall"]
            ),
            "num_test_samples": int(
                metrics["num_samples"]
            ),
            "optimizer_updates": int(
                run["metadata"][
                    "total_optimizer_updates"
                ]
            ),
            "training_time_seconds": float(
                run["metadata"][
                    "total_training_time_seconds"
                ]
            ),
            "checkpoint_sha256": (
                run["checkpoint_sha256"]
            ),
        }

        rows.append(
            row
        )

        combined["experiments"][
            run["id"]
        ] = detail

        print(
            f"Checkpoint epoch: {checkpoint['epoch']}"
        )
        print(
            f"Test loss:        {metrics['loss']:.4f}"
        )
        print(
            f"Test accuracy:    {metrics['accuracy']:.2%}"
        )
        print(
            f"Macro precision:  "
            f"{metrics['macro_precision']:.4f}"
        )
        print(
            f"Macro recall:     "
            f"{metrics['macro_recall']:.4f}"
        )
        print(
            f"Macro F1:         "
            f"{metrics['macro_f1']:.4f}"
        )

    fields = [
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
        output_dir
        / "cifar100_final_test_comparison.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    save_json(
        combined,
        output_dir
        / "cifar100_final_test_results.json",
    )

    print("\n" + "=" * 78)
    print("FINAL CIFAR-100 TEST COMPARISON")
    print("=" * 78)

    for row in rows:
        gap = 100.0 * (
            row["validation_accuracy"]
            - row["test_accuracy"]
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
    print(
        f"Final outputs saved to: {output_dir}"
    )
    print(
        "The CIFAR-100 test set is now opened for "
        "final reporting. Do not retune E1-E4."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
