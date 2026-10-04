"""Rebuild the frozen CIFAR-100 E1-E4 primary runs.

Use this only because the original CIFAR-100 best-validation checkpoints were
lost. The controller settings remain frozen and the official CIFAR-100 test
set is not evaluated here.

Primary usage:
    python -m experiments.rebuild_cifar100_primary

Optional subset:
    python -m experiments.rebuild_cifar100_primary --runs E1 E2

This script creates fresh official CIFAR-100 rerun artifacts:
    results/cifar100/E1 ... E4
    checkpoints/cifar100/E1 ... E4

If the deterministic rerun differs numerically from the historical teammate
report, the NEW rerun becomes the official validation-stage record because
these are the checkpoints that will be used for final test evaluation.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.torch_version import TorchVersion

from config import (
    DEVICE,
    MOMENTUM,
    WEIGHT_DECAY,
    set_seed,
)
from data.cifar100 import CIFAR100DataModule
from evaluation.metrics import evaluate_model
from models.custom_cnn import CustomCNN
from training.batch_controller import AdaptiveBatchController
from training.lr_controller import LRController
from training.trainer import Trainer


SEED = 42
EPOCHS = 20
INITIAL_BATCH = 32
REFERENCE_LR = 0.01
VAL_BATCH_SIZE = 256

BATCH_OPTIONS = [32, 64, 128]
CV_WINDOW = 3
STABILITY_THRESHOLD = 0.25
BATCH_PLATEAU_PATIENCE = 3
BATCH_MIN_DELTA = 0.01
BATCH_COOLDOWN = 2

REFERENCE_BATCH = 32
LR_ALPHA = 0.5
LR_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 5
LR_WORSENING_PATIENCE = 3
LR_COOLDOWN = 2
LR_WARMUP = 0
MIN_LR = 1e-5
MIN_DELTA_LOSS = 1e-3
MIN_DELTA_ACC = 0.002

RESULTS_ROOT = Path("results/cifar100")
CHECKPOINT_ROOT = Path("checkpoints/cifar100")

RUN_NAMES = {
    "E1": "cifar100_E1_fixed_fixed_batch32_lr0.01_seed42",
    "E2": "cifar100_E2_adaptive_batch_fixed_lr0.01_seed42",
    "E3": "cifar100_E3_fixed_batch_adaptive_lr_seed42",
    "E4": "cifar100_E4_adaptive_batch_adaptive_lr_seed42",
}

LABELS = {
    "E1": "Fixed batch + fixed LR",
    "E2": "Adaptive batch + fixed LR",
    "E3": "Fixed batch + adaptive LR",
    "E4": "Adaptive batch + adaptive LR",
}


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_best_checkpoint(model, checkpoint_path):
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return checkpoint


def frozen_metadata(exp):
    base = {
        "dataset": "CIFAR-100",
        "experiment": exp,
        "num_classes": 100,
        "training_samples": 45000,
        "validation_samples": 5000,
        "test_set_used_during_training": False,
        "epochs": EPOCHS,
        "seed": SEED,
        "optimizer": "SGD",
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "rerun_reason": (
            "Original CIFAR-100 validation-selected checkpoint files were lost. "
            "This is a frozen-settings reconstruction run performed before "
            "final CIFAR-100 test evaluation."
        ),
    }

    if exp == "E1":
        base.update(
            {
                "batch_size": INITIAL_BATCH,
                "learning_rate": REFERENCE_LR,
                "batch_policy": "fixed",
                "learning_rate_policy": "fixed",
            }
        )

    elif exp == "E2":
        base.update(
            {
                "initial_batch_size": INITIAL_BATCH,
                "allowed_batch_sizes": BATCH_OPTIONS,
                "batch_growth_policy": "grow_only",
                "learning_rate": REFERENCE_LR,
                "learning_rate_policy": "fixed",
                "cv_window": CV_WINDOW,
                "stability_threshold": STABILITY_THRESHOLD,
                "plateau_patience": BATCH_PLATEAU_PATIENCE,
                "min_delta": BATCH_MIN_DELTA,
                "cooldown_epochs": BATCH_COOLDOWN,
            }
        )

    elif exp == "E3":
        base.update(
            {
                "batch_size": INITIAL_BATCH,
                "batch_policy": "fixed",
                "reference_learning_rate": REFERENCE_LR,
                "reference_batch_size": REFERENCE_BATCH,
                "alpha": LR_ALPHA,
                "lr_factor": LR_FACTOR,
                "plateau_patience": LR_PLATEAU_PATIENCE,
                "worsening_patience": LR_WORSENING_PATIENCE,
                "cooldown_epochs": LR_COOLDOWN,
                "warmup_epochs": LR_WARMUP,
                "min_lr": MIN_LR,
                "min_delta_loss": MIN_DELTA_LOSS,
                "min_delta_acc": MIN_DELTA_ACC,
            }
        )

    elif exp == "E4":
        base.update(
            {
                "initial_batch_size": INITIAL_BATCH,
                "allowed_batch_sizes": BATCH_OPTIONS,
                "batch_growth_policy": "grow_only",
                "batch_controller": {
                    "cv_window": CV_WINDOW,
                    "stability_threshold": STABILITY_THRESHOLD,
                    "plateau_patience": BATCH_PLATEAU_PATIENCE,
                    "min_delta": BATCH_MIN_DELTA,
                    "cooldown_epochs": BATCH_COOLDOWN,
                },
                "reference_learning_rate": REFERENCE_LR,
                "reference_batch_size": REFERENCE_BATCH,
                "lr_controller": {
                    "alpha": LR_ALPHA,
                    "lr_factor": LR_FACTOR,
                    "plateau_patience": LR_PLATEAU_PATIENCE,
                    "worsening_patience": LR_WORSENING_PATIENCE,
                    "cooldown_epochs": LR_COOLDOWN,
                    "warmup_epochs": LR_WARMUP,
                    "min_lr": MIN_LR,
                    "min_delta_loss": MIN_DELTA_LOSS,
                    "min_delta_acc": MIN_DELTA_ACC,
                },
            }
        )

    else:
        raise ValueError(f"Unknown experiment: {exp}")

    return base


def make_lr_controller(optimizer):
    return LRController(
        optimizer,
        mode="adaptive",
        reference_batch_size=REFERENCE_BATCH,
        reference_lr=REFERENCE_LR,
        alpha=LR_ALPHA,
        lr_factor=LR_FACTOR,
        plateau_patience=LR_PLATEAU_PATIENCE,
        worsening_patience=LR_WORSENING_PATIENCE,
        cooldown_epochs=LR_COOLDOWN,
        warmup_epochs=LR_WARMUP,
        min_lr=MIN_LR,
        min_delta_loss=MIN_DELTA_LOSS,
        min_delta_acc=MIN_DELTA_ACC,
    )


def make_batch_controller():
    return AdaptiveBatchController(
        batch_sizes=BATCH_OPTIONS,
        initial_batch_size=INITIAL_BATCH,
        cv_window=CV_WINDOW,
        stability_threshold=STABILITY_THRESHOLD,
        plateau_patience=BATCH_PLATEAU_PATIENCE,
        min_delta=BATCH_MIN_DELTA,
        cooldown_epochs=BATCH_COOLDOWN,
    )


def run_one(exp, overwrite=False):
    run_name = RUN_NAMES[exp]
    results_dir = RESULTS_ROOT / exp
    checkpoint_dir = CHECKPOINT_ROOT / exp
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"

    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"{checkpoint_path} already exists. Refusing to overwrite a "
            "completed official rerun. Use --overwrite only if you "
            "intentionally need to repeat it before test evaluation."
        )

    set_seed(SEED)

    data = CIFAR100DataModule()
    data.train_generator.manual_seed(SEED)

    train_loader = data.get_train_loader(
        batch_size=INITIAL_BATCH
    )
    val_loader = data.get_val_loader(
        batch_size=VAL_BATCH_SIZE
    )

    class_names = tuple(
        data.val_dataset.dataset.classes
    )

    if len(class_names) != 100:
        raise ValueError(
            f"Expected 100 CIFAR-100 classes, found {len(class_names)}."
        )

    model = CustomCNN(
        num_classes=100
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=REFERENCE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    batch_controller = None
    lr_controller = None

    if exp in ("E2", "E4"):
        batch_controller = make_batch_controller()

    if exp in ("E3", "E4"):
        lr_controller = make_lr_controller(
            optimizer
        )
        lr_controller.set_batch_size(
            INITIAL_BATCH
        )

    print("\n" + "#" * 78)
    print(f"{exp} | {LABELS[exp]}")
    print("#" * 78)
    print("Dataset:             CIFAR-100")
    print("Training samples:    45,000")
    print("Validation samples:  5,000")
    print("Epochs:              20")
    print("Seed:                42")
    print("Initial batch:       32")
    print("Reference LR:        0.01")
    print("Test evaluation:     DISABLED")
    print("#" * 78)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        run_metadata=frozen_metadata(exp),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        batch_size=INITIAL_BATCH,
        batch_controller=batch_controller,
        train_loader_factory=(
            data.get_train_loader
            if batch_controller is not None
            else None
        ),
        lr_controller=lr_controller,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trainer finished but best checkpoint is missing: "
            f"{checkpoint_path}"
        )

    checkpoint = load_best_checkpoint(
        model,
        checkpoint_path,
    )

    if int(checkpoint["epoch"]) != int(trainer.best_epoch):
        raise ValueError(
            "Best checkpoint epoch does not match trainer.best_epoch."
        )

    metrics = evaluate_model(
        model,
        val_loader,
        DEVICE,
        class_names=class_names,
    )

    validation_record = {
        "split": "validation",
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(
            metrics["macro_precision"]
        ),
        "macro_recall": float(
            metrics["macro_recall"]
        ),
        "macro_f1": float(
            metrics["macro_f1"]
        ),
        "num_samples": int(
            metrics["num_samples"]
        ),
        "per_class": metrics["per_class"],
    }

    save_json(
        validation_record,
        results_dir / "validation_metrics.json",
    )

    if lr_controller is not None:
        save_json(
            lr_controller.lr_history,
            results_dir / "lr_history.json",
        )

    summary = {
        "experiment": exp,
        "label": LABELS[exp],
        "run_name": run_name,
        "best_epoch": int(
            trainer.best_epoch
        ),
        "best_validation_loss": float(
            trainer.best_val_loss
        ),
        "validation_accuracy": float(
            metrics["accuracy"]
        ),
        "validation_macro_f1": float(
            metrics["macro_f1"]
        ),
        "optimizer_updates": int(
            trainer.optimizer_updates
        ),
        "training_time_seconds": float(
            trainer.total_training_seconds
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    }

    print("\nOfficial rerun validation summary")
    print("-" * 78)
    print(
        f"Best epoch:          {summary['best_epoch']}"
    )
    print(
        f"Best val loss:       "
        f"{summary['best_validation_loss']:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{summary['validation_accuracy']:.2%}"
    )
    print(
        f"Validation macro F1: "
        f"{summary['validation_macro_f1']:.4f}"
    )
    print(
        f"Optimizer updates:   "
        f"{summary['optimizer_updates']:,}"
    )
    print(
        f"Training time:       "
        f"{summary['training_time_seconds']:.2f}s"
    )
    print(
        f"Checkpoint:          "
        f"{summary['checkpoint_path']}"
    )
    print("-" * 78)

    del trainer
    del optimizer
    del criterion
    del model
    del train_loader
    del val_loader
    del data

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def save_combined_summary(rows):
    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        RESULTS_ROOT
        / "cifar100_official_rerun_summary.json"
    )

    save_json(
        {
            "status": "completed",
            "purpose": (
                "Frozen-settings reconstruction because original "
                "CIFAR-100 checkpoint files were lost before test "
                "evaluation."
            ),
            "test_set_evaluated": False,
            "experiments": rows,
        },
        json_path,
    )

    csv_path = (
        RESULTS_ROOT
        / "cifar100_official_rerun_summary.csv"
    )

    fields = [
        "experiment",
        "label",
        "run_name",
        "best_epoch",
        "best_validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "optimizer_updates",
        "training_time_seconds",
        "checkpoint_path",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("CIFAR-100 OFFICIAL RERUN SUMMARY")
    print("=" * 78)

    for row in rows:
        print(
            f"{row['experiment']} | "
            f"epoch {row['best_epoch']} | "
            f"val acc {row['validation_accuracy']:.2%} | "
            f"macro F1 {row['validation_macro_f1']:.4f} | "
            f"updates {row['optimizer_updates']:,}"
        )

    print("=" * 78)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("The official CIFAR-100 test set was NOT evaluated.")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild lost CIFAR-100 best-validation checkpoints "
            "using the frozen E1-E4 settings."
        )
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        choices=("E1", "E2", "E3", "E4"),
        default=["E1", "E2", "E3", "E4"],
        help=(
            "Experiments to rebuild. Default: E1 E2 E3 E4."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing checkpoint. "
            "Do not use casually."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rows = []

    for exp in args.runs:
        rows.append(
            run_one(
                exp,
                overwrite=args.overwrite,
            )
        )

    save_combined_summary(rows)


if __name__ == "__main__":
    main()
