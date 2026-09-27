import argparse
import gc
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE_OPTIONS,
    CHECKPOINT_DIR,
    DEVICE,
    EPOCHS,
    INITIAL_LEARNING_RATE,
    MOMENTUM,
    NORMALIZATION,
    OPTIMIZER,
    RESULTS_DIR,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE,
    WEIGHT_DECAY,
    get_device_name,
    set_seed,
)

from data.cifar10 import CIFAR10DataModule
from models.custom_cnn import CustomCNN
from training.trainer import Trainer


def completed_run_exists(run_name):
    """
    Return True only when the metadata file says that the
    experiment completed successfully.

    This prevents us from accidentally repeating a finished
    experiment while still allowing interrupted runs to restart.
    """

    metadata_path = (
        Path(RESULTS_DIR)
        / f"{run_name}_metadata.json"
    )

    if not metadata_path.exists():
        return False

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        return (
            metadata.get("status")
            == "completed"
        )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return False


def build_optimizer(model):
    """
    Create the optimizer used by the fixed-batch experiments.
    """

    if OPTIMIZER.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=INITIAL_LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(
        f"Unsupported optimizer: {OPTIMIZER}"
    )


def run_fixed_experiment(
    batch_size,
    epochs,
    seed,
    overwrite=False,
):
    """
    Run one fixed-batch, fixed-learning-rate experiment.
    """

    run_name = (
        f"fixed_fixed"
        f"_batch{batch_size}"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{seed}"
    )

    print("\n" + "#" * 78)
    print(f"Experiment: {run_name}")
    print("#" * 78)

    # -----------------------------------------------------
    # Skip experiments that already finished
    # -----------------------------------------------------

    if (
        completed_run_exists(run_name)
        and not overwrite
    ):
        print(
            "Completed results already exist."
        )

        print(
            "Skipping this experiment."
        )

        print(
            "Use --overwrite if you intentionally "
            "want to run it again."
        )

        return

    # -----------------------------------------------------
    # Reset random state before EVERY experiment
    # -----------------------------------------------------

    set_seed(seed)

    # -----------------------------------------------------
    # Fresh dataset/data loaders
    # -----------------------------------------------------
    #
    # Recreating the data module ensures that the training
    # generator starts from the same seed for each batch-size
    # experiment.
    # -----------------------------------------------------

    data = CIFAR10DataModule()

    train_loader = data.get_train_loader(
        batch_size=batch_size
    )

    val_loader = data.get_val_loader(
        batch_size=256
    )

    # The test loader is intentionally NOT created here.
    # The test set stays untouched until experimental
    # settings have been finalized.

    # -----------------------------------------------------
    # Fresh model
    # -----------------------------------------------------

    model = CustomCNN().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model
    )

    # -----------------------------------------------------
    # Run metadata
    # -----------------------------------------------------

    cuda_runtime = (
        torch.version.cuda
        if torch.cuda.is_available()
        else None
    )

    expected_updates_per_epoch = (
        len(train_loader)
    )

    expected_total_updates = (
        expected_updates_per_epoch
        * epochs
    )

    run_metadata = {
        "run_name": run_name,
        "experiment_type": (
            "fixed_batch_fixed_lr"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": "CustomCNN",
        "dataset": "CIFAR-10",

        "training_samples": TRAIN_SIZE,
        "validation_samples": VAL_SIZE,

        "test_set_used_during_training": False,

        "seed": seed,

        "epochs": epochs,

        "batch_size": batch_size,

        "learning_rate": (
            INITIAL_LEARNING_RATE
        ),

        "optimizer": OPTIMIZER,

        "momentum": MOMENTUM,

        "weight_decay": WEIGHT_DECAY,

        "normalization": NORMALIZATION,

        "updates_per_epoch": (
            expected_updates_per_epoch
        ),

        "expected_total_updates": (
            expected_total_updates
        ),

        "device_type": DEVICE.type,

        "device_name": get_device_name(),

        "python_version": sys.version,

        "platform": platform.platform(),

        "pytorch_version": (
            torch.__version__
        ),

        "cuda_runtime": cuda_runtime,
    }

    # -----------------------------------------------------
    # Print experiment settings
    # -----------------------------------------------------

    print(
        f"Device:              "
        f"{get_device_name()}"
    )

    print(
        f"Training samples:    "
        f"{TRAIN_SIZE:,}"
    )

    print(
        f"Validation samples:  "
        f"{VAL_SIZE:,}"
    )

    print(
        f"Batch size:          "
        f"{batch_size}"
    )

    print(
        f"Learning rate:       "
        f"{INITIAL_LEARNING_RATE}"
    )

    print(
        f"Epochs:              "
        f"{epochs}"
    )

    print(
        f"Updates per epoch:   "
        f"{expected_updates_per_epoch:,}"
    )

    print(
        f"Expected updates:    "
        f"{expected_total_updates:,}"
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        results_dir=RESULTS_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        run_name=run_name,
        run_metadata=run_metadata,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        batch_size=batch_size,
    )

    # -----------------------------------------------------
    # Release memory before next experiment
    # -----------------------------------------------------

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


def parse_arguments():
    """
    Command-line options for the fixed-batch experiment.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-batch, fixed-learning-rate "
            "CIFAR-10 experiments."
        )
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=BATCH_SIZE_OPTIONS,
        help=(
            "Batch sizes to test. "
            "Example: --batch-sizes 16 64 128"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of epochs for each experiment."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Repeat experiments even when completed "
            "results already exist."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # -----------------------------------------------------
    # Validate batch sizes
    # -----------------------------------------------------

    invalid_batch_sizes = [
        batch_size
        for batch_size in args.batch_sizes
        if batch_size not in BATCH_SIZE_OPTIONS
    ]

    if invalid_batch_sizes:
        raise ValueError(
            "Unsupported batch size(s): "
            f"{invalid_batch_sizes}. "
            "Allowed values are "
            f"{BATCH_SIZE_OPTIONS}."
        )

    print("=" * 78)
    print("Fixed Batch + Fixed Learning Rate Experiments")
    print("=" * 78)

    print(
        f"Batch sizes: "
        f"{args.batch_sizes}"
    )

    print(
        f"Epochs:      "
        f"{args.epochs}"
    )

    print(
        f"Seed:        "
        f"{args.seed}"
    )

    print(
        f"Fixed LR:    "
        f"{INITIAL_LEARNING_RATE}"
    )

    print("=" * 78)

    for batch_size in args.batch_sizes:
        run_fixed_experiment(
            batch_size=batch_size,
            epochs=args.epochs,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    print("\n" + "=" * 78)
    print("Requested fixed-batch experiments finished.")
    print("=" * 78)


if __name__ == "__main__":
    main()