"""E4: adaptive batch size + adaptive learning-rate experiment.

This runner combines the frozen E2 batch controller with the frozen E3
learning-rate controller. The validation set drives both controllers.
The CIFAR-10 test set is not used during training or model selection.

Primary run:
    python -m experiments.adaptive_combined

Short development run:
    python -m experiments.adaptive_combined --epochs 3 --pilot
"""

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    EPOCHS,
    INITIAL_BATCH_SIZE,
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
from evaluation.eval_pipeline import run_evaluation
from models.custom_cnn import CustomCNN
from training.batch_controller import AdaptiveBatchController
from training.lr_controller import LRController
from training.trainer import Trainer


# ---------------------------------------------------------------------
# Frozen E4 settings
# ---------------------------------------------------------------------
# These are copied directly from the already-frozen E2 and E3 policies.
# Do not retune them based on E4 results.

# E2 batch controller
ADAPTIVE_BATCH_OPTIONS = [32, 64, 128]
BATCH_CV_WINDOW = 3
BATCH_STABILITY_THRESHOLD = 0.25
BATCH_PLATEAU_PATIENCE = 3
BATCH_MIN_DELTA = 0.01
BATCH_COOLDOWN_EPOCHS = 2

# E3 LR controller
LR_REFERENCE_BATCH_SIZE = 32
LR_REFERENCE_LR = 0.01
LR_ALPHA = 0.2
LR_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 5
LR_WORSENING_PATIENCE = 3
LR_COOLDOWN_EPOCHS = 2
LR_WARMUP_EPOCHS = 5
LR_MIN = 1e-5
LR_MIN_DELTA_LOSS = 1e-3
LR_MIN_DELTA_ACC = 0.002

TARGET_ACCURACY = 0.80


def build_optimizer(model):
    """Create the SGD optimizer shared by E1-E4."""
    if OPTIMIZER.lower() != "sgd":
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

    return optim.SGD(
        model.parameters(),
        lr=LR_REFERENCE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


def validate_frozen_settings():
    """Fail closed if shared project defaults drift from the E4 design."""
    if INITIAL_BATCH_SIZE != 32:
        raise ValueError(
            "E4 is frozen to initial batch size 32. "
            f"Current INITIAL_BATCH_SIZE is {INITIAL_BATCH_SIZE}."
        )

    if not math.isclose(
        INITIAL_LEARNING_RATE,
        LR_REFERENCE_LR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "E4 reference LR must match the shared baseline LR. "
            f"Config has {INITIAL_LEARNING_RATE}; "
            f"E4 expects {LR_REFERENCE_LR}."
        )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run E4: adaptive batch size + adaptive learning rate "
            "on CIFAR-10."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of epochs. The primary E4 run must use "
            f"{EPOCHS}; use --pilot for shorter development runs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Model/shuffle random seed.",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Mark this as a development run so it cannot be confused "
            "with the primary E4 experiment."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Root directory for E4 outputs.",
    )

    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")

    if not args.pilot and args.epochs != EPOCHS:
        parser.error(
            f"Primary E4 must use {EPOCHS} epochs. "
            "Use --pilot for a shorter development run."
        )

    return args


def run_experiment(args):
    """Run E4 and evaluate the best validation checkpoint."""
    validate_frozen_settings()
    set_seed(args.seed)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    data = CIFAR10DataModule()

    # The split itself remains fixed by config.SEED. This seed controls
    # model initialization and the persistent training-shuffle generator.
    data.train_generator.manual_seed(args.seed)

    train_loader = data.get_train_loader(
        batch_size=INITIAL_BATCH_SIZE
    )
    val_loader = data.get_val_loader(
        batch_size=256
    )

    # Intentionally do not create the test loader here.
    # Test evaluation stays separate until final model selection is locked.

    # -----------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------
    model = CustomCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    # -----------------------------------------------------
    # Controllers
    # -----------------------------------------------------
    batch_controller = AdaptiveBatchController(
        batch_sizes=ADAPTIVE_BATCH_OPTIONS,
        initial_batch_size=INITIAL_BATCH_SIZE,
        cv_window=BATCH_CV_WINDOW,
        stability_threshold=BATCH_STABILITY_THRESHOLD,
        plateau_patience=BATCH_PLATEAU_PATIENCE,
        min_delta=BATCH_MIN_DELTA,
        cooldown_epochs=BATCH_COOLDOWN_EPOCHS,
    )

    lr_controller = LRController(
        optimizer,
        mode="adaptive",
        reference_batch_size=LR_REFERENCE_BATCH_SIZE,
        reference_lr=LR_REFERENCE_LR,
        alpha=LR_ALPHA,
        lr_factor=LR_FACTOR,
        plateau_patience=LR_PLATEAU_PATIENCE,
        worsening_patience=LR_WORSENING_PATIENCE,
        cooldown_epochs=LR_COOLDOWN_EPOCHS,
        warmup_epochs=LR_WARMUP_EPOCHS,
        min_lr=LR_MIN,
        min_delta_loss=LR_MIN_DELTA_LOSS,
        min_delta_acc=LR_MIN_DELTA_ACC,
    )

    # set_batch_size does not consume controller history. Trainer.fit()
    # applies the same initial setting again before epoch 1.
    initial_lr = lr_controller.set_batch_size(
        INITIAL_BATCH_SIZE
    )

    # -----------------------------------------------------
    # Output directory + metadata
    # -----------------------------------------------------
    primary = not args.pilot

    run_name = (
        "adaptive_batch_adaptive_lr"
        f"_refLR{LR_REFERENCE_LR}"
        f"_seed{args.seed}"
    )

    if args.pilot:
        run_name += f"_pilot{args.epochs}"

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(args.output_dir) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cuda_runtime = (
        torch.version.cuda
        if torch.cuda.is_available()
        else None
    )

    metadata = {
        "run_name": run_name,
        "run_id": run_id,
        "experiment_type": "adaptive_batch_adaptive_lr",
        "primary_experiment": primary,
        "created_utc": started_at.isoformat(),
        "output_dir": str(run_dir),
        "model": "CustomCNN",
        "dataset": "CIFAR-10",
        "training_samples": TRAIN_SIZE,
        "validation_samples": VAL_SIZE,
        "test_set_used_during_training": False,
        "seed": args.seed,
        "data_split_seed": SEED,
        "epochs": args.epochs,
        "initial_batch_size": INITIAL_BATCH_SIZE,
        "allowed_batch_sizes": ADAPTIVE_BATCH_OPTIONS,
        "batch_growth_policy": "grow_only",
        "gradient_stability_measure": (
            "3-epoch rolling mean of within-epoch "
            "gradient-norm coefficient of variation"
        ),
        "batch_controller": {
            "cv_window": BATCH_CV_WINDOW,
            "stability_threshold": BATCH_STABILITY_THRESHOLD,
            "plateau_patience": BATCH_PLATEAU_PATIENCE,
            "min_delta": BATCH_MIN_DELTA,
            "cooldown_epochs": BATCH_COOLDOWN_EPOCHS,
        },
        "lr_controller": {
            "reference_batch_size": LR_REFERENCE_BATCH_SIZE,
            "reference_lr": LR_REFERENCE_LR,
            "alpha": LR_ALPHA,
            "lr_factor": LR_FACTOR,
            "plateau_patience": LR_PLATEAU_PATIENCE,
            "worsening_patience": LR_WORSENING_PATIENCE,
            "cooldown_epochs": LR_COOLDOWN_EPOCHS,
            "warmup_epochs": LR_WARMUP_EPOCHS,
            "min_lr": LR_MIN,
            "min_delta_loss": LR_MIN_DELTA_LOSS,
            "min_delta_acc": LR_MIN_DELTA_ACC,
        },
        "initial_learning_rate": initial_lr,
        "target_accuracy": TARGET_ACCURACY,
        "optimizer": OPTIMIZER,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "normalization": NORMALIZATION,
        "device_type": DEVICE.type,
        "device_name": get_device_name(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": str(torch.__version__),
        "cuda_runtime": cuda_runtime,
    }

    print("=" * 78)
    print("E4: Adaptive Batch Size + Adaptive Learning Rate")
    print("=" * 78)
    print(f"Results directory:    {run_dir}")
    print(f"Device:               {get_device_name()}")
    print(f"Training samples:     {TRAIN_SIZE:,}")
    print(f"Validation samples:   {VAL_SIZE:,}")
    print(f"Epochs:               {args.epochs}")
    print(f"Initial batch:        {INITIAL_BATCH_SIZE}")
    print(f"Allowed batches:      {ADAPTIVE_BATCH_OPTIONS}")
    print(f"Initial LR:           {initial_lr:.8f}")
    print(f"Reference LR:         {LR_REFERENCE_LR}")
    print(f"LR alpha:             {LR_ALPHA}")
    print(f"LR factor:            {LR_FACTOR}")
    print(f"Warmup epochs:        {LR_WARMUP_EPOCHS}")
    print(f"Pilot run:            {args.pilot}")
    print("=" * 78)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        results_dir=run_dir,
        checkpoint_dir=run_dir,
        run_name=run_name,
        run_metadata=metadata,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        batch_size=INITIAL_BATCH_SIZE,
        batch_controller=batch_controller,
        train_loader_factory=data.get_train_loader,
        lr_controller=lr_controller,
    )

    # Preserve the complete LR-controller decision history.
    (run_dir / "lr_history.json").write_text(
        json.dumps(
            lr_controller.lr_history,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Evaluate best validation checkpoint only
    # -----------------------------------------------------
    trainer.save_metadata(
        {
            "status": "completed",
            "evaluation_status": "started",
            "best_epoch": trainer.best_epoch,
            "best_validation_loss": trainer.best_val_loss,
            "total_optimizer_updates": trainer.optimizer_updates,
            "total_training_time_seconds": trainer.total_training_seconds,
            "final_batch_size_used": history[-1]["batch_size"],
            "final_batch_size_selected": batch_controller.current_batch_size,
            "final_learning_rate_used": history[-1]["learning_rate"],
            "final_learning_rate_selected": optimizer.param_groups[0]["lr"],
        }
    )

    result = run_evaluation(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        history=history,
        total_training_seconds=trainer.total_training_seconds,
        checkpoint_path=run_dir / f"{run_name}_best.pt",
        output_dir=run_dir,
        run_name=run_name,
        split="validation",
        target_accuracy=TARGET_ACCURACY,
    )

    trainer.save_metadata(
        {
            "status": "completed",
            "evaluation_status": "completed",
            "best_epoch": trainer.best_epoch,
            "best_validation_loss": trainer.best_val_loss,
            "total_optimizer_updates": trainer.optimizer_updates,
            "total_training_time_seconds": trainer.total_training_seconds,
            "final_batch_size_used": history[-1]["batch_size"],
            "final_batch_size_selected": batch_controller.current_batch_size,
            "final_learning_rate_used": history[-1]["learning_rate"],
            "final_learning_rate_selected": optimizer.param_groups[0]["lr"],
        }
    )

    return {
        "output_dir": run_dir,
        "history": history,
        "evaluation": result,
    }


def main():
    run_experiment(parse_arguments())


if __name__ == "__main__":
    main()
