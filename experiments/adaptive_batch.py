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
    CHECKPOINT_DIR,
    DEVICE,
    EPOCHS,
    INITIAL_BATCH_SIZE,
    INITIAL_LEARNING_RATE,
    MOMENTUM,
    NORMALIZATION,
    OPTIMIZER,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    get_device_name,
    set_seed,
)

from experiments.datasets import add_dataset_argument, dataset_settings, dataset_run_name
from evaluation.eval_pipeline import run_evaluation
from models.factory import add_model_argument, create_model, model_run_name
from training.batch_controller import AdaptiveBatchController
from training.trainer import Trainer


# ---------------------------------------------------------------------
# Frozen E2 adaptive-batch settings
# ---------------------------------------------------------------------

ADAPTIVE_BATCH_OPTIONS = [
    32,
    64,
    128,
    256
]

CV_WINDOW = 3
STABILITY_THRESHOLD = 0.25
PLATEAU_PATIENCE = 3
MIN_DELTA = 0.01
COOLDOWN_EPOCHS = 2


def build_optimizer(model):
    """
    Create the fixed-learning-rate optimizer used by E2.
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


def build_run_name(
    epochs,
    seed,
    pilot,
):
    """
    Keep pilot outputs separate from the primary E2 run.
    """

    run_name = (
        f"adaptive_batch_fixed"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{seed}"
    )

    if pilot:
        run_name += (
            f"_pilot{epochs}"
        )

    return run_name


def run_experiment(
    epochs,
    seed,
    overwrite=False,
    pilot=False,
    dataset="cifar10",
    model_name="custom_cnn",
    enforce_frozen=True,
):
    """
    Run E2: adaptive batch size + fixed learning rate.
    """

    if enforce_frozen and INITIAL_BATCH_SIZE != 32:
        raise ValueError(
            "E2 is frozen to initial batch size 32. "
            f"Current INITIAL_BATCH_SIZE is {INITIAL_BATCH_SIZE}."
        )

    if (
        not pilot
        and epochs != EPOCHS
    ):
        raise ValueError(
            "The primary E2 run must use the configured "
            f"{EPOCHS}-epoch budget. "
            "Use --pilot for a non-primary short run."
        )

    run_name = build_run_name(
        epochs=epochs,
        seed=seed,
        pilot=pilot,
    )

    run_name = dataset_run_name(dataset, run_name)
    run_name = model_run_name(model_name, run_name)
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(RESULTS_DIR) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results directory: {run_dir}")


    print(
        "\n"
        + "#"
        * 78
    )

    print(
        f"Experiment: {run_name}"
    )

    print(
        "#"
        * 78
    )

    set_seed(seed)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    data_module, num_classes, dataset_label = dataset_settings(dataset)
    data = data_module()
    data.train_generator.manual_seed(seed)
    class_names = list(data.val_dataset.dataset.classes)
    if len(class_names) != num_classes:
        raise ValueError("Dataset class names do not match model output size")
    train_size = len(data.train_dataset)
    val_size = len(data.val_dataset)

    train_loader = (
        data.get_train_loader(
            batch_size=INITIAL_BATCH_SIZE
        )
    )

    val_loader = (
        data.get_val_loader(
            batch_size=256
        )
    )

    # The test loader is intentionally NOT created here.
    # Test evaluation remains separate from controller
    # development and model selection.

    # -----------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------

    model = create_model(model_name, num_classes=num_classes).to(
        DEVICE
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        build_optimizer(
            model
        )
    )

    # -----------------------------------------------------
    # Adaptive batch controller
    # -----------------------------------------------------

    controller = AdaptiveBatchController(
        batch_sizes=ADAPTIVE_BATCH_OPTIONS,
        initial_batch_size=INITIAL_BATCH_SIZE,
        cv_window=CV_WINDOW,
        stability_threshold=STABILITY_THRESHOLD,
        plateau_patience=PLATEAU_PATIENCE,
        min_delta=MIN_DELTA,
        cooldown_epochs=COOLDOWN_EPOCHS,
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    cuda_runtime = (
        torch.version.cuda
        if torch.cuda.is_available()
        else None
    )

    run_metadata = {
        "run_name": run_name,
        "run_id": run_id,
        "output_dir": str(run_dir),

        "experiment_type": (
            "adaptive_batch_fixed_lr"
        ),

        "primary_experiment": (
            not pilot
        ),

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model": type(model).__name__, "model_name": model_name,
        "model_channels": getattr(model, "channels", None),
        "conv_layers": sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()),
        "dropout_p": model.dropout.p if hasattr(model, "dropout") else 0.0,

        "dataset": dataset_label,
        "num_classes": num_classes,
        "class_names": class_names,

        "training_samples": train_size,

        "validation_samples": val_size,

        "test_set_used_during_training": False,

        "seed": seed,

        "epochs": epochs,

        "initial_batch_size": (
            INITIAL_BATCH_SIZE
        ),

        "allowed_batch_sizes": (
            ADAPTIVE_BATCH_OPTIONS
        ),

        "batch_growth_policy": (
            "grow_only"
        ),

        "learning_rate": (
            INITIAL_LEARNING_RATE
        ),

        "learning_rate_policy": (
            "fixed"
        ),

        "optimizer": OPTIMIZER,

        "momentum": MOMENTUM,

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "normalization": (
            NORMALIZATION
        ),

        "gradient_stability_measure": (
            "3-epoch rolling mean of within-epoch "
            "gradient-norm coefficient of variation"
        ),

        "cv_window": (
            CV_WINDOW
        ),

        "stability_threshold": (
            STABILITY_THRESHOLD
        ),

        "plateau_patience": (
            PLATEAU_PATIENCE
        ),

        "min_delta": (
            MIN_DELTA
        ),

        "cooldown_epochs": (
            COOLDOWN_EPOCHS
        ),

        "device_type": (
            DEVICE.type
        ),

        "device_name": (
            get_device_name()
        ),

        "python_version": (
            sys.version
        ),

        "platform": (
            platform.platform()
        ),

        "pytorch_version": (
            torch.__version__
        ),

        "cuda_runtime": (
            cuda_runtime
        ),
    }

    # -----------------------------------------------------
    # Print experiment settings
    # -----------------------------------------------------

    print(f"Dataset:             {dataset_label}; classes: {num_classes}")

    print(
        f"Device:              "
        f"{get_device_name()}"
    )

    print(
        f"Training samples:    "
        f"{train_size:,}"
    )

    print(
        f"Validation samples:  "
        f"{val_size:,}"
    )

    print(
        f"Epochs:              "
        f"{epochs}"
    )

    print(
        f"Initial batch size:  "
        f"{INITIAL_BATCH_SIZE}"
    )

    print(
        f"Allowed batches:     "
        f"{ADAPTIVE_BATCH_OPTIONS}"
    )

    print(
        f"Fixed learning rate: "
        f"{INITIAL_LEARNING_RATE}"
    )

    print(
        f"CV window:           "
        f"{CV_WINDOW}"
    )

    print(
        f"Stability threshold: "
        f"{STABILITY_THRESHOLD}"
    )

    print(
        f"Plateau patience:    "
        f"{PLATEAU_PATIENCE}"
    )

    print(
        f"Minimum delta:       "
        f"{MIN_DELTA}"
    )

    print(
        f"Cooldown epochs:     "
        f"{COOLDOWN_EPOCHS}"
    )

    print(
        f"Pilot run:           "
        f"{pilot}"
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        results_dir=run_dir,
        checkpoint_dir=run_dir,
        run_name=run_name,
        run_metadata=run_metadata,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        batch_size=INITIAL_BATCH_SIZE,
        batch_controller=controller,
        train_loader_factory=(
            data.get_train_loader
        ),
    )

    # Evaluate the checkpoint selected by validation loss using the shared pipeline.
    evaluation_dir = run_dir
    completion_metadata = {
        "status": "completed",
        "best_epoch": trainer.best_epoch,
        "best_validation_loss": trainer.best_val_loss,
        "total_optimizer_updates": trainer.optimizer_updates,
        "total_training_time_seconds": trainer.total_training_seconds,
        "final_batch_size_used": history[-1]["batch_size"],
        "final_batch_size_selected": controller.current_batch_size,
        "evaluation_output_dir": str(evaluation_dir),
    }
    trainer.save_metadata({**completion_metadata, "evaluation_status": "started"})
    result = run_evaluation(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        history=history,
        total_training_seconds=trainer.total_training_seconds,
        checkpoint_path=run_dir / f"{run_name}_best.pt",
        output_dir=evaluation_dir,
        run_name=run_name,
        split="validation",
        class_names=class_names,
    )
    trainer.save_metadata({**completion_metadata, "evaluation_status": "completed"})

    # -----------------------------------------------------
    # Release memory
    # -----------------------------------------------------

    del trainer
    del controller
    del optimizer
    del criterion
    del model
    del train_loader
    del val_loader
    del data

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"output_dir": evaluation_dir, "history": history, "evaluation": result}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run E2: adaptive batch size with "
            "fixed learning rate on CIFAR-10/CIFAR-100."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of epochs. The primary E2 run must "
            f"use {EPOCHS}; use --pilot for shorter runs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Mark this as a development/pilot run so its "
            "outputs cannot be confused with primary E2."
        ),
    )

    parser.add_argument(
        "--overwrite", action="store_true",
        help="Deprecated compatibility flag; every run creates a new timestamp directory.",
    )

    add_dataset_argument(parser)
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = (
        parse_arguments()
    )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    print(
        "="
        * 78
    )

    print(
        "E2: Adaptive Batch Size + Fixed Learning Rate"
    )

    print(
        "="
        * 78
    )

    run_experiment(
        epochs=args.epochs,
        seed=args.seed,
        overwrite=args.overwrite,
        pilot=args.pilot,
        dataset=args.dataset,
        model_name=args.model,
    )

    print(
        "\n"
        + "="
        * 78
    )

    print(
        "Requested E2 experiment finished."
    )

    print(
        "="
        * 78
    )


if __name__ == "__main__":
    main()
