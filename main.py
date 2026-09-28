import platform
import sys
from datetime import datetime, timezone

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
    TRAIN_SIZE,
    VAL_SIZE,
    WEIGHT_DECAY,
    get_device_name,
    print_config,
    set_seed,
)

from data.cifar10 import CIFAR10DataModule
from models.custom_cnn import CustomCNN
from training.trainer import Trainer
from evaluation.eval_pipeline import run_evaluation

def main():
    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    set_seed(SEED)

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    print_config()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    data = CIFAR10DataModule()

    train_loader = data.get_train_loader(
        INITIAL_BATCH_SIZE
    )

    val_loader = data.get_val_loader(
        batch_size=256
    )

    # The test set is intentionally not used during
    # development or hyperparameter selection.

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = CustomCNN().to(
        DEVICE
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    if OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=INITIAL_LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {OPTIMIZER}"
        )

    # -----------------------------------------------------
    # Run name
    # -----------------------------------------------------

    run_name = (
        f"fixed_fixed"
        f"_batch{INITIAL_BATCH_SIZE}"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )
    
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = RESULTS_DIR / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Results directory: {run_dir}")

    # -----------------------------------------------------
    # Run metadata
    # -----------------------------------------------------

    cuda_runtime = (
        torch.version.cuda
        if torch.cuda.is_available()
        else None
    )

    run_metadata = {
        "run_name": run_name,
        "experiment_type": "fixed_batch_fixed_lr",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": "CustomCNN",
        "dataset": "CIFAR-10",
        "training_samples": TRAIN_SIZE,
        "validation_samples": VAL_SIZE,
        "test_set_used_during_training": False,

        "seed": SEED,

        "epochs": EPOCHS,
        "batch_size": INITIAL_BATCH_SIZE,
        "learning_rate": INITIAL_LEARNING_RATE,

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

    # -----------------------------------------------------
    # Fixed baseline pilot
    # -----------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        batch_size=INITIAL_BATCH_SIZE,
    )

    evaluation_result = run_evaluation(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        history=history,
        total_training_seconds=trainer.total_training_seconds,
        checkpoint_path=run_dir / f"{run_name}_best.pt",
        output_dir=run_dir,
        run_name=run_name,
        split="validation",
        target_accuracy=0.80,
    )


if __name__ == "__main__":
    main()