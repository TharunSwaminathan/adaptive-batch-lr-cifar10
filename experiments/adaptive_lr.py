"""Fixed batch size + adaptive LR experiment.

From the project root:
    python -m experiments.adaptive_lr --epochs 10 --batch-size 32
    python -m experiments.adaptive_lr --reference-lr 0.01 --alpha 0.5
    python -m experiments.adaptive_lr --dataset cifar100 --epochs 30

--lr is an alias for --reference-lr: it is the LR at --reference-batch-size,
not necessarily the actual starting LR. Initial LR includes batch scaling
and optional --warmup-epochs (default 0, disabled).
"""

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Also allow: python experiments/adaptive_lr.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import (
    DEVICE, EPOCHS, INITIAL_BATCH_SIZE, MOMENTUM, NORMALIZATION,
    OPTIMIZER, RESULTS_DIR, SEED, WEIGHT_DECAY, get_device_name, set_seed,
)
from data.cifar10 import CIFAR10DataModule
from data.cifar100 import CIFAR100DataModule
from evaluation.eval_pipeline import run_evaluation
from models.custom_cnn import CustomCNN
from training.lr_controller import LRController
from training.trainer import Trainer


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10",
                        help="Dataset to train and validate on (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=INITIAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reference-lr", "--lr", dest="reference_lr", type=float, default=0.01)
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--worsening-patience", type=int, default=3)
    parser.add_argument("--cooldown-epochs", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs; 0 disables warmup")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta-loss", type=float, default=1e-3)
    parser.add_argument("--min-delta-acc", type=float, default=0.002)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "reference_batch_size", "plateau_patience", "worsening_patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs must be non-negative")
    if args.cooldown_epochs < 0:
        parser.error("--cooldown-epochs must be non-negative")
    for name in ("reference_lr", "alpha", "lr_factor", "min_lr", "min_delta_loss", "min_delta_acc", "target_accuracy"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.reference_lr == 0 or not 0 < args.lr_factor < 1:
        parser.error("reference LR must be positive and LR factor must be in (0, 1)")
    if args.target_accuracy > 1 or args.min_delta_acc > 1:
        parser.error("accuracy values must use fractions in [0, 1]")
    return args


def run_adaptive_experiment(args):
    """Train a fresh model, then evaluate the best validation checkpoint."""
    if OPTIMIZER.lower() != "sgd":
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")
    set_seed(args.seed)
    dataset_name = getattr(args, "dataset", "cifar10")
    dataset_options = {
        "cifar10": (CIFAR10DataModule, 10, "CIFAR-10"),
        "cifar100": (CIFAR100DataModule, 100, "CIFAR-100"),
    }
    if dataset_name not in dataset_options:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    data_module, num_classes, dataset_label = dataset_options[dataset_name]
    model = CustomCNN(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.reference_lr,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    controller_settings = {
        name: getattr(args, name) for name in (
            "reference_lr", "reference_batch_size", "alpha", "lr_factor",
            "plateau_patience", "worsening_patience", "cooldown_epochs", "warmup_epochs",
            "min_lr", "min_delta_loss", "min_delta_acc",
        )
    }
    controller = LRController(optimizer, mode="adaptive", **controller_settings)
    initial_lr = controller.set_batch_size(args.batch_size)

    data = data_module()
    # The validation Subset wraps the dataset carrying its ordered class names.
    class_names = list(data.val_dataset.dataset.classes)
    if len(class_names) != num_classes:
        raise ValueError("Dataset class names do not match the model output size")
    # Keep the shared config.SEED data split; vary shuffle/model seed explicitly.
    data.train_generator.manual_seed(args.seed)
    train_loader = data.get_train_loader(args.batch_size)
    val_loader = data.get_val_loader(batch_size=256)

    run_name = f"adaptive_lr_batch{args.batch_size}_refLR{args.reference_lr}_seed{args.seed}"
    # Preserve existing CIFAR-10 paths used by the final-results report.
    if dataset_name != "cifar10":
        run_name = f"{dataset_name}_{run_name}"
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(args.output_dir) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_name": run_name, "run_id": run_id,
        "experiment_type": "fixed_batch_adaptive_lr",
        "created_utc": started_at.isoformat(), "output_dir": str(run_dir),
        "model": "CustomCNN", "dataset": dataset_label,
        "num_classes": num_classes, "class_names": class_names,
        "training_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "test_set_used_during_training": False,
        "seed": args.seed, "data_split_seed": SEED,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "initial_learning_rate": initial_lr, "lr_controller": controller_settings,
        "target_accuracy": args.target_accuracy,
        "optimizer": OPTIMIZER, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "normalization": NORMALIZATION, "device_type": DEVICE.type,
        "device_name": get_device_name(), "python_version": sys.version,
        "platform": platform.platform(), "pytorch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda if DEVICE.type == "cuda" else None,
    }
    print(f"Results directory: {run_dir}")
    print(f"Dataset: {dataset_label}; classes: {num_classes}")
    print(f"Fixed batch: {args.batch_size}; initial scaled LR: {initial_lr:.8f}; epochs: {args.epochs}")
    trainer = Trainer(
        model=model, criterion=torch.nn.CrossEntropyLoss(), optimizer=optimizer,
        device=DEVICE, results_dir=run_dir, checkpoint_dir=run_dir,
        run_name=run_name, run_metadata=metadata,
    )
    history = trainer.fit(
        train_loader=train_loader, val_loader=val_loader,
        epochs=args.epochs, batch_size=args.batch_size, lr_controller=controller,
    )
    (run_dir / "lr_history.json").write_text(
        json.dumps(controller.lr_history, indent=2, allow_nan=False), encoding="utf-8",
    )
    # Mark evaluation separately: Trainer's completed status only covers training.
    trainer.save_metadata({
        "status": "completed", "evaluation_status": "started",
        "best_epoch": trainer.best_epoch, "best_validation_loss": trainer.best_val_loss,
        "total_optimizer_updates": trainer.optimizer_updates,
        "total_training_time_seconds": trainer.total_training_seconds,
    })
    result = run_evaluation(
        model=model, data_loader=val_loader, device=DEVICE, history=history,
        total_training_seconds=trainer.total_training_seconds,
        checkpoint_path=run_dir / f"{run_name}_best.pt", output_dir=run_dir,
        run_name=run_name, split="validation", target_accuracy=args.target_accuracy,
        class_names=class_names,
    )
    trainer.save_metadata({
        "status": "completed", "evaluation_status": "completed",
        "best_epoch": trainer.best_epoch, "best_validation_loss": trainer.best_val_loss,
        "total_optimizer_updates": trainer.optimizer_updates,
        "total_training_time_seconds": trainer.total_training_seconds,
    })
    return {"output_dir": run_dir, "history": history, "evaluation": result}


def main():
    run_adaptive_experiment(parse_arguments())


if __name__ == "__main__":
    main()
