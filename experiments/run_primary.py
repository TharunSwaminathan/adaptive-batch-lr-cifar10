"""Run one frozen 40-epoch primary experiment on CIFAR-10 or CIFAR-100."""

import argparse
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    RESULTS_DIR,
    get_device_name,
    set_seed,
)
from evaluation.eval_pipeline import run_evaluation
from experiments.datasets import DATASETS, dataset_settings
from experiments.primary_protocol import PRIMARY_CONFIGS, validate_primary_protocol
from models.custom_cnn import CustomCNN
from training.batch_controller import AdaptiveBatchController
from training.lr_controller import LRController
from training.trainer import Trainer

TARGET_ACCURACY = {
    "cifar10": 0.80,
    "cifar100": 0.30,
}


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _build_optimizer(model, cfg):
    opt = cfg["optimizer"]
    if opt["name"].lower() != "sgd":
        raise ValueError("Primary protocol currently supports SGD only.")

    lr_cfg = cfg["learning_rate"]
    if lr_cfg["policy"] == "fixed":
        initial_lr = lr_cfg["value"]
    else:
        initial_lr = lr_cfg["reference_lr"]

    return torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=opt["momentum"],
        weight_decay=opt["weight_decay"],
    )


def _build_batch_controller(cfg):
    batch_cfg = cfg["batch"]
    if batch_cfg["policy"] == "fixed":
        return None
    return AdaptiveBatchController(
        batch_sizes=batch_cfg["allowed"],
        initial_batch_size=batch_cfg["initial"],
        cv_window=batch_cfg["cv_window"],
        stability_threshold=batch_cfg["stability_threshold"],
        plateau_patience=batch_cfg["plateau_patience"],
        min_delta=batch_cfg["min_delta"],
        cooldown_epochs=batch_cfg["cooldown_epochs"],
    )


def _build_lr_controller(optimizer, cfg):
    lr_cfg = cfg["learning_rate"]
    if lr_cfg["policy"] == "fixed":
        return None
    return LRController(
        optimizer,
        mode="adaptive",
        reference_batch_size=lr_cfg["reference_batch_size"],
        reference_lr=lr_cfg["reference_lr"],
        alpha=lr_cfg["alpha"],
        lr_factor=lr_cfg["factor"],
        plateau_patience=lr_cfg["plateau_patience"],
        worsening_patience=lr_cfg["worsening_patience"],
        cooldown_epochs=lr_cfg["cooldown_epochs"],
        warmup_epochs=lr_cfg["warmup_epochs"],
        min_lr=lr_cfg["min_lr"],
        min_delta_loss=lr_cfg["min_delta_loss"],
        min_delta_acc=lr_cfg["min_delta_acc"],
    )


def _initial_batch(cfg):
    batch_cfg = cfg["batch"]
    return batch_cfg["size"] if batch_cfg["policy"] == "fixed" else batch_cfg["initial"]


def run_primary(experiment, dataset, *, pilot_epochs=None):
    validate_primary_protocol()
    if experiment not in PRIMARY_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = deepcopy(PRIMARY_CONFIGS[experiment])
    is_pilot = pilot_epochs is not None
    epochs = int(pilot_epochs) if is_pilot else cfg["epochs"]
    if epochs <= 0:
        raise ValueError("pilot_epochs must be positive")

    seed = cfg["seed"]
    set_seed(seed)

    data_cls, num_classes, dataset_label = dataset_settings(dataset)
    data = data_cls()
    data.train_generator.manual_seed(seed)
    class_names = tuple(data.class_names)
    if len(class_names) != num_classes:
        raise ValueError("Dataset class-name count does not match model output size.")

    batch_size = _initial_batch(cfg)
    train_loader = data.get_train_loader(batch_size)
    val_loader = data.get_val_loader(batch_size=256)

    model = CustomCNN(num_classes=num_classes).to(DEVICE)
    optimizer = _build_optimizer(model, cfg)
    batch_controller = _build_batch_controller(cfg)
    lr_controller = _build_lr_controller(optimizer, cfg)

    # Apply the adaptive LR's batch-scaling rule before epoch 1.
    if lr_controller is not None:
        lr_controller.set_batch_size(batch_size)

    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S_%fZ")
    run_name = f"{dataset}_{experiment}_primary_seed{seed}"
    if is_pilot:
        run_name += f"_pilot{epochs}"

    category = "pilots" if is_pilot else "primary"
    results_dir = Path(RESULTS_DIR) / category / dataset / experiment / run_id
    checkpoint_dir = Path(CHECKPOINT_DIR) / category / dataset / experiment / run_id

    metadata = {
        "run_name": run_name,
        "run_id": run_id,
        "created_utc": created.isoformat(),
        "experiment": experiment,
        "label": cfg["label"],
        "dataset": dataset_label,
        "dataset_key": dataset,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "training_samples": len(data.train_dataset),
        "validation_samples": len(data.val_dataset),
        "official_test_evaluated_during_training": False,
        "epochs_planned": epochs,
        "primary_epoch_budget": cfg["epochs"],
        "is_pilot": is_pilot,
        "seed": seed,
        "protocol": cfg,
        "normalization": (
            "CIFAR-10 fixed project statistics"
            if dataset == "cifar10"
            else "Member-3 historical full-precision CIFAR-100 statistics"
        ),
        "device_type": DEVICE.type,
        "device_name": get_device_name(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "git_commit": _git_commit(),
        "results_dir": str(results_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }

    print("=" * 78)
    print(f"{experiment} | {cfg['label']}")
    print(f"Dataset: {dataset_label}")
    print(f"Epochs: {epochs}{' (PILOT)' if is_pilot else ' (PRIMARY)'}")
    print(f"Seed: {seed}")
    print(f"Initial batch: {batch_size}")
    print(f"Device: {get_device_name()}")
    if experiment == "E4":
        print("E4 coupling: batch decision -> next_batch_size -> LR decision")
    print("Official test evaluation: DISABLED during training")
    print("=" * 78)

    trainer = Trainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=DEVICE,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        run_metadata=metadata,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        batch_size=batch_size,
        batch_controller=batch_controller,
        train_loader_factory=data.get_train_loader if batch_controller is not None else None,
        lr_controller=lr_controller,
    )

    if not is_pilot and len(history) != 40:
        raise RuntimeError(f"Primary {experiment} completed {len(history)} epochs instead of 40.")

    if batch_controller is not None:
        (results_dir / "batch_history.json").write_text(
            json.dumps(batch_controller.decision_history, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    if lr_controller is not None:
        (results_dir / "lr_history.json").write_text(
            json.dumps(lr_controller.lr_history, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    evaluation = run_evaluation(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        history=history,
        total_training_seconds=trainer.total_training_seconds,
        checkpoint_path=checkpoint_path,
        output_dir=results_dir,
        run_name=run_name,
        split="validation",
        target_accuracy=TARGET_ACCURACY[dataset],
        class_names=class_names,
    )

    manifest = {
        "experiment": experiment,
        "dataset": dataset,
        "status": "completed",
        "is_pilot": is_pilot,
        "epochs_completed": len(history),
        "best_checkpoint_epoch": int(evaluation["checkpoint_epoch"]),
        "best_validation_loss": float(trainer.best_val_loss),
        "validation_accuracy": float(evaluation["metrics"]["accuracy"]),
        "validation_macro_f1": float(evaluation["metrics"]["macro_f1"]),
        "optimizer_updates": int(trainer.optimizer_updates),
        "training_time_seconds": float(trainer.total_training_seconds),
        "results_dir": str(results_dir),
        "checkpoint_path": str(checkpoint_path),
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Run manifest:", results_dir / "run_manifest.json")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(PRIMARY_CONFIGS), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=None,
        help="Development-only shorter run. Omit for the frozen 40-epoch primary run.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_primary(args.experiment, args.dataset, pilot_epochs=args.pilot_epochs)


if __name__ == "__main__":
    main()
