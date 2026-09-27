"""Post-training evaluation for baseline and adaptive experiments."""

import json
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from .metrics import evaluate_model, summarize_training
from .confusion_matrix import plot_confusion_matrix
from .plots import plot_training_curves


def convert_training_history(history):
    """Convert Trainer records into the evaluation history format."""
    converted = []
    previous_updates = 0

    for row in history:
        cumulative_updates = row["optimizer_updates"]

        converted.append({
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "train_acc": row["train_accuracy"] / 100.0,
            "val_acc": row["val_accuracy"] / 100.0,
            "elapsed_seconds": row["elapsed_seconds"],
            "optimizer_updates": cumulative_updates - previous_updates,
        })

        previous_updates = cumulative_updates

    return converted


def save_json(data, path):
    """Save ordinary Python values as formatted JSON."""
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_evaluation(
    *,
    model,
    data_loader,
    device,
    history,
    total_training_seconds,
    checkpoint_path,
    output_dir,
    run_name,
    split="validation",
    target_accuracy=0.80,
):
    """Save training summaries, curves and checkpoint evaluation results.

    history must use the current Trainer's record format.
    Target accuracy is always measured on validation history.

    Loads checkpoint weights into the supplied model and leaves those
    weights loaded. Use after training, not inside the training loop.
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'")

    evaluation_history = convert_training_history(history)

    summary = summarize_training(
        evaluation_history,
        total_training_seconds=total_training_seconds,
        target_accuracy=target_accuracy,
    )

    # Evaluate the selected checkpoint rather than the final epoch.
    # Allow version metadata in checkpoints produced by this project.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate_model(model, data_loader, device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        evaluation_history,
        output_dir / "evaluation_history.json",
    )
    save_json(
        summary,
        output_dir / "training_summary.json",
    )
    save_json(
        {
            "split": split,
            "checkpoint_epoch": checkpoint["epoch"],
            **metrics,
        },
        output_dir / f"{split}_metrics.json",
    )

    plot_training_curves(
        {run_name: evaluation_history},
        output_dir / "training_curves.png",
    )

    for normalize in (False, True):
        suffix = "_normalized" if normalize else ""

        plot_confusion_matrix(
            metrics["y_true"],
            metrics["y_pred"],
            output_dir / f"{split}_confusion_matrix{suffix}.png",
            normalize=normalize,
            title=f"{split.capitalize()} confusion matrix{suffix.replace('_', ' ')}",
        )

    print(f"Evaluated checkpoint: epoch {checkpoint['epoch']}")
    print(f"{split.capitalize()} accuracy: {metrics['accuracy']:.2%}")
    print(f"{split.capitalize()} macro F1: {metrics['macro_f1']:.4f}")
    print(f"Evaluation results saved to: {output_dir}")

    return {
        "metrics": metrics,
        "training_summary": summary,
        "checkpoint_epoch": checkpoint["epoch"],
    }