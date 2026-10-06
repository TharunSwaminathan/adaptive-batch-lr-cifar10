"""Classification metrics and experiment summaries.

Accuracies, precision, recall and F1 use fractions in [0, 1]. Times use
seconds. See README.md for the epoch-history format and integration example.
"""

import math
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _validate_labels(y_true, y_pred, class_names):
    """Reject empty, mismatched, or out-of-range class labels."""
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be nonempty and unique")
    arrays = [np.asarray(values) for values in (y_true, y_pred)]
    for values in arrays:
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Labels must be nonempty one-dimensional arrays")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Labels must be integer class indices")
        if np.any(values < 0) or np.any(values >= len(class_names)):
            raise ValueError("Label index is outside class_names")
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("y_true and y_pred must have the same length")
    return arrays


def classification_metrics(y_true, y_pred, class_names=CIFAR10_CLASSES):
    """Compute accuracy, macro scores and per-class scores.

    All configured classes contribute to macro averages, including classes
    absent from the input. Undefined precision/recall/F1 values become zero.
    The result contains ordinary Python values suitable for JSON output.
    """
    y_true, y_pred = _validate_labels(y_true, y_pred, class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "num_samples": int(y_true.size),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


def evaluate_model(model, data_loader, device, class_names=CIFAR10_CLASSES):
    """Evaluate single-label classification with unweighted cross entropy.

    The model must already be on device and return [batch, num_classes]
    logits. Returns metrics plus y_true/y_pred lists. Original module
    training flags are restored, even if evaluation fails.
    """
    original_modes = [(module, module.training) for module in model.modules()]
    total_loss = 0.0
    y_true, y_pred = [], []
    try:
        model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                if logits.ndim != 2 or logits.shape[1] != len(class_names):
                    raise ValueError("Model output must have shape [batch, num_classes]")
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss).item():
                    raise ValueError("Evaluation loss is not finite")
                total_loss += loss.item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    finally:
        for module, training in original_modes:
            module.training = training

    result = classification_metrics(y_true, y_pred, class_names)
    result.update(loss=total_loss / len(y_true), y_true=y_true, y_pred=y_pred)
    return result


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class TrainingTimer:
    """Wall-clock timer that includes validation inside the timed region.

    GPU work is synchronized when reading elapsed time and on exit. Use one
    timer per run; read elapsed_seconds after validation for each history row.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self._start = None
        self.total_seconds = None

    @contextmanager
    def measure(self):
        if self._start is not None:
            raise RuntimeError("Use a new TrainingTimer for each run")
        _synchronize(self.device)
        self._start = perf_counter()
        try:
            yield self
        finally:
            _synchronize(self.device)
            self.total_seconds = perf_counter() - self._start

    @property
    def elapsed_seconds(self):
        if self._start is None:
            raise RuntimeError("Timer has not started")
        if self.total_seconds is not None:
            return self.total_seconds
        _synchronize(self.device)
        return perf_counter() - self._start


def validate_history(history):
    """Validate a nonempty sequence of per-epoch dictionaries; return a list."""
    rows = list(history)
    if not rows:
        raise ValueError("history must contain at least one epoch")
    previous_epoch, previous_time = 0, 0.0
    for row in rows:
        epoch = row["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= previous_epoch:
            raise ValueError("epoch must be a positive, strictly increasing integer")
        for key in ("train_loss", "val_loss", "elapsed_seconds"):
            value = row[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        if row["elapsed_seconds"] < previous_time:
            raise ValueError("elapsed_seconds must be cumulative and nondecreasing")
        for key in ("train_acc", "val_acc"):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"{key} must be a fraction in [0, 1]")
        updates = row["optimizer_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("optimizer_updates must be a non-negative integer for this epoch")
        previous_epoch, previous_time = epoch, row["elapsed_seconds"]
    return rows


def summarize_training(history, total_training_seconds, target_accuracy=0.8):
    """Summarize cost and the FIRST epoch reaching target validation accuracy.

    The target uses validation accuracy to avoid repeated test-set selection.
    optimizer_updates are per epoch, not cumulative or inferred from batches.
    Return None for target epoch/time if the target was never reached.
    """
    rows = validate_history(history)
    if not math.isfinite(target_accuracy) or not 0 <= target_accuracy <= 1:
        raise ValueError("target_accuracy must be a fraction in [0, 1]")
    if not math.isfinite(total_training_seconds) or total_training_seconds < rows[-1]["elapsed_seconds"]:
        raise ValueError("total_training_seconds must cover all recorded epochs")
    reached = next((row for row in rows if row["val_acc"] >= target_accuracy), None)
    return {
        "training_time_seconds": float(total_training_seconds),
        "optimizer_updates": sum(row["optimizer_updates"] for row in rows),
        "target_accuracy": float(target_accuracy),
        "target_split": "validation",
        "epoch_to_target": reached["epoch"] if reached else None,
        "time_to_target_seconds": float(reached["elapsed_seconds"]) if reached else None,
    }


def generalization_gap(train_accuracy, test_accuracy):
    """Return train minus test accuracy, in percentage points.

    Evaluate BOTH datasets on the same checkpoint in eval mode. The training
    evaluation loader should disable random augmentation. Do not substitute
    the online accuracy collected while model weights were changing.
    """
    for value in (train_accuracy, test_accuracy):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Accuracies must be fractions in [0, 1]")
    return 100.0 * (train_accuracy - test_accuracy)
