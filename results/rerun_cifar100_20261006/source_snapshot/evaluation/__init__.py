"""Public evaluation helpers; see README.md for integration instructions."""

from .metrics import (
    CIFAR10_CLASSES,
    TrainingTimer,
    classification_metrics,
    evaluate_model,
    generalization_gap,
    summarize_training,
)

__all__ = [
    "CIFAR10_CLASSES", "TrainingTimer", "classification_metrics",
    "evaluate_model", "generalization_gap", "summarize_training",
]
