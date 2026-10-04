"""Confusion-matrix plotting shared by CIFAR-10 and CIFAR-100."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .metrics import CIFAR10_CLASSES, _validate_labels


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path,
    class_names=CIFAR10_CLASSES,
    normalize=False,
    title="Confusion matrix",
):
    y_true, y_pred = _validate_labels(y_true, y_pred, class_names)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix, dtype=float),
            where=totals != 0,
        )

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 10 if len(class_names) <= 20 else 12
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    try:
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None, aspect="auto")
        fig.colorbar(image, ax=ax, label="Fraction within true class" if normalize else "Images")
        positions = np.arange(len(class_names))
        if len(class_names) <= 20:
            ax.set_xticks(positions, labels=class_names, rotation=45, ha="right")
            ax.set_yticks(positions, labels=class_names)
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in positions:
                for column in positions:
                    value = matrix[row, column]
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}" if normalize else str(value),
                        ha="center",
                        va="center",
                        color="white" if value > threshold else "black",
                        fontsize=7,
                    )
        else:
            ticks = np.arange(0, len(class_names), 10)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)

        ax.set(xlabel="Predicted class", ylabel="True class", title=title)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return matrix
