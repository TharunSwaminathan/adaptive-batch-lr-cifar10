"""Compare training histories from baseline and adaptive strategies."""

from pathlib import Path

import matplotlib.pyplot as plt

from .metrics import validate_history


def plot_training_curves(histories, save_path):
    """Save train/validation loss and accuracy against epoch and elapsed time.

    histories maps strategy names to history rows documented in README.md.
    Solid lines are training; dashed lines are validation. Each strategy
    uses one color across all panels. Accuracy is displayed as a percentage.
    """
    if not histories:
        raise ValueError("Provide at least one strategy history")
    histories = {name: validate_history(rows) for name, rows in histories.items()}
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    try:
        for name, rows in histories.items():
            color = None
            for column, x_key in enumerate(("epoch", "elapsed_seconds")):
                x = [row[x_key] for row in rows]
                for axis_row, metric in enumerate(("loss", "acc")):
                    ax = axes[axis_row, column]
                    scale = 100 if metric == "acc" else 1
                    line, = ax.plot(
                        x, [scale * row[f"train_{metric}"] for row in rows],
                        color=color, label=f"{name}: train",
                    )
                    color = line.get_color()
                    ax.plot(
                        x, [scale * row[f"val_{metric}"] for row in rows],
                        color=color, linestyle="--", label=f"{name}: validation",
                    )
        for column, xlabel in enumerate(("Epoch", "Elapsed wall-clock time (seconds)")):
            for row, ylabel in enumerate(("Cross-entropy loss", "Accuracy (%)")):
                ax = axes[row, column]
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.grid(alpha=0.25)
                ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
