"""Compare training histories from baseline and adaptive strategies."""

from pathlib import Path

import matplotlib.pyplot as plt

from .metrics import validate_history


def plot_training_curves(histories, save_path):
    """Save loss and accuracy vs epoch, each with actual batch size.

    Optional batch_size values describe the batch used in each completed epoch,
    not the controller decision for the following epoch. Legacy histories without
    batch sizes still render the loss and accuracy curves.
    """
    if not histories:
        raise ValueError("Provide at least one strategy history")
    histories = {name: validate_history(rows) for name, rows in histories.items()}
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 1)
    loss_ax = fig.add_subplot(grid[0, 0])
    accuracy_ax = fig.add_subplot(grid[1, 0])
    batch_axes = {}
    try:
        for name, rows in histories.items():
            epochs = [row["epoch"] for row in rows]
            line, = loss_ax.plot(epochs, [row["train_loss"] for row in rows],
                                 label=f"{name}: train")
            color = line.get_color()
            loss_ax.plot(epochs, [row["val_loss"] for row in rows],
                         color=color, linestyle="--", label=f"{name}: validation")
            if all("batch_size" in row for row in rows):
                batches = [row["batch_size"] for row in rows]
                if any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in batches):
                    raise ValueError("batch_size must be a positive integer")
                for ax in (loss_ax, accuracy_ax):
                    if ax not in batch_axes:
                        batch_axes[ax] = ax.twinx()
                        batch_axes[ax].set_ylabel("Batch size")
                    # Midpoint boundaries center each epoch's actual batch value.
                    batch_axes[ax].step(
                        epochs, batches, where="mid", color=color,
                        linestyle=":", linewidth=2, label=f"{name}: batch size",
                    )
            accuracy_ax.plot(epochs, [100 * row["train_acc"] for row in rows],
                             color=color, label=f"{name}: train")
            accuracy_ax.plot(epochs, [100 * row["val_acc"] for row in rows],
                             color=color, linestyle="--", label=f"{name}: validation")
        loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
                    title="Loss and batch size by epoch")
        accuracy_ax.set(xlabel="Epoch", ylabel="Accuracy (%)",
                        title="Accuracy and batch size by epoch")
        for ax in (loss_ax, accuracy_ax):
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if ax in batch_axes:
                batch_handles, batch_labels = batch_axes[ax].get_legend_handles_labels()
                handles += batch_handles
                labels += batch_labels
            ax.legend(handles, labels)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
