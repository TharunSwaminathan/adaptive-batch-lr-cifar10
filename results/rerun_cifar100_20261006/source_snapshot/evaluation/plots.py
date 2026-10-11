"""Training/validation curves by epoch or measured cumulative training time."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import validate_history

# Preserve the E3/E4 colors across individual and four-strategy figures.
STRATEGY_COLORS = {"E1": "#7f7f7f", "E2": "#9467bd", "E3": "#1f77b4", "E4": "#ff7f0e"}


def plot_training_curves(histories, save_path, *, x_axis="epoch", metric=None):
    """Plot paired curves or a separate loss/accuracy figure, with actual batch.

    Time comes from cumulative measured elapsed_seconds (training + validation),
    never from rescaling epoch numbers. Lines stop at each run's measured end.
    """
    if not histories:
        raise ValueError("Provide at least one strategy history")
    if x_axis not in ("epoch", "time") or metric not in (None, "loss", "accuracy"):
        raise ValueError("Invalid x_axis or metric")
    histories = {name: validate_history(rows) for name, rows in histories.items()}
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [metric] if metric else ["loss", "accuracy"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 if metric else 9), squeeze=False)
    try:
        for ax, kind in zip(axes[:, 0], metrics):
            batch_ax = None
            for name, rows in histories.items():
                x = [row["epoch"] if x_axis == "epoch" else row["elapsed_seconds"] for row in rows]
                train_key, val_key = ("train_loss", "val_loss") if kind == "loss" else ("train_acc", "val_acc")
                scale = 1 if kind == "loss" else 100
                line, = ax.plot(x, [scale * row[train_key] for row in rows], label=f"{name}: train", color=STRATEGY_COLORS.get(name))
                color = line.get_color()
                ax.plot(x, [scale * row[val_key] for row in rows], color=color, linestyle="--", label=f"{name}: validation")
                if all("batch_size" in row for row in rows):
                    batches = [row["batch_size"] for row in rows]
                    if any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in batches):
                        raise ValueError("batch_size must be a positive integer")
                    if batch_ax is None:
                        batch_ax = ax.twinx()
                        batch_ax.set_ylabel("Batch size")
                    # Each sample marks the end of an epoch; its batch was used
                    # over the preceding interval, including on the time axis.
                    batch_ax.step([0] + x, [batches[0]] + batches, where="pre", color=color,
                                  linestyle=":", linewidth=2, label=f"{name}: batch size")
            label = "Epoch" if x_axis == "epoch" else "Elapsed training time (seconds)"
            ax.set(xlabel=label, ylabel="Cross-entropy loss" if kind == "loss" else "Accuracy (%)",
                   title=f"{kind.capitalize()} and batch size by {'epoch' if x_axis == 'epoch' else 'training time'}")
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if batch_ax is not None:
                extra_handles, extra_labels = batch_ax.get_legend_handles_labels()
                handles += extra_handles
                labels += extra_labels
            ax.legend(handles, labels, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def save_curve_suite(histories, output_dir, *, prefix=""):
    """Save combined epoch/time plots plus four separate PNG and PDF plots."""
    root = Path(output_dir)
    stem = f"{prefix}_" if prefix else ""
    outputs = []
    for axis in ("epoch", "time"):
        combined = "training_curves" + ("_time" if axis == "time" else "")
        outputs.append(plot_training_curves(histories, root / f"{stem}{combined}.png", x_axis=axis))
        for metric in ("loss", "accuracy"):
            for suffix in ("png", "pdf"):
                outputs.append(plot_training_curves(histories, root / f"{stem}{metric}_{axis}.{suffix}",
                                                    x_axis=axis, metric=metric))
    return outputs
