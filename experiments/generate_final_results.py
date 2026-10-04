"""Generate final E1-E4 report tables and figures from locked results.

This script is reporting-only:
- it performs no training,
- it does not alter checkpoints or controller settings,
- it reads the already-completed validation/test artifacts.

Run from the repository root:
    python -m experiments.generate_final_results
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.datasets import add_dataset_argument, dataset_settings, dataset_run_name

from config import (
    CHECKPOINT_DIR,
    INITIAL_LEARNING_RATE,
    RESULTS_DIR,
    SEED,
)


LABELS = {
    "E1": "Fixed batch + fixed LR",
    "E2": "Adaptive batch + fixed LR",
    "E3": "Fixed batch + adaptive LR",
    "E4": "Adaptive batch + adaptive LR",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def newest_subdir(root):
    root = require(root, "results directory")
    dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        reverse=True,
    )
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return dirs[0]


def resolve_artifacts(dataset="cifar10"):
    dataset_settings(dataset)
    e1_name = (
        f"fixed_fixed_batch32_lr{INITIAL_LEARNING_RATE}_seed{SEED}"
    )
    e2_name = (
        f"adaptive_batch_fixed_lr{INITIAL_LEARNING_RATE}_seed{SEED}"
    )
    e3_name = (
        f"adaptive_lr_batch32_refLR{INITIAL_LEARNING_RATE}_seed{SEED}"
    )
    e4_name = (
        f"adaptive_batch_adaptive_lr_refLR{INITIAL_LEARNING_RATE}_seed{SEED}"
    )

    e1_name = dataset_run_name(dataset, e1_name)
    e2_name = dataset_run_name(dataset, e2_name)
    e3_name = dataset_run_name(dataset, e3_name)
    e4_name = dataset_run_name(dataset, e4_name)

    e3_dir = newest_subdir(Path(RESULTS_DIR) / e3_name)
    e4_dir = newest_subdir(Path(RESULTS_DIR) / e4_name)
    final_test_dir = newest_subdir(Path(RESULTS_DIR) / dataset_run_name(dataset, "final_test_evaluation"))

    return {
        "E1": {
            "run_name": e1_name,
            "history": require(
                Path(RESULTS_DIR) / f"{e1_name}.csv",
                "E1 training history",
            ),
            "metadata": require(
                Path(RESULTS_DIR) / f"{e1_name}_metadata.json",
                "E1 metadata",
            ),
            "validation": require(
                Path(RESULTS_DIR)
                / "fixed_validation_evaluation"
                / e1_name
                / "validation_metrics.json",
                "E1 validation metrics",
            ),
            "test": require(
                final_test_dir / "E1" / "test_metrics.json",
                "E1 test metrics",
            ),
        },
        "E2": {
            "run_name": e2_name,
            "history": require(
                Path(RESULTS_DIR) / f"{e2_name}.csv",
                "E2 training history",
            ),
            "metadata": require(
                Path(RESULTS_DIR) / f"{e2_name}_metadata.json",
                "E2 metadata",
            ),
            "validation": require(
                Path(RESULTS_DIR)
                / "adaptive_batch_validation_evaluation"
                / e2_name
                / "validation_metrics.json",
                "E2 validation metrics",
            ),
            "test": require(
                final_test_dir / "E2" / "test_metrics.json",
                "E2 test metrics",
            ),
        },
        "E3": {
            "run_name": e3_name,
            "history": require(
                e3_dir / f"{e3_name}.csv",
                "E3 training history",
            ),
            "metadata": require(
                e3_dir / f"{e3_name}_metadata.json",
                "E3 metadata",
            ),
            "validation": require(
                e3_dir / "validation_metrics.json",
                "E3 validation metrics",
            ),
            "test": require(
                final_test_dir / "E3" / "test_metrics.json",
                "E3 test metrics",
            ),
        },
        "E4": {
            "run_name": e4_name,
            "history": require(
                e4_dir / f"{e4_name}.csv",
                "E4 training history",
            ),
            "metadata": require(
                e4_dir / f"{e4_name}_metadata.json",
                "E4 metadata",
            ),
            "validation": require(
                e4_dir / "validation_metrics.json",
                "E4 validation metrics",
            ),
            "test": require(
                final_test_dir / "E4" / "test_metrics.json",
                "E4 test metrics",
            ),
        },
        "_final_test_dir": final_test_dir,
        "_dataset": dataset,
    }


def unpack_test_metrics(path):
    data = read_json(path)
    metrics = data.get("metrics", data)
    return data, metrics


def load_summary(artifacts):
    rows = []

    for exp in ("E1", "E2", "E3", "E4"):
        item = artifacts[exp]
        metadata = read_json(item["metadata"])
        expected_dataset = dataset_settings(artifacts.get("_dataset", "cifar10"))[2]
        if metadata.get("dataset") != expected_dataset:
            raise ValueError(f"{exp} dataset mismatch: expected {expected_dataset}")
        validation = read_json(item["validation"])
        test_detail, test = unpack_test_metrics(item["test"])

        rows.append(
            {
                "experiment": exp,
                "label": LABELS[exp],
                "checkpoint_epoch": int(
                    test_detail.get(
                        "checkpoint_epoch",
                        validation.get("checkpoint_epoch"),
                    )
                ),
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_macro_f1": float(validation["macro_f1"]),
                "test_loss": float(test["loss"]),
                "test_accuracy": float(test["accuracy"]),
                "test_macro_f1": float(test["macro_f1"]),
                "validation_minus_test_accuracy_pp": (
                    100.0
                    * (
                        float(validation["accuracy"])
                        - float(test["accuracy"])
                    )
                ),
                "optimizer_updates": int(
                    metadata["total_optimizer_updates"]
                ),
                "training_time_seconds": float(
                    metadata["total_training_time_seconds"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(summary, output_dir):
    x = list(range(len(summary)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        [value - width / 2 for value in x],
        summary["validation_accuracy"] * 100,
        width,
        label="Validation",
    )
    ax.bar(
        [value + width / 2 for value in x],
        summary["test_accuracy"] * 100,
        width,
        label="Test",
    )
    ax.set_xticks(x, summary["experiment"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation and test accuracy across E1-E4")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "01_validation_test_accuracy.png")


def plot_macro_f1(summary, output_dir):
    x = list(range(len(summary)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        [value - width / 2 for value in x],
        summary["validation_macro_f1"],
        width,
        label="Validation",
    )
    ax.bar(
        [value + width / 2 for value in x],
        summary["test_macro_f1"],
        width,
        label="Test",
    )
    ax.set_xticks(x, summary["experiment"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Validation and test macro F1 across E1-E4")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "02_validation_test_macro_f1.png")


def plot_updates(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        summary["experiment"],
        summary["optimizer_updates"],
    )
    ax.set_ylabel("Optimizer updates")
    ax.set_title("Optimizer-update cost across E1-E4")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "03_optimizer_updates.png")


def plot_training_time(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        summary["experiment"],
        summary["training_time_seconds"],
    )
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("Observed wall-clock training time")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "04_training_time.png")


def load_histories(artifacts):
    return {
        exp: pd.read_csv(artifacts[exp]["history"])
        for exp in ("E1", "E2", "E3", "E4")
    }


def plot_history_metric(histories, metric, ylabel, title, filename, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in ("E1", "E2", "E3", "E4"):
        frame = histories[exp]
        ax.plot(
            frame["epoch"],
            frame[metric],
            marker="o",
            markersize=3,
            label=exp,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / filename)


def plot_batch_schedule(histories, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for exp in ("E2", "E4"):
        frame = histories[exp]
        ax.step(
            frame["epoch"],
            frame["batch_size"],
            where="mid",
            label=exp,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Batch size")
    ax.set_title("Adaptive batch-size schedules")
    ax.set_yticks([32, 64, 128])
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / "08_adaptive_batch_schedule.png")


def plot_lr_schedule(histories, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for exp in ("E3", "E4"):
        frame = histories[exp]
        ax.step(
            frame["epoch"],
            frame["learning_rate"],
            where="mid",
            label=exp,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Adaptive learning-rate schedules")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / "09_adaptive_learning_rate_schedule.png")


def write_key_findings(summary, output_dir, dataset="cifar10"):
    dataset_label = dataset_settings(dataset)[2]
    indexed = summary.set_index("experiment")

    e1 = indexed.loc["E1"]
    e2 = indexed.loc["E2"]
    e3 = indexed.loc["E3"]
    e4 = indexed.loc["E4"]

    update_reduction = (
        100.0
        * (e1["optimizer_updates"] - e2["optimizer_updates"])
        / e1["optimizer_updates"]
    )

    findings = f"""# Locked E1-E4 result summary

These values come from the frozen seed-{SEED} primary experiments and the
final test evaluation. They are descriptive results for this experiment,
not claims about all random seeds or all {dataset_label} training settings.

## Main observations

- E1 fixed batch + fixed LR: test accuracy {100*e1['test_accuracy']:.2f}%,
  macro F1 {e1['test_macro_f1']:.4f}.
- E2 adaptive batch + fixed LR: test accuracy {100*e2['test_accuracy']:.2f}%,
  macro F1 {e2['test_macro_f1']:.4f}, using {int(e2['optimizer_updates']):,}
  optimizer updates.
- E3 fixed batch + adaptive LR: test accuracy {100*e3['test_accuracy']:.2f}%,
  macro F1 {e3['test_macro_f1']:.4f}.
- E4 adaptive batch + adaptive LR: test accuracy {100*e4['test_accuracy']:.2f}%,
  macro F1 {e4['test_macro_f1']:.4f}, using {int(e4['optimizer_updates']):,}
  optimizer updates.

Relative to E1, E2 improved test accuracy by
{100*(e2['test_accuracy']-e1['test_accuracy']):.2f} percentage points and
used {int(e1['optimizer_updates']-e2['optimizer_updates']):,} fewer optimizer
updates ({update_reduction:.1f}% fewer). E3 improved test accuracy by
{100*(e3['test_accuracy']-e1['test_accuracy']):.2f} percentage points. E4
improved test accuracy by
{100*(e4['test_accuracy']-e1['test_accuracy']):.2f} percentage points and used
the same reduced update count as E2.

E2 and E4 differed by only
{100*abs(e2['test_accuracy']-e4['test_accuracy']):.2f} percentage points in
test accuracy. With only one primary seed, this small difference should not
be treated as evidence that one adaptive strategy is generally superior.

## Reporting caution

Wall-clock time should be reported as observed hardware-specific runtime.
The update-count comparison is the cleaner computational-effort comparison
here because E2's recorded wall-clock time was substantially longer than the
other runs despite using fewer optimizer updates.
"""

    (output_dir / "KEY_FINDINGS.md").write_text(
        findings,
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create final report assets from frozen E1-E4 results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "final_report_assets",
    )
    add_dataset_argument(parser)
    args = parser.parse_args()
    if args.dataset == "cifar100" and args.output_dir == Path(RESULTS_DIR) / "final_report_assets":
        args.output_dir = Path(RESULTS_DIR) / "cifar100_final_report_assets"
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = resolve_artifacts(args.dataset)
    summary = load_summary(artifacts)
    histories = load_histories(artifacts)

    summary.to_csv(
        args.output_dir / "final_results_summary.csv",
        index=False,
    )

    plot_accuracy(summary, args.output_dir)
    plot_macro_f1(summary, args.output_dir)
    plot_updates(summary, args.output_dir)
    plot_training_time(summary, args.output_dir)

    plot_history_metric(
        histories,
        "val_accuracy",
        "Validation accuracy (%)",
        "Validation accuracy by epoch",
        "05_validation_accuracy_curves.png",
        args.output_dir,
    )
    plot_history_metric(
        histories,
        "val_loss",
        "Validation cross-entropy loss",
        "Validation loss by epoch",
        "06_validation_loss_curves.png",
        args.output_dir,
    )
    plot_history_metric(
        histories,
        "gradient_norm_cv",
        "Gradient-norm CV",
        "Within-epoch gradient-norm variability",
        "07_gradient_cv_curves.png",
        args.output_dir,
    )

    plot_batch_schedule(histories, args.output_dir)
    plot_lr_schedule(histories, args.output_dir)
    write_key_findings(summary, args.output_dir, args.dataset)

    print("=" * 78)
    print("FINAL REPORT ASSETS GENERATED")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("=" * 78)
    print(f"Saved to: {args.output_dir}")
    print("No training or test-set model selection was performed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
