import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    BATCH_SIZE_OPTIONS,
    CHECKPOINT_DIR,
    DEVICE,
    INITIAL_LEARNING_RATE,
    RESULTS_DIR,
    SEED,
    set_seed,
)

from data.cifar10 import CIFAR10DataModule
from evaluation.eval_pipeline import run_evaluation
from models.custom_cnn import CustomCNN


def load_training_history(csv_path):
    """
    Load an existing Trainer CSV and convert it into the
    history format expected by the evaluation pipeline.

    Older experiment CSV files contain epoch_time_seconds
    but not elapsed_seconds. For those runs, cumulative
    elapsed time is reconstructed from the recorded
    per-epoch times.
    """

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "optimizer_updates",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing history columns in {csv_path}: "
            f"{sorted(missing_columns)}"
        )

    has_elapsed_seconds = (
        "elapsed_seconds"
        in dataframe.columns
    )

    has_epoch_time = (
        "epoch_time_seconds"
        in dataframe.columns
    )

    if (
        not has_elapsed_seconds
        and not has_epoch_time
    ):
        raise ValueError(
            f"{csv_path} contains neither "
            "'elapsed_seconds' nor "
            "'epoch_time_seconds'."
        )

    history = []

    cumulative_elapsed = 0.0

    for row in dataframe.to_dict(
        orient="records"
    ):
        if has_elapsed_seconds:
            elapsed_seconds = float(
                row["elapsed_seconds"]
            )
        else:
            cumulative_elapsed += float(
                row["epoch_time_seconds"]
            )

            elapsed_seconds = (
                cumulative_elapsed
            )

        history.append(
            {
                "epoch": int(
                    row["epoch"]
                ),
                "train_loss": float(
                    row["train_loss"]
                ),
                "train_accuracy": float(
                    row["train_accuracy"]
                ),
                "val_loss": float(
                    row["val_loss"]
                ),
                "val_accuracy": float(
                    row["val_accuracy"]
                ),
                "optimizer_updates": int(
                    row["optimizer_updates"]
                ),
                "elapsed_seconds": (
                    elapsed_seconds
                ),
            }
        )

    return history


def load_metadata(metadata_path):
    """
    Load and validate metadata from a completed experiment.
    """

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata.get("status") != "completed":
        raise ValueError(
            f"Experiment is not marked completed: "
            f"{metadata_path}"
        )

    if "total_training_time_seconds" not in metadata:
        raise ValueError(
            "Metadata does not contain "
            "total_training_time_seconds."
        )

    return metadata


def evaluate_fixed_run(
    batch_size,
    data,
    target_accuracy,
):
    """
    Evaluate one already-trained fixed-batch experiment.

    This does NOT train the network.
    """

    run_name = (
        f"fixed_fixed"
        f"_batch{batch_size}"
        f"_lr{INITIAL_LEARNING_RATE}"
        f"_seed{SEED}"
    )

    csv_path = (
        Path(RESULTS_DIR)
        / f"{run_name}.csv"
    )

    metadata_path = (
        Path(RESULTS_DIR)
        / f"{run_name}_metadata.json"
    )

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / f"{run_name}_best.pt"
    )

    required_files = [
        csv_path,
        metadata_path,
        checkpoint_path,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print(
            f"\nSkipping {run_name}."
        )

        print(
            "Missing required file(s):"
        )

        for path in missing_files:
            print(
                f"  {path}"
            )

        return

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"Evaluating existing run: "
        f"{run_name}"
    )

    print(
        "=" * 78
    )

    history = load_training_history(
        csv_path
    )

    metadata = load_metadata(
        metadata_path
    )

    val_loader = (
        data.get_val_loader(
            batch_size=256
        )
    )

    model = CustomCNN().to(
        DEVICE
    )

    output_dir = (
        Path(RESULTS_DIR)
        / "fixed_validation_evaluation"
        / run_name
    )

    result = run_evaluation(
        model=model,
        data_loader=val_loader,
        device=DEVICE,
        history=history,
        total_training_seconds=float(
            metadata[
                "total_training_time_seconds"
            ]
        ),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        run_name=run_name,
        split="validation",
        target_accuracy=target_accuracy,
    )

    metrics = result[
        "metrics"
    ]

    print(
        f"Validation loss:      "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Validation accuracy:  "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Macro precision:      "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro recall:         "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:             "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Checkpoint epoch:     "
        f"{result['checkpoint_epoch']}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-trained fixed-batch "
            "CIFAR-10 experiments."
        )
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=BATCH_SIZE_OPTIONS,
        help=(
            "Existing fixed batch sizes to evaluate. "
            "Example: --batch-sizes 32"
        ),
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        help=(
            "Validation target accuracy used in "
            "the training summary."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    invalid_batch_sizes = [
        batch_size
        for batch_size in args.batch_sizes
        if batch_size
        not in BATCH_SIZE_OPTIONS
    ]

    if invalid_batch_sizes:
        raise ValueError(
            f"Unsupported batch size(s): "
            f"{invalid_batch_sizes}"
        )

    if not (
        0.0
        <= args.target_accuracy
        <= 1.0
    ):
        raise ValueError(
            "target_accuracy must be "
            "between 0 and 1."
        )

    set_seed(
        SEED
    )

    data = (
        CIFAR10DataModule()
    )

    print(
        "\nPost-training evaluation only."
    )

    print(
        "No optimizer or training loop "
        "will be executed."
    )

    for batch_size in args.batch_sizes:
        evaluate_fixed_run(
            batch_size=batch_size,
            data=data,
            target_accuracy=(
                args.target_accuracy
            ),
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Requested evaluations finished."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()