import argparse
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    INITIAL_LEARNING_RATE,
    RESULTS_DIR,
    SEED,
    set_seed,
)
from data.cifar10 import CIFAR10DataModule
from evaluation.eval_pipeline import run_evaluation
from experiments.evaluate_fixed_runs import (
    load_metadata,
    load_training_history,
)
from models.custom_cnn import CustomCNN


def evaluate_adaptive_batch_run(target_accuracy):
    """
    Evaluate the already-trained primary E2 checkpoint on validation only.

    This script does not train the model and does not use the test set.
    """

    run_name = (
        f"adaptive_batch_fixed"
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
        print("Missing required file(s):")

        for path in missing_files:
            print(f"  {path}")

        raise FileNotFoundError(
            "Primary E2 artifacts are incomplete."
        )

    history = load_training_history(
        csv_path
    )

    metadata = load_metadata(
        metadata_path
    )

    set_seed(SEED)

    data = CIFAR10DataModule()

    val_loader = data.get_val_loader(
        batch_size=256
    )

    model = CustomCNN().to(
        DEVICE
    )

    output_dir = (
        Path(RESULTS_DIR)
        / "adaptive_batch_validation_evaluation"
        / run_name
    )

    print("\nPost-training E2 validation evaluation only.")
    print("No optimizer or training loop will be executed.")
    print("The test set is not being evaluated.\n")

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

    metrics = result["metrics"]

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
            "Evaluate the completed E2 adaptive-batch "
            "checkpoint on the validation set."
        )
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        help=(
            "Validation target accuracy used by the "
            "training summary."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not (
        0.0
        <= args.target_accuracy
        <= 1.0
    ):
        raise ValueError(
            "target_accuracy must be between 0 and 1."
        )

    evaluate_adaptive_batch_run(
        target_accuracy=args.target_accuracy
    )


if __name__ == "__main__":
    main()
