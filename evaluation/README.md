# Evaluation Integration Guide

This directory handles evaluation only. It does not start training or modify the optimizer. All accuracy, precision, recall, and F1 inputs and outputs use **fractions between 0 and 1**; multiply by 100 to display percentages. Times are measured in seconds. The generalization gap is returned in **percentage points**.

## Files and Features

- `metrics.py`: Test loss, accuracy, macro precision/recall/F1, per-class metrics, training timing, time to target accuracy, optimizer update counts, and generalization gap.
- `confusion_matrix.py`: Saves confusion matrices as raw counts or normalized by true-class support.
- `plots.py`: Compares training/validation loss and accuracy across strategies, using epochs and cumulative elapsed time as the horizontal axes.

Import these modules from the project root rather than running the files in this directory directly.

## Test Set Evaluation

```python
import json
from pathlib import Path
from evaluation import evaluate_model, generalization_gap
from evaluation.confusion_matrix import plot_confusion_matrix

# Load the checkpoint selected using validation performance and move model to device.
result = evaluate_model(model, data.get_test_loader(), device)
print(f"Test accuracy: {result['accuracy']:.2%}")
print(f"Macro F1: {result['macro_f1']:.4f}")

output = Path("results") / "adaptive_lr"
output.mkdir(parents=True, exist_ok=True)
(output / "test_metrics.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8",
)
plot_confusion_matrix(
    result["y_true"], result["y_pred"], output / "confusion_matrix.png",
)
plot_confusion_matrix(
    result["y_true"], result["y_pred"], output / "confusion_matrix_normalized.png",
    normalize=True,
)
```

`evaluate_model` uses cross entropy without class weights and averages loss over the actual number of samples, including an incomplete final batch. It automatically uses evaluation mode and disables gradient computation, then restores each module's original training/evaluation state. Empty datasets and non-finite loss values raise errors. Classification metrics include all ten classes by default; undefined precision, recall, and F1 values are set to zero.

## Training History Interface (for Dev to Integrate)

Append one dictionary after validation at the end of each epoch:

```python
history.append({
    "epoch": epoch,                         # Starts at 1
    "train_loss": train_loss,               # Average over samples
    "val_loss": val_loss,                   # Average over samples
    "train_acc": train_accuracy,            # 0–1
    "val_acc": val_accuracy,                # 0–1
    "elapsed_seconds": timer.elapsed_seconds,  # Cumulative since training started
    "optimizer_updates": epoch_updates,     # Actual optimizer steps in this epoch only
})
```

Increment `optimizer_updates` only when a parameter update actually occurs. Do not substitute the batch count or `ceil(N / batch_size)`. This distinction matters with gradient accumulation and updates skipped by AMP. This field is a per-epoch count; the summary function adds the counts together.

Use the timer as follows, with training and validation provided by the shared trainer:

```python
from evaluation import TrainingTimer, summarize_training
from evaluation.plots import plot_training_curves

timer = TrainingTimer(device)
history = []
with timer.measure():
    # Run the complete epoch loop here: train, validate, and append a history row.
    # Read timer.elapsed_seconds after validation.
    ...

# Run after history has been populated:
summary = summarize_training(
    history,
    total_training_seconds=timer.total_seconds,
    target_accuracy=0.80,
)
plot_training_curves({"adaptive_lr": history}, "results/adaptive_lr/curves.png")
```

This is an integration template, not a standalone training script. Timing includes data loading, training, validation, and other operations inside the loop. Keep dataset downloads, final testing, and plotting outside the timed region. Use the same timing boundaries for all strategies. CUDA/MPS work is synchronized when reading elapsed time so that pending device computation is included.

`summary` contains:

- `training_time_seconds`: Total duration of the timed region.
- `optimizer_updates`: Total number of actual optimizer updates.
- `epoch_to_target` / `time_to_target_seconds`: The **first** validation epoch that reaches the target accuracy and its cumulative elapsed time. Both are `None` (`null` in JSON) if the target is never reached. Measurements have end-of-epoch resolution, not batch-level resolution.
- `target_accuracy` / `target_split`: The target value and the dataset split used to measure it (`validation`).

To compare multiple strategies, pass `{"fixed_fixed": history_a, "adaptive_batch": history_b, "adaptive_lr": history_c, "adaptive_combined": history_d}`. Strategies may have different numbers of epochs.

## Generalization Gap

```python
# Use the original training subset with random cropping, flipping, and other augmentation disabled.
train_result = evaluate_model(model, train_eval_loader, device)
gap = generalization_gap(train_result["accuracy"], result["accuracy"])
print(f"Generalization gap: {gap:.2f} percentage points")
```

Evaluate both datasets using the same checkpoint. The current `get_train_loader()` applies random augmentation, so it is not suitable for this training-set evaluation. The data/training module owner should provide a loader for the training subset with random augmentation disabled. Do not use training accuracy accumulated while model weights were changing to calculate the final generalization gap. Preserve negative gap values when they occur.


Training curve figures show loss and accuracy only against epoch. Both panels overlay
optional `batch_size` values as a dotted step curve on the right axis, using the
batch actually used for each epoch, not `next_batch_size`. `run_evaluation`
preserves this field when converting Trainer history. Accuracy is displayed as a percentage. Legacy histories without batch sizes are supported.
