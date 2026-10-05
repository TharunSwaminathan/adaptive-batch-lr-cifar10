# adaptive-batch-lr-cifar10

CNN experiments on CIFAR-10 with fixed or adaptive batch sizes and learning rates.
This guide describes the implemented **fixed batch size + adaptive learning rate**
experiment in `experiments/adaptive_lr.py`.

## Setup

Run commands from the project root, the directory containing `config.py`,
`requirements.txt`, and `experiments/`. Activate your Python environment and
install the project dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

The data module downloads CIFAR-10 on first use. Dataset sizes, device selection,
normalization, SGD momentum, and weight decay come from `config.py`. The current
adaptive LR entry point supports SGD. It evaluates the validation set after
training; it does not evaluate the test set or use it to adjust LR.

## Run the Adaptive LR Experiment

Show all available command-line options without starting training:

```bash
python -m experiments.adaptive_lr --help
```

Run 30 epochs with a fixed batch size of 32 and reference LR of 0.01:

```bash
python -m experiments.adaptive_lr \
  --epochs 30 \
  --batch-size 32 \
  --reference-lr 0.01
```

Direct script execution is also supported:

```bash
python experiments/adaptive_lr.py --epochs 30 --batch-size 32 --lr 0.01
```

`--lr` and `--reference-lr` are aliases. Use either one. **They specify the LR at
`--reference-batch-size`, not necessarily the actual starting LR.** By default,
reference batch size is 32 and batch scaling uses a square root. Thus, without
warmup, reference LR 0.01 gives actual starting LR 0.01 at batch 32 and about
0.01414 at batch 64. `config.py`'s `INITIAL_LEARNING_RATE` does not supply this
entry point's default reference LR; its default is explicitly 0.01.

Enable five warmup epochs:

```bash
python -m experiments.adaptive_lr \
  --epochs 30 --batch-size 32 --lr 0.01 --warmup-epochs 5
```

At these settings, the first five training epochs use LR values 0.002, 0.004,
0.006, 0.008, and 0.010. Warmup is included in the 30-epoch budget. Omitting
`--warmup-epochs`, or setting it to 0, disables warmup.

Customize validation feedback and the output root:

```bash
python -m experiments.adaptive_lr \
  --epochs 50 --batch-size 64 --seed 42 \
  --reference-lr 0.01 --reference-batch-size 32 --alpha 0.5 \
  --lr-factor 0.5 --plateau-patience 5 --worsening-patience 3 \
  --cooldown-epochs 2 --warmup-epochs 5 \
  --min-lr 0.00001 --min-delta-loss 0.001 --min-delta-acc 0.002 \
  --target-accuracy 0.80 --output-dir results
```

For a short training check, use `--epochs 2`. This reduces the number of epochs,
not the dataset size. Batch size remains fixed throughout each run.

### Local Conda Dynamic-Library Workaround

If the local `py312` environment reports `CXXABI_1.3.15 not found`, the following
command uses that environment's C++ libraries for this process only:

```bash
LD_LIBRARY_PATH=/home/ams098z/miniforge3/envs/py312/lib \
/home/ams098z/miniforge3/envs/py312/bin/python -m experiments.adaptive_lr \
  --epochs 30 --batch-size 32 --lr 0.01 --warmup-epochs 5
```

These paths are specific to the current machine; adjust them for other systems.

## Adjustable Parameters

Command-line arguments override the defaults for the current run without editing
`config.py`. All floating-point arguments must be finite.

| Argument | Type | Default | Meaning and valid values |
| --- | --- | --- | --- |
| `--epochs` | Integer | `config.EPOCHS` | Total training epochs, including warmup; greater than 0. |
| `--batch-size` | Integer | `config.INITIAL_BATCH_SIZE` | Fixed training batch size; greater than 0. |
| `--seed` | Integer | `config.SEED` | Model/global random seed and training shuffle seed; 0 through 2^32 - 1. The shared data split still uses `config.SEED`. |
| `--reference-lr`, `--lr` | Float | `0.01` | LR at the reference batch size before feedback decay or warmup; greater than 0. |
| `--reference-batch-size` | Integer | `32` | Batch size used as the scaling reference; greater than 0. |
| `--alpha` | Float | `0.5` | Non-negative batch-scaling exponent. 0 disables scaling; 0.5 uses square-root scaling; 1 uses linear scaling. |
| `--lr-factor` | Float | `0.5` | Multiplier applied on each feedback decay; strictly between 0 and 1. |
| `--plateau-patience` | Integer | `5` | Consecutive plateau epochs needed for a decay; greater than 0. |
| `--worsening-patience` | Integer | `3` | Consecutive worsening epochs needed for a decay; greater than 0. |
| `--cooldown-epochs` | Integer | `2` | Epochs suppressing further feedback decay after a reduction; non-negative. |
| `--warmup-epochs` | Integer | `0` | Linear warmup duration; non-negative. 0 disables it. If greater than the total epoch budget, the run ends during warmup. |
| `--min-lr` | Float | `1e-5` | Non-negative LR floor, also enforced during warmup. |
| `--min-delta-loss` | Float | `1e-3` | Non-negative absolute loss threshold for improvement and worsening. |
| `--min-delta-acc` | Float | `0.002` | Accuracy improvement threshold in [0, 1]; 0.002 means 0.2 percentage points. |
| `--target-accuracy` | Float | `0.80` | Validation accuracy target in [0, 1] for time/epoch reporting. It does not stop training or control LR. |
| `--output-dir` | Path | `config.RESULTS_DIR` | Root directory for per-run artifacts; relative paths resolve from the working directory. |

## Controller Behavior

For batch size `B`, the base LR is:

```text
lr_base = reference_lr * (B / reference_batch_size) ** alpha
actual_lr = max(min_lr, lr_base * decay_multiplier * warmup_factor)
```

`decay_multiplier` starts at 1.0 and is multiplied by `lr_factor` when feedback
triggers a reduction. During warmup, `warmup_factor = min(1, epoch / warmup_epochs)`
with epochs numbered from 1. With warmup disabled, the factor is 1.

After each validation:

- Loss improvement means `val_loss < best_val_loss - min_delta_loss`.
- Accuracy improvement means `val_accuracy > best_val_accuracy + min_delta_acc`.
- Either improvement resets both patience counters. Accuracy alone never triggers a reduction.
- Without improvement, loss above `best_val_loss + min_delta_loss` counts as worsening; otherwise it counts as a plateau.
- Switching between plateau and worsening resets the other counter, so patience counts consecutive epochs.
- Reaching either patience limit triggers a decay and starts cooldown. At the LR floor, no further decay is accumulated when all groups are at the floor.
- During warmup, best metrics are tracked but feedback counters remain zero. The first adaptive decision is after validation of epoch `warmup_epochs + 1`.
- During cooldown, best metrics are tracked while feedback decay remains disabled.

The LR chosen after validation applies to the next training epoch. The batch-size
scaling factor remains constant in this experiment because batch size is fixed.
No early stopping or automatic resume is implemented by this entry point.

## Saved Results

Each invocation creates a new timestamped directory, even when the parameters
match a previous run:

```text
results/
  adaptive_lr_batch32_refLR0.01_seed42/
    <UTC timestamp>/
      <run_name>.csv
      <run_name>_metadata.json
      <run_name>_best.pt
      lr_history.json
      evaluation_history.json
      training_summary.json
      validation_metrics.json
      training_curves.png
      validation_confusion_matrix.png
      validation_confusion_matrix_normalized.png
```

- The CSV records losses, accuracies, gradient statistics, cumulative optimizer
  updates, timing, and controller decisions. `learning_rate` is the LR actually
  used in the completed epoch. `next_learning_rate` and `actual_learning_rate`
  are the LR selected after validation for the next epoch.
- `lr_history.json` stores the controller's per-epoch decisions, including
  `lr_base`, `decay_multiplier`, counters, `lr_change_reason`, `warmup_active`, and
  `next_warmup_factor`. `warmup_active` describes the completed epoch;
  `next_warmup_factor` describes the following epoch.
- Metadata records all controller settings, including warmup, plus run details
  and separate training/evaluation completion status.
- The best checkpoint is selected by the lowest validation loss. Evaluation
  reloads this checkpoint rather than using the final epoch's weights.
- `training_summary.json` reports training time, total updates, and the first
  validation epoch/time reaching the target. Unreached targets are `null`.
- Classification results include accuracy, macro precision/recall/F1, and per-class
  metrics. This entry point generates validation results, not final test results
  or a train-test generalization gap.

Trainer CSV accuracies use percentages (0–100), while evaluation JSON accuracies
and controller accuracy inputs use fractions (0–1). The pipeline converts these
units and converts cumulative update counts into per-epoch counts for evaluation.
An interrupted run may contain only partial output. All files for one run share
the same directory.


## Dataset Selection Across Experiments

All four training runners accept `--dataset cifar10` (default) or
`--dataset cifar100`. Model outputs, ordered evaluation class names, and
metadata follow the selected dataset. CIFAR-100 run names have a `cifar100_`
prefix to keep checkpoints and results separate from CIFAR-10.

The existing data modules use 20,000 training / 5,000 validation images for
CIFAR-10 and 45,000 training / 5,000 validation images for CIFAR-100. Both keep
the official 10,000-image test set separate from training and model selection.
Missing data is downloaded to `data/downloads/`.

Run from the project root in an environment with PyTorch and torchvision:

```bash
python -m experiments.fixed_fixed --dataset cifar100 --batch-sizes 32
python -m experiments.adaptive_batch --dataset cifar100
python -m experiments.adaptive_lr --dataset cifar100 --reference-lr 0.1
python -m experiments.adaptive_combined --dataset cifar100
```

The example reference LR of 0.1 matches the current shared configuration and
combined runner. For a short check, add `--epochs 3 --pilot` to adaptive batch
or combined; fixed and adaptive LR accept `--epochs 3` directly.

The reporting-only runner also accepts `--dataset cifar100`:

```bash
python -m experiments.generate_final_results --dataset cifar100
```

It requires completed validation and final-test artifacts; it does not create
them or run training. E1-E4 now save training CSV, metadata, best checkpoint,
and validation artifacts together in `results/<run_name>/<UTC timestamp>/`.
The report reader supports the new E1/E2 layout and falls back to the old flat
CSV/metadata and separate validation directories when no timestamp run root exists.
Final-test artifacts for CIFAR-100 belong in
`results/cifar100_final_test_evaluation/<timestamp>/E1/` through `E4/`.
Reports default to `results/cifar100_final_report_assets/`. E3/E4 report lookup
uses the reference LR in `config.INITIAL_LEARNING_RATE`, so use matching LRs
when generating a four-experiment report.

Adaptive batch and fixed-batch training call the shared `run_evaluation`
pipeline after training. It evaluates the best validation-loss checkpoint and
prints checkpoint accuracy and macro F1. Each run directory contains
`validation_metrics.json`, `evaluation_history.json`, `training_summary.json`,
`training_curves.png`, and `validation_confusion_matrix_normalized.png`.
The metadata records `evaluation_status` separately from training completion.
CIFAR-100 uses its ordered 100 class names; evaluation does not use the test set.

Every training invocation starts a fresh model, optimizer, Trainer, data shuffle
generator and any adaptive controllers. The chosen seed is reapplied before
initialization. No controller history or optimizer state is resumed from a previous
run. Identical configurations are allowed to run again and produce separate UTC
timestamp directories. E1/E2 retain `--overwrite` only as a compatibility flag;
it no longer overwrites existing results or controls whether training runs.
