# Adaptive Batch Size and Learning Rate Experiments

CNN experiments on CIFAR-10 and CIFAR-100 using four combinations of fixed or
adaptive batch sizes and learning rates. Use the unified runner to execute
E1–E4 together, or launch each experiment individually.

## 1. Run the Unified Experiments

Run all commands from the project root:

```bash
cd /home/ams098z/Courses/CSCE5218/MidetermProj/adaptive-batch-lr-cifar10
```

Use a Python environment containing the project dependencies. On the current
machine, configure the existing environment with:

```bash
export PATH="/home/ams098z/miniforge3/envs/py312/bin:$PATH"
export LD_LIBRARY_PATH="/home/ams098z/miniforge3/envs/py312/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The library path addresses the possible `CXXABI_1.3.15 not found` error in this
Conda environment. Adjust these paths on other machines. Install missing
dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

### Shared Configuration

Edit [`experiments/suite_config.json`](experiments/suite_config.json).
The top-level fields control all four experiments:

```json
{
  "epochs": 120,
  "model": "deeper_cnn_wide",
  "dataset": "cifar100"
}
```

This snippet shows only the shared fields. Keep the `E1`, `E2`, `E3`, and `E4`
sections in the complete configuration file. Shared fields override fields of
the same name within individual experiment sections.

| Field | Supported values / meaning |
| --- | --- |
| `epochs` | Positive integer; shared training budget for E1–E4 |
| `dataset` | `cifar10` or `cifar100`; model output size follows the dataset |
| `model` | `custom_cnn`, `deeper_cnn`, or `deeper_cnn_wide` |

Other settings remain independently configurable for each experiment:

| Field | Purpose |
| --- | --- |
| `seed` | Model initialization and training shuffle seed; dataset splitting still uses `config.SEED` |
| `batch_size` | Fixed batch for E1/E3; initial batch for E2/E4 |
| `learning_rate` | Fixed LR for E1/E2; reference LR for E3/E4, subject to batch scaling and warmup |
| `weight_decay` | SGD weight decay |
| `momentum` | SGD momentum |
| `dropout_p` | Dropout probability for the deeper models; `custom_cnn` has no dropout |
| `batch_controller` | Independent batch growth policy for E2/E4 |
| `lr_controller` | Independent LR policy for E3/E4 |

The batch controller accepts `batch_sizes`, `cv_window`, `stability_threshold`,
`plateau_patience`, `min_delta`, and `cooldown_epochs`. The initial batch size
must appear in the candidate list.

The LR controller accepts `reference_batch_size`, `alpha`, `lr_factor`,
`plateau_patience`, `worsening_patience`, `cooldown_epochs`, `warmup_epochs`,
`min_lr`, `min_delta_loss`, and `min_delta_acc`.

### Validate and Launch

Validate settings without downloading data or starting training:

```bash
python -m experiments.run_suite --dry-run
```

Run E1 → E2 → E3 → E4 sequentially:

```bash
python -m experiments.run_suite
```

Specify a configuration file or output root:

```bash
python -m experiments.run_suite \
  --config experiments/suite_config.json \
  --output-dir results/unified_experiments
```

Each experiment runs in an isolated process. Each invocation creates a new UTC
timestamp directory. Training output goes to `E1.log` through `E4.log`; the
terminal reports experiment starts and completions. If an experiment fails,
the suite stops and retains completed results and logs. Automatic resume is
not implemented.

The suite runs E2/E4 under an independently configured protocol, marks them as
pilot runs, and permits custom budgets and initial batch sizes. Pilot mode
uses the full configured dataset. Standalone frozen-protocol checks remain
in place.

### Unified Results

```text
results/unified_experiments/<timestamp>/
  settings.json              # Resolved settings for all four experiments
  E1.log ... E4.log          # Training logs
  E1_result.json ... E4_result.json
  runs/E1/ ... runs/E4/      # Actual training and evaluation artifacts
  best_results/E1 ... E4     # Symbolic links to actual run directories
  comparison.csv            # Best epoch, validation accuracy/loss, F1, time
  figures/                  # Eight comparison PNGs across two groups
```

Comparison groups are limited to `E2_E3` and `E1_E2_E3_E4`. Each group contains
`loss_epoch.png`, `accuracy_epoch.png`, `loss_time.png`, and `accuracy_time.png`,
with the group name as the filename prefix. No combined loss/accuracy
comparison figures or PDFs are generated.

## 2. Run Individual Experiments

| Experiment | Batch policy | LR policy | Main purpose |
| --- | --- | --- | --- |
| E1 | Fixed | Fixed | Baseline |
| E2 | Adaptive growth | Fixed | Compare training efficiency and validation performance |
| E3 | Fixed | Validation feedback | Compare convergence and validation performance |
| E4 | Adaptive growth | Validation feedback | Compare the combined policies |

The examples below use CIFAR-100, the twelve-convolution CNN, seed 42, and
60 epochs. **Individual entry points do not read `suite_config.json`.** They
use command-line arguments, `config.py`, and constants in their runner modules.
The standalone E4 LR policy therefore may differ from the suite's E3/E4 policy.

### E1: Fixed Batch and Fixed LR

```bash
python -m experiments.fixed_fixed \
  --dataset cifar100 --model deeper_cnn_wide \
  --epochs 60 --seed 42 --batch-sizes 32
```

`--batch-sizes` can launch multiple runs, for example `16 32 64 128`.
Allowed values come from `config.BATCH_SIZE_OPTIONS`. Fixed LR comes from
`config.INITIAL_LEARNING_RATE`, currently `0.01`.

### E2: Adaptive Batch and Fixed LR

```bash
python -m experiments.adaptive_batch \
  --dataset cifar100 --model deeper_cnn_wide \
  --epochs 60 --seed 42 --pilot
```

The current initial batch size is 32, with candidates `32/64/128/256`.
The policy only grows batch size. Decisions use gradient-norm variability
and validation-loss stagnation. Fixed LR comes from
`config.INITIAL_LEARNING_RATE`.

Standalone batch-controller settings are defined in
`experiments/adaptive_batch.py` and are not exposed as CLI options.
Without `--pilot`, the epoch budget must equal `config.EPOCHS`.
The standalone protocol requires an initial batch size of 32.

### E3: Fixed Batch and Adaptive LR

```bash
python -m experiments.adaptive_lr \
  --dataset cifar100 --model deeper_cnn_wide \
  --epochs 60 --seed 42 --batch-size 32 \
  --reference-lr 0.01 --reference-batch-size 32 --alpha 0.2 \
  --lr-factor 0.5 --plateau-patience 5 --worsening-patience 5 \
  --cooldown-epochs 3 --warmup-epochs 0 \
  --min-lr 1e-5 --min-delta-loss 0.001 --min-delta-acc 0.002
```

This is an example candidate policy; it does not modify the suite configuration.
With reference and actual batch sizes both 32 and warmup disabled, the actual
initial LR is `0.01`. `--lr` is an alias for `--reference-lr`.

```text
lr_base = reference_lr * (batch_size / reference_batch_size) ** alpha
actual_lr = max(min_lr, lr_base * decay_multiplier * warmup_factor)
```

Significant improvement in validation loss or accuracy resets the patience
counters. Otherwise, plateau or worsening epochs accumulate. Reaching the
corresponding patience limit multiplies LR by `lr_factor`, then starts cooldown.
`min_delta_acc=0.002` requires an accuracy increase exceeding 0.2 percentage
points. The LR selected after validation applies to the following epoch.

Standalone defaults include reference batch size 256, `alpha=0.2`, and
patience 2/2; omitting these options does not reproduce the example above.
Additional options include `--output-dir` and `--target-accuracy`. The target
only controls time-to-target reporting, not early stopping or LR decisions.

### E4: Adaptive Batch and Adaptive LR

```bash
python -m experiments.adaptive_combined \
  --dataset cifar100 --model deeper_cnn_wide \
  --epochs 60 --seed 42 --pilot
```

The batch and LR controllers operate together; LR also scales with batch size.
Standalone controller settings are defined in `experiments/adaptive_combined.py`.
Current built-in settings include reference LR `0.01`, reference batch size 32,
patience 2/2, and five warmup epochs. Use the suite's E4 section for independent
control over the full policy.

Without `--pilot`, the budget must equal `config.EPOCHS`. The standalone runner
also requires initial batch size 32 and matching shared baseline/reference LRs.

Use `python -m experiments.<module_name> --help` to inspect each runner's options.
Standalone weight decay, momentum, and normalization come from `config.py`;
dropout uses model constructor defaults. JSON changes do not affect standalone
runs.

## 3. Models and Datasets

| Model key | Architecture | Default dropout |
| --- | --- | --- |
| `custom_cnn` | Three convolutions; channels 32, 64, 128 | None |
| `deeper_cnn` | Six convolutions; stages 32/64/128, two convolutions each | 0.2 |
| `deeper_cnn_wide` | Twelve convolutions; stages 64/128/256, four convolutions each | 0.2 |

All models use global average pooling (GAP) and a linear classifier.
The twelve-convolution model pools after convolutions 4, 8, and 12, reducing
spatial resolution from `32x32` to `16x16`, `8x8`, then `4x4`. Convolutions use
3x3 kernels, stride 1, and padding 1. Normalization is selected through
`config.NORMALIZATION`. Dropout follows GAP and flattening; it is enabled
in training and disabled in evaluation.

Earlier six- or eight-convolution `deeper_cnn_wide` checkpoints require their
original architectures and cannot load directly into the current model.
Models are registered in `models/factory.py`; dataset selection is defined
in `experiments/datasets.py`.

| Dataset key | Training / validation samples | Official test samples |
| --- | --- | --- |
| `cifar10` | 20,000 / 5,000 | 10,000 |
| `cifar100` | 45,000 / 5,000 | 10,000 |

Data downloads automatically to `data/downloads/` on first use. Training uses
random crops and horizontal flips; validation uses no random augmentation.
The official test set is excluded from training, LR feedback, and model selection.

## 4. Best Checkpoints and Evaluation

Standalone runs create `results/<run_name>/<UTC timestamp>/`. Suite runs store
these artifacts under their unified output directory. Main files include:

- `<run_name>_best.pt`: lowest-validation-loss checkpoint, including model and optimizer state, epoch, and metadata.
- `<run_name>.csv` and `<run_name>_metadata.json`: training records and actual configuration.
- `validation_metrics.json`: accuracy, loss, and class metrics evaluated after reloading the best checkpoint.
- `evaluation_history.json` and `training_summary.json`: full history, timing, and target reporting.
- `lr_history.json` for E3/E4: per-epoch LR decisions; `batch_history.json` for E2: batch decisions.

The best checkpoint need not have the highest accuracy or be from the final
epoch. Each invocation starts fresh training; automatic early stopping is not
implemented. CSV accuracies use percentages, while evaluation JSON uses
fractions. CSV `learning_rate` records the completed epoch's LR;
`next_learning_rate` records the next epoch's selected LR.

Individual evaluations save separate PNG curves, combined loss/accuracy PNGs,
and a normalized confusion matrix. Cross-experiment plotting produces only the
eight separate PNGs described above. Curves are unsmoothed and not extrapolated;
time axes use measured cumulative elapsed time. Regenerate comparisons with:

```bash
python -m experiments.plot_comparison \
  --results-root results/unified_experiments/<timestamp>
```

The directory must contain `evaluation_history.json` under each
`best_results/E1` through `best_results/E4` directory. Regeneration does not
remove existing legacy figures or PDFs.

For controlled comparisons, keep the model, data split, seed, regularization,
and training budget consistent where appropriate. Freeze settings before
final test evaluation; the best validation score after repeated tuning is not
a final test result.

## 5. Historical Utilities

`experiments.rerun_cifar100` provides an earlier parameter-screening workflow.
`experiments.generate_final_results` builds reports from existing validation/test
artifacts. `experiments.evaluate_selected` evaluates historically selected
checkpoints on the test set. These utilities contain historical architecture,
directory, or protocol assumptions and are not the recommended entry points
for the current twelve-convolution suite. Current suite outputs should not be
assumed compatible with those utilities without checking their requirements.

See [`evaluation/README.md`](evaluation/README.md) for detailed evaluation APIs.
