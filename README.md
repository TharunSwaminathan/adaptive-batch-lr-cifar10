# Adaptive Batch Size and Learning Rate Scheduling for Efficient CNN Training

This is the cleaned, combined team codebase for the final project. It supports the same controlled E1-E4 design on CIFAR-10 and CIFAR-100 and uses a **40-epoch primary budget** for every experiment.

## Primary experiments

| Experiment | Batch size | Learning rate |
|---|---|---|
| E1 | Fixed | Fixed |
| E2 | Adaptive | Fixed |
| E3 | Fixed | Adaptive |
| E4 | Adaptive | Adaptive |

The complete frozen settings live in `experiments/primary_protocol.py`. E4 has its own complete configuration. Its batch controller acts first, and the selected `next_batch_size` is passed to the LR adapter before the next epoch.

## Install

Install the PyTorch/torchvision build appropriate for the machine and CUDA version first. Then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Validate the code before training

```bash
python -m experiments.primary_protocol
python -m unittest discover -s tests -v
```

## Run primary experiments

CIFAR-10:

```bash
python -m experiments.run_e1 --dataset cifar10
python -m experiments.run_e2 --dataset cifar10
python -m experiments.run_e3 --dataset cifar10
python -m experiments.run_e4 --dataset cifar10
```

CIFAR-100:

```bash
python -m experiments.run_e1 --dataset cifar100
python -m experiments.run_e2 --dataset cifar100
python -m experiments.run_e3 --dataset cifar100
python -m experiments.run_e4 --dataset cifar100
```

Each primary run executes exactly 40 epochs. The trainer separately reports `Epochs completed: 40` and `Best checkpoint epoch: X` so checkpoint selection is not confused with the training budget.

For a short development smoke test, use `--pilot-epochs`, for example:

```bash
python -m experiments.run_e4 --dataset cifar10 --pilot-epochs 2
```

Pilot outputs are kept separate from primary outputs.

## Summarize validation runs

After E1-E4 complete:

```bash
python -m experiments.summarize_primary --dataset cifar10
python -m experiments.summarize_primary --dataset cifar100
```

This reads all four per-run manifests and creates one aggregate summary without the old single-run overwrite bug.

## Final test evaluation

Only after all four 40-epoch primary runs are completed and frozen:

```bash
python -m experiments.evaluate_primary_test --dataset cifar10
python -m experiments.evaluate_primary_test --dataset cifar100
```

The evaluator validates all four completed runs before it creates the official test loader. It refuses a repeat final test by default.

## Generated output layout

```text
results/
  primary/<dataset>/<E1-E4>/<run_id>/
  pilots/<dataset>/<E1-E4>/<run_id>/
  final_test/<dataset>/<timestamp>/

checkpoints/
  primary/<dataset>/<E1-E4>/<run_id>/
  pilots/<dataset>/<E1-E4>/<run_id>/
```

Generated results, checkpoints, downloaded datasets, caches, and IDE files are excluded from source control.

See `docs/EXPERIMENT_PROTOCOL.md`, `docs/PROVENANCE.md`, and `docs/DIRECTORY_GUIDE.md` for the merge decisions and experiment details.
