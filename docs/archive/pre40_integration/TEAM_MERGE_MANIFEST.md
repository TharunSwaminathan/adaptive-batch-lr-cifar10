# Combined Team Project Merge Manifest

This package combines the current canonical project with the additional work
provided by the other team member, while avoiding destructive overwrites of
already-frozen experiments.

## Canonical project root

The files at the package root are the version to use for the final project.
They include:

- completed CIFAR-10 E1-E4 training, validation, and locked final test results;
- fixed-batch ablations and reporting assets;
- CIFAR-100 data pipeline and teammate validation-stage notebook/report;
- the CIFAR-100 locked final-test workflow;
- the lost-checkpoint recovery workflow, because the original CIFAR-100
  best-validation checkpoints are no longer available;
- final-results reporting scripts and the CIFAR-10-only paper draft.

The newest CIFAR-100 recovery files are:

- `experiments/rebuild_cifar100_primary.py`
- `experiments/evaluate_cifar100_final_test.py`
- `docs/CIFAR100_CHECKPOINT_RECOVERY_README.txt`

## Team Member 3 work

The supplied Team Member 3 archive has been preserved in full (except `.git`,
downloaded datasets, and Python cache files) under:

`team_contributions/member3_snapshot/`

It includes source changes, additional checkpoints, and exploratory result
artifacts. It is intentionally **not overlaid on the canonical source tree**.
The member's branch used different project-wide defaults (including LR 0.1 and
40 epochs) and includes exploratory CIFAR-10/CIFAR-100 runs. Overwriting the
canonical source with those files would silently change the frozen E1-E4 study.

The snapshot contains 127 result artifact files and 20 `.pt`
checkpoint files.

Notable Team Member 3 contributions include:

- CIFAR-100 preprocessing work;
- dataset-selection support (`experiments/datasets.py`) in the member snapshot;
- adaptive-LR warmup/controller experimentation;
- multiple CIFAR-10 LR/batch exploratory runs;
- CIFAR-100 adaptive-LR runs at several reference learning rates and batch 256;
- CIFAR-100 combined adaptive batch/LR exploratory results;
- additional fixed-setting checkpoints.

## Why source conflicts were preserved rather than overwritten

The team branches diverged. Some filenames are the same but represent different
experimental assumptions. The canonical root therefore keeps the frozen study
used for the current paper/results, while the member snapshot preserves the
other member's work for review, comparison, and selective integration.

This is especially important because the official CIFAR-100 E1-E4 checkpoints
from the earlier validation report were lost. The correct recovery is to rerun
the frozen canonical settings and select new best-validation checkpoints before
opening the CIFAR-100 test set; exploratory Team Member 3 CIFAR-100 checkpoints
must not be substituted for those official reruns.

## Excluded non-project/runtime material

To keep the combined archive portable, the package omits:

- `.git/` internals from all source archives;
- downloaded CIFAR dataset binaries/tarballs;
- `__pycache__` files.

Existing experiment checkpoints/results that are actual project artifacts are
retained. Dataset files can be downloaded automatically by torchvision.

## Recommended next action

1. Use the canonical root, not the Team Member 3 source snapshot, for official runs.
2. Compile the two recovery scripts.
3. Commit them in the working Git repository if not already committed.
4. Run `python -m experiments.rebuild_cifar100_primary` to reconstruct the lost
   official CIFAR-100 checkpoints without touching the test set.
5. Review the new validation summary before running the locked CIFAR-100 test evaluator.
