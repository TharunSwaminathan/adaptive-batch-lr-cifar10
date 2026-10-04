CIFAR-100 LOST-CHECKPOINT RECOVERY

Situation:
The original CIFAR-100 best-validation checkpoints no longer exist.
The official CIFAR-100 test set has not been evaluated.

This package:
1. creates experiments\rebuild_cifar100_primary.py
2. replaces experiments\evaluate_cifar100_final_test.py

WHY THE EVALUATOR CHANGED
The old evaluator hardcoded the historical best epochs (17, 17, 20, 17).
Because the original checkpoint files are gone, the new deterministic reruns
must become the official CIFAR-100 validation-stage runs. If their best epochs
differ slightly, the evaluator must verify the new validation-selected epochs,
not force the historical ones.

STEP 1 — copy both files into the repo.

STEP 2 — compile:
    python -m py_compile experiments\rebuild_cifar100_primary.py
    python -m py_compile experiments\evaluate_cifar100_final_test.py

STEP 3 — commit the recovery protocol BEFORE training:
    git add experiments\rebuild_cifar100_primary.py
    git add experiments\evaluate_cifar100_final_test.py
    git commit -m "Add CIFAR-100 checkpoint reconstruction protocol"
    git push origin tharun
    git status

STEP 4 — rebuild all four frozen experiments:
    python -m experiments.rebuild_cifar100_primary

This runs E1, E2, E3, and E4 for 20 epochs each using the same frozen settings.
It performs validation evaluation only. It does NOT evaluate the official test
set.

Expected output folders:
    checkpoints\cifar100\E1 ... E4
    results\cifar100\E1 ... E4

Summary files:
    results\cifar100\cifar100_official_rerun_summary.json
    results\cifar100\cifar100_official_rerun_summary.csv

IMPORTANT:
If the new results differ from the old teammate report, DO NOT tune parameters
to reproduce the historical numbers. The fresh reruns become the official
validation-stage record.

STEP 5 — send the complete rerun summary/output for review.

ONLY AFTER REVIEW:
    python -m experiments.evaluate_cifar100_final_test
