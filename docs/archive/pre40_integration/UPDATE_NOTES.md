# Update Notes — CIFAR-10 + CIFAR-100 Project Snapshot

This snapshot consolidates the project work completed through the CIFAR-100 validation-stage extension.

## Added

- `data/cifar100.py` — reusable CIFAR-100 pipeline matching the teammate notebook.
- `notebooks/Deep_Learning_Midterm_CIFAR100.ipynb` — teammate's CIFAR-100 experiment notebook.
- `docs/CIFAR100_Adaptive_Batch_LR_Run_Report.docx` — CIFAR-100 validation-stage report.
- `docs/CIFAR100_EXTENSION_STATUS.md` — concise status, frozen results, controller transitions, and next-step protocol.
- `experiments/evaluate_cifar100_final_test.py` — locked final CIFAR-100 test evaluator.
- `docs/drafts/Adaptive_Batch_LR_Final_Paper_CIFAR10_only.*` — the earlier CIFAR-10-only manuscript retained as a draft pending CIFAR-100 final test results.

## Updated

- `README.md` now describes both CIFAR-10 and CIFAR-100 phases, final CIFAR-10 test results, and current CIFAR-100 validation status.
- `.gitignore` now ignores nested checkpoint binaries and downloaded datasets more reliably.
- `models/custom_cnn.py` documentation now reflects use on both CIFAR-10 and CIFAR-100; model behavior is unchanged.

## Important status

CIFAR-10 is complete, including final locked test evaluation.

CIFAR-100 E1-E4 validation runs are frozen. The official CIFAR-100 test set should be evaluated only with the already-selected checkpoints:

- E1: epoch 17
- E2: epoch 17
- E3: epoch 20
- E4: epoch 17

Do not retune the CIFAR-100 experiments after the test set is opened.

## Reproducibility note

The CIFAR-100 normalization statistics in the teammate notebook were calculated from the original 50,000-image CIFAR-100 training partition before the internal 45k/5k split. The reusable `data/cifar100.py` preserves those exact values so the existing runs remain reproducible.

## Packaging note

This distributed ZIP intentionally omits downloaded dataset files, `.git` history, and Python cache directories. Those files are not needed to run the source code and would greatly increase archive size.
