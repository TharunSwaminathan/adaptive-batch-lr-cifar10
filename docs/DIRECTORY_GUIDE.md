# Directory Guide

Only active source code and one historical reference notebook are kept in this package.

```text
adaptive-batch-lr-combined-40ep-clean/
├── config.py
├── data/
│   ├── cifar10.py
│   └── cifar100.py
├── models/
│   └── custom_cnn.py
├── training/
│   ├── trainer.py
│   ├── batch_controller.py
│   └── lr_controller.py
├── evaluation/
│   ├── metrics.py
│   ├── eval_pipeline.py
│   ├── confusion_matrix.py
│   └── plots.py
├── experiments/
│   ├── datasets.py
│   ├── primary_protocol.py
│   ├── run_primary.py
│   ├── run_e1.py
│   ├── run_e2.py
│   ├── run_e3.py
│   ├── run_e4.py
│   ├── summarize_primary.py
│   └── evaluate_primary_test.py
├── tests/
│   ├── test_primary_protocol.py
│   └── test_e4_coupling.py
├── notebooks/reference/
│   └── Deep_Learning_Midterm_CIFAR100_member3_historical.ipynb
├── docs/
├── scripts/
├── results/
└── checkpoints/
```

Not included in the clean package:

- `.git/` internals from old ZIPs
- downloaded CIFAR archives/images
- `__pycache__` and `.pyc`
- old checkpoints
- old generated plots/results
- duplicated experiment runners
- Member 2 exploratory LR/batch variants that are not used in the final protocol
- recovery-only scripts from the earlier lost-checkpoint episode
