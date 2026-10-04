# CIFAR-100 Extension Status

The project now includes a second-dataset extension on CIFAR-100 using the same four-condition design as the completed CIFAR-10 study.

## Experimental setup

- Dataset: CIFAR-100
- Training split: 45,000 images (450 per class)
- Validation split: 5,000 images (50 per class)
- Official test set: 10,000 images
- Seed: 42
- Epochs: 20
- Optimizer: SGD, momentum 0.9, weight decay 5e-4
- Initial/reference batch size: 32
- Initial/reference learning rate: 0.01
- Model: `CustomCNN(num_classes=100)`

The CIFAR-100 data module uses channel statistics calculated from the original 50,000-image CIFAR-100 training partition before the internal 45k/5k split:

- mean = `(0.507075, 0.486549, 0.440918)`
- std = `(0.267334, 0.256438, 0.276150)`

## Frozen validation-stage results

| Run | Strategy | Best epoch | Best val loss | Val accuracy | Macro F1 | Optimizer updates | Time |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | Fixed batch + fixed LR | 17 | 2.7953 | 30.00% | 0.2847 | 28,140 | 444.38 s |
| E2 | Adaptive batch + fixed LR | 17 | 2.7228 | 30.96% | 0.2890 | 23,219 | 436.16 s |
| E3 | Fixed batch + adaptive LR | 20 | 2.6273 | 33.32% | 0.3152 | 28,140 | 457.03 s |
| E4 | Adaptive batch + adaptive LR | 17 | 2.6809 | 31.82% | 0.2974 | 23,219 | 442.02 s |

E2 and E4 used 17.49% fewer optimizer updates than E1. E3 produced the strongest validation metrics. All four trajectories are numerically identical through epoch 13 and diverge only when the adaptive controllers first alter the training configuration.

## Controller transitions

- E1: batch 32 and LR 0.01 for all 20 epochs.
- E2: batch 32 -> 64 after epoch 13. Batch 128 was selected after epoch 20 but never used.
- E3: LR 0.01 -> 0.005 after epoch 13; LR 0.005 -> 0.0025 after epoch 19, with 0.0025 used in epoch 20.
- E4: after epoch 13, batch 32 -> 64 and the coupled LR becomes approximately 0.007071 after square-root batch scaling and the 0.5 decay. Batch 128/LR 0.005 were selected after epoch 20 but never used.

## Test-set status

The CIFAR-100 official test set is still considered unopened for model comparison in this repository snapshot. Do not retune E1-E4. The next step is to run:

```bash
python -m experiments.evaluate_cifar100_final_test
```

The evaluator verifies the expected validation-selected checkpoint epochs before evaluating the official test set. After that test evaluation is run, its metrics are final reporting results and must not be used to retune the four experiments.

## Source materials

- `notebooks/Deep_Learning_Midterm_CIFAR100.ipynb` — teammate's CIFAR-100 experimental notebook.
- `docs/CIFAR100_Adaptive_Batch_LR_Run_Report.docx` — validation-stage report.
- `experiments/evaluate_cifar100_final_test.py` — locked final test evaluator.

The existing paper draft under `docs/drafts/` is the earlier CIFAR-10-only manuscript. It should be revised only after the frozen CIFAR-100 final test evaluation is complete.
