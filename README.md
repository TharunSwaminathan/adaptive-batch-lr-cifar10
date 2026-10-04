# Adaptive Batch Size and Learning Rate Scheduling for Efficient CNN Training

This repository contains a controlled study of adaptive batch-size and learning-rate scheduling for CNN training. The original study uses CIFAR-10, and the project now includes a CIFAR-100 extension using the same E1-E4 experimental design.

## Research question

Can batch size and learning rate be adapted automatically during CNN training to improve the accuracy-efficiency tradeoff compared with fixed training settings?

## Experimental design

| Experiment | Batch size | Learning rate |
|---|---|---|
| E1 | Fixed | Fixed |
| E2 | Adaptive | Fixed |
| E3 | Fixed | Adaptive |
| E4 | Adaptive | Adaptive |

The adaptive batch controller uses validation-loss plateau behavior together with a 3-epoch rolling mean of within-epoch gradient-norm coefficient of variation. The adaptive LR controller responds to validation behavior and, in E4, also reacts to changes in batch size.

## CIFAR-10 status: complete

CIFAR-10 uses a fixed stratified 20,000-image training subset, 5,000-image validation subset, and the full official 10,000-image test set. The final locked test evaluation has been completed.

| Run | Test accuracy | Test macro F1 | Optimizer updates |
|---|---:|---:|---:|
| E1 | 56.60% | 0.5577 | 12,500 |
| E2 | **62.15%** | **0.6089** | 10,316 |
| E3 | 60.35% | 0.5964 | 12,500 |
| E4 | 61.70% | 0.6020 | 10,316 |

Under the frozen seed-42 CIFAR-10 setting, E2 improved test accuracy by 5.55 percentage points over E1 while using 17.5% fewer optimizer updates. The small E2-E4 difference should not be treated as evidence of general superiority because the primary comparison uses one seed.

## CIFAR-100 extension: validation complete, final test pending

The CIFAR-100 extension uses a stratified 45,000/5,000 train-validation split and the same controller logic.

| Run | Validation accuracy | Macro F1 | Optimizer updates |
|---|---:|---:|---:|
| E1 | 30.00% | 0.2847 | 28,140 |
| E2 | 30.96% | 0.2890 | 23,219 |
| E3 | **33.32%** | **0.3152** | 28,140 |
| E4 | 31.82% | 0.2974 | 23,219 |

E2 and E4 used 17.49% fewer optimizer updates than E1, while E3 produced the strongest validation metrics. The official CIFAR-100 test comparison remains locked and must be run only with the already-selected validation checkpoints.

See `docs/CIFAR100_EXTENSION_STATUS.md` for the full extension status and controller transitions.

## Repository structure

```text
data/
  cifar10.py
  cifar100.py
models/
  custom_cnn.py
training/
  trainer.py
  batch_controller.py
  lr_controller.py
evaluation/
  metrics.py
  confusion_matrix.py
  eval_pipeline.py
  plots.py
experiments/
  fixed_fixed.py
  adaptive_batch_fixed.py
  adaptive_lr.py
  adaptive_combined.py
  evaluate_final_test.py
  evaluate_cifar100_final_test.py
  generate_final_results.py
notebooks/
  Deep_Learning_Midterm_CIFAR100.ipynb
docs/
  results_discussion_draft.md
  CIFAR100_EXTENSION_STATUS.md
  CIFAR100_Adaptive_Batch_LR_Run_Report.docx
```

## Reproducibility

The primary experiments use seed 42. Python, NumPy, PyTorch, and CUDA seeds are set where applicable. CUDA deterministic behavior is enabled and cuDNN benchmarking is disabled. Group Normalization is used so normalization statistics do not depend on minibatch size.

PyTorch and torchvision are intentionally not pinned in `requirements.txt` because installation depends on the target CUDA/CPU environment. Install the correct PyTorch build first, then install the remaining requirements.

## Final locked CIFAR-100 test evaluation

Before running the CIFAR-100 test evaluation, confirm the validation-selected checkpoints exist under:

```text
checkpoints/cifar100/E1/
checkpoints/cifar100/E2/
checkpoints/cifar100/E3/
checkpoints/cifar100/E4/
```

Then run:

```bash
python -m py_compile experiments/evaluate_cifar100_final_test.py
python -m experiments.evaluate_cifar100_final_test
```

Expected checkpoint epochs are E1=17, E2=17, E3=20, and E4=17. The evaluator refuses to proceed if an unexpected checkpoint is supplied.

After viewing the CIFAR-100 test results, do not retune E1-E4.
