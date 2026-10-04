# Merge Provenance and Reproducibility Notes

This cleaned package combines the useful parts of the three team contributions without carrying forward duplicate outputs, downloaded datasets, caches, obsolete scripts, or exploratory hyperparameter variants.

## Contribution 1 — original CIFAR-10 project

Used as the foundation for the shared CNN, Trainer, gradient-norm logging, adaptive batch controller, adaptive LR controller, evaluation pipeline, and controlled E1-E4 design.

## Contribution 2 — generalized CIFAR-10/CIFAR-100 work

Used for the CIFAR-100 data module, dataset-selection abstraction, 100-class model support, and dataset-aware evaluation. Exploratory settings such as LR 0.1, E4 alpha 0.2, 5-epoch warmup, batch 256, and long experimental variants are not part of the final primary protocol.

## Contribution 3 — historical CIFAR-100 notebook

The historical notebook is retained under `notebooks/reference/`. It supplied the full-precision CIFAR-100 normalization constants and documents the earlier 20-epoch CIFAR-100 E1-E4 workflow.

The final combined primary study uses **40 epochs**, as required for the merged project. Historical 20-epoch results should therefore be described as earlier reference results rather than final 40-epoch results.

## CIFAR-100 normalization note

The retained values are:

- mean = (0.5070751592371324, 0.486548873314951, 0.4409178433670344)
- std = (0.2673342858792406, 0.25643846291708805, 0.276150471325684)

The historical notebook calculated these from the original 50,000-image CIFAR-100 training partition before making the internal 45k/5k train-validation split. This is not label leakage, but validation images contribute to preprocessing statistics. The combined code preserves these values for protocol continuity and documents the limitation rather than silently changing the historical preprocessing.

## Test-set handling

The combined data modules load the official test dataset lazily. Training and validation runs do not instantiate the test set through `get_test_loader()`.

The project history includes earlier test evaluations from prior 20-epoch work/recovery. Therefore the final report should not claim that the test sets were literally never seen by the team. The correct statement is that the final 40-epoch protocol is frozen before its final evaluation and test results are not used to tune E1-E4.
