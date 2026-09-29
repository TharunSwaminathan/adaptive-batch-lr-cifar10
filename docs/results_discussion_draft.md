# Results and Discussion — Draft

## Experimental comparison

Four primary experiments were evaluated using the same CIFAR-10 subset, CNN architecture, seed, optimizer family, and 20-epoch training budget. E1 used a fixed batch size and fixed learning rate. E2 adapted batch size while keeping the learning rate fixed. E3 kept the batch size fixed while adapting the learning rate. E4 combined both controllers. Model checkpoints were selected using validation loss, and the official CIFAR-10 test set was evaluated only after the E1–E4 configurations were frozen.

| Experiment | Strategy | Best validation accuracy | Test accuracy | Test macro F1 | Optimizer updates |
|---|---|---:|---:|---:|---:|
| E1 | Fixed batch + fixed LR | 57.82% | 56.60% | 0.5577 | 12,500 |
| E2 | Adaptive batch + fixed LR | 62.08% | 62.15% | 0.6089 | 10,316 |
| E3 | Fixed batch + adaptive LR | 60.86% | 60.35% | 0.5964 | 12,500 |
| E4 | Adaptive batch + adaptive LR | 62.04% | 61.70% | 0.6020 | 10,316 |

The fixed baseline reached 56.60% test accuracy. Adaptive batch sizing produced the largest improvement observed in this run. E2 reached 62.15% test accuracy, an increase of 5.55 percentage points over E1, while reducing the number of optimizer updates from 12,500 to 10,316. This corresponds to 2,184 fewer updates, or about a 17.5% reduction.

Adaptive learning rate alone also improved the result. E3 reached 60.35% test accuracy, 3.75 percentage points above the baseline. The learning-rate controller retained 0.01 for most of training and reduced it to 0.005 late in the run after the validation-loss worsening condition was satisfied. The epoch trained at the reduced learning rate became the selected checkpoint.

The combined E4 strategy reached 61.70% test accuracy while also using 10,316 optimizer updates. Its batch controller increased the batch size from 32 to 64 after epoch 13, once the validation plateau and gradient-stability conditions were both satisfied. Because E4 couples batch size and learning rate, that batch increase changed the learning rate from 0.01 to approximately 0.014142 using the configured square-root scaling rule. The controller retained that rate for the remainder of the run.

## Validation-to-test consistency

Validation and test performance were reasonably close for all four experiments. E1 had a 1.22 percentage-point validation-to-test difference, E2 differed by only -0.07 percentage points, E3 by 0.51 percentage points, and E4 by 0.34 percentage points. This consistency suggests that the validation-selected checkpoints transferred to the official test set without a large performance drop in this single-seed experiment.

The E2 and E4 test accuracies differed by only 0.45 percentage points. Since the primary comparison currently contains one seed, that small difference should not be interpreted as evidence that E2 is generally better than E4. A repeated-seed study would be needed before making a stronger claim about the relative ordering of the two adaptive strategies.

## Efficiency interpretation

Optimizer-update count gives a useful comparison of training effort because increasing the batch size reduces the number of parameter updates required to process the same number of training examples. Both adaptive-batch experiments used 10,316 updates, compared with 12,500 for E1 and E3. This is a 17.5% reduction.

Wall-clock time should be interpreted more carefully. E1, E3, and E4 were all around 11 minutes on the recorded system, while the earlier E2 run was substantially slower despite requiring fewer optimizer updates. Because runtime can be affected by system load, data loading, hardware state, and other implementation-level effects, the present results do not support a general claim that adaptive batch sizing reduced wall-clock time. The update-count reduction is the cleaner efficiency observation from these runs.

## Relationship to large-minibatch SGD literature

Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, showed that large-minibatch training can retain accuracy when optimization issues are handled appropriately. Their work used a linear learning-rate scaling rule as minibatch size increased and introduced learning-rate warmup to stabilize the early phase of large-batch training.

The present project is related to that idea but does not reproduce the same training recipe. Here, batch size changes dynamically in response to validation behavior and gradient-norm stability rather than being increased primarily for distributed scaling. The combined E4 controller uses a square-root scaling exponent of 0.5 rather than the linear scaling rule, and warmup is disabled in the frozen primary experiment. For that reason, the Goyal et al. paper should be used as motivation for coordinating batch size and learning rate, not as a claim that the present controller implements their method directly.

The E4 result gives some support to the broader idea that batch size and learning rate should not be treated as completely independent hyperparameters. When the controller changed the batch size from 32 to 64, the learning rate was immediately adjusted as part of the same decision process. However, under the present 20-epoch budget and seed, the combined strategy did not produce an additional accuracy gain beyond adaptive batch sizing alone.

## Main finding

For the current experiment, adaptive batch sizing accounted for most of the improvement over the fixed baseline. Adaptive learning rate also improved performance, and the combined controller preserved most of the adaptive-batch gain while reducing optimizer updates. The results support adaptive control as a useful direction, but the single-seed design and relatively small CIFAR-10 training subset limit how broadly the numerical differences can be generalized.

## Limitations and next reporting steps

The main limitation is that the primary E1–E4 comparison uses one random seed. The current results therefore show what happened under the frozen seed-42 configuration rather than an estimate of mean performance across repeated runs. If time allows, repeated runs should be treated as a separate robustness study rather than as an opportunity to retune the already-opened test-set experiment.

For the final report, the most useful figures are the validation/test accuracy comparison, validation/test macro-F1 comparison, optimizer-update comparison, validation-accuracy curves, gradient-CV curves, adaptive batch-size schedule, and adaptive learning-rate schedule. The confusion matrices generated during final test evaluation can be used to discuss class-level behavior without changing the selected models.
