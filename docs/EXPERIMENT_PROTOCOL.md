# Combined Primary Experiment Protocol

The final combined project uses a **40-epoch** primary budget for every E1-E4 run. All four experiments use seed 42, SGD, momentum 0.9, weight decay 5e-4, an initial batch size of 32, and an initial/reference learning rate of 0.01.

## E1 — Fixed batch + fixed learning rate

- Epochs: 40
- Batch: 32 for all epochs
- LR: 0.01 for all epochs

## E2 — Adaptive batch + fixed learning rate

- Epochs: 40
- Initial batch: 32
- Allowed batches: 32 -> 64 -> 128
- Grow-only policy
- 3-epoch rolling mean of within-epoch gradient-norm CV
- Stability threshold: 0.25
- Validation-loss plateau patience: 3
- Minimum significant loss improvement: 0.01
- Batch cooldown: 2 epochs
- LR remains 0.01

## E3 — Fixed batch + adaptive learning rate

- Epochs: 40
- Batch: 32
- Reference LR: 0.01 at reference batch 32
- Batch-scaling exponent alpha: 0.5
- LR decay factor: 0.5
- Plateau patience: 5
- Worsening patience: 3
- LR cooldown: 2 epochs
- Warmup: 0
- Minimum LR: 1e-5
- Loss delta: 0.001
- Accuracy delta: 0.002

## E4 — Adaptive batch + adaptive learning rate

E4 has its own complete configuration. It is not defined in code by merging E2 and E3 dictionaries.

### E4 batch controller

- Epochs: 40
- Initial batch: 32
- Allowed batches: 32 -> 64 -> 128
- Grow-only policy
- CV window: 3
- Stability threshold: 0.25
- Plateau patience: 3
- Minimum significant loss improvement: 0.01
- Cooldown: 2 epochs

### E4 LR controller

- Reference LR: 0.01
- Reference batch: 32
- Alpha: 0.5
- Decay factor: 0.5
- Plateau patience: 5
- Worsening patience: 3
- Cooldown: 2 epochs
- Warmup: 0
- Minimum LR: 1e-5
- Loss delta: 0.001
- Accuracy delta: 0.002

### E4 controller coupling

At the end of epoch t:

1. The batch controller receives validation loss and gradient-norm CV.
2. It selects `next_batch_size`.
3. The LR controller receives validation loss, validation accuracy, and that **next batch size**.
4. The LR controller computes the learning rate for epoch t+1 using:

`LR_base = reference_lr * (next_batch_size / reference_batch_size) ** alpha`

5. Any persistent decay multiplier is then applied.

For batch 64, alpha 0.5, and no decay:

`0.01 * sqrt(64/32) = 0.0141421356`

With a 0.5 decay multiplier:

`0.0141421356 * 0.5 = 0.0070710678`

The automated coupling test confirms that `Trainer` passes `next_batch_size`, not the old batch size, to the LR controller.
