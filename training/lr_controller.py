"""Epoch-level learning-rate control based on validation loss."""

import math


class LRController:
    """Keep LR fixed or reduce it when validation loss stops improving.

    Call step(val_loss) once after each epoch's validation. The first loss
    establishes a baseline. In adaptive mode, patience consecutive epochs
    without an improvement greater than min_delta trigger a reduction.
    For example, patience=3 reduces LR on the third bad epoch.

    All optimizer parameter groups are supported. Batch size is unchanged.
    Fixed mode leaves the optimizer's learning rates untouched.
    """

    def __init__(
        self,
        optimizer,
        mode="adaptive",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        min_delta=1e-4,
    ):
        if mode not in ("fixed", "adaptive"):
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        if not math.isfinite(factor) or not 0 < factor < 1:
            raise ValueError("factor must be between 0 and 1 (exclusive)")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(min_lr) or min_lr < 0:
            raise ValueError("min_lr must be finite and non-negative")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if not optimizer.param_groups:
            raise ValueError("optimizer must have at least one parameter group")
        for group in optimizer.param_groups:
            lr = float(group["lr"])
            if not math.isfinite(lr) or lr < 0:
                raise ValueError("optimizer learning rates must be finite and non-negative")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best_loss = None
        self.bad_epochs = 0

    def get_lrs(self):
        """Return the current LR of every optimizer parameter group."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, val_loss):
        """Update LR for the next epoch; return whether any LR decreased.

        val_loss must be the finite, sample-averaged validation loss.
        NaN/inf are rejected before changing the controller or optimizer.
        Record get_lrs() before this call for the epoch just completed,
        or after this call for the next epoch's learning rates.
        """
        val_loss = float(val_loss)
        if not math.isfinite(val_loss):
            raise ValueError("val_loss must be finite")

        if self.mode == "fixed":
            return False

        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return False

        changed = False
        for group in self.optimizer.param_groups:
            old_lr = float(group["lr"])
            # Never raise LR, even if a group is already below min_lr.
            new_lr = min(old_lr, max(old_lr * self.factor, self.min_lr))
            if new_lr < old_lr:
                group["lr"] = new_lr
                changed = True

        self.bad_epochs = 0
        return changed
