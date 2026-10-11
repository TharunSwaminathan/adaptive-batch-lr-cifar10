"""Fixed learning rates or batch-scaled LR with validation feedback."""

import math


class LRController:
    """Control optimizer learning rates without changing batch size.

    Adaptive LR = reference_lr * (batch_size / reference_batch_size)**alpha
    times a persistent decay_multiplier, with a per-group min_lr floor.
    Accuracy inputs are fractions in [0, 1]. Loss is the primary signal.

    Create once per run. Before the first epoch, call set_batch_size(B) to
    apply the initial batch-scaled LR. After each validation, call update
    with the batch size selected for the NEXT epoch. Its returned LR and
    last_info describe that next epoch, not the epoch just completed.

    Optional warmup uses target_lr * min(1, epoch / warmup_epochs), clipped
    to min_lr. Zero disables it. First W epochs track best metrics but do
    not accumulate plateau/worsening evidence or trigger feedback decay.
    The first adaptive decision is after validation of epoch W+1.

    Fixed mode preserves existing optimizer LRs, including when batch size
    changes. The original fixed-mode step(val_loss) call still returns False
    and rejects non-finite loss without changing optimizer/controller state.

    Multiple parameter groups retain their initial LR ratios before floors
    are applied; reference_lr refers to the first group. Adaptive mode
    requires strictly positive initial group LRs.
    """

    def __init__(
        self,
        optimizer,
        mode="adaptive",
        factor=0.5,
        patience=None,
        min_lr=None,
        min_delta=None,
        *,
        reference_batch_size=32,
        reference_lr=0.01,
        alpha=0.5,
        lr_factor=None,
        plateau_patience=5,
        worsening_patience=3,
        cooldown_epochs=2,
        warmup_epochs=0,
        min_delta_loss=1e-3,
        min_delta_acc=0.002,
    ):
        # Keep old keyword names as aliases. Fixed-mode defaults are unchanged.
        if mode not in ("fixed", "adaptive"):
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        self._finite(factor, "factor", positive=True)
        if factor >= 1:
            raise ValueError("factor must be less than 1")
        if patience is not None:
            self._integer(patience, "patience", minimum=1)
            plateau_patience = patience
        if min_delta is not None:
            self._finite(min_delta, "min_delta")
            min_delta_loss = min_delta
        if min_lr is None:
            min_lr = 1e-6 if mode == "fixed" else 1e-5
        self._finite(min_lr, "min_lr")
        if not optimizer.param_groups:
            raise ValueError("optimizer must have at least one parameter group")
        initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        for lr in initial_lrs:
            self._finite(lr, "optimizer learning rate", positive=mode == "adaptive")

        self.optimizer = optimizer
        self.mode = mode
        self.min_lr = min_lr
        # Legacy fixed-mode attributes retained for compatibility.
        self.factor = factor
        self.patience = 3 if patience is None else patience
        self.min_delta = 1e-4 if min_delta is None else min_delta
        self.best_loss = None
        self.bad_epochs = 0
        if mode == "fixed":
            return

        self._integer(reference_batch_size, "reference_batch_size", minimum=1)
        self._finite(reference_lr, "reference_lr", positive=True)
        self._finite(alpha, "alpha")
        lr_factor = factor if lr_factor is None else lr_factor
        self._finite(lr_factor, "lr_factor", positive=True)
        if lr_factor >= 1:
            raise ValueError("lr_factor must be less than 1")
        self._integer(plateau_patience, "plateau_patience", minimum=1)
        self._integer(worsening_patience, "worsening_patience", minimum=1)
        self._integer(cooldown_epochs, "cooldown_epochs", minimum=0)
        self._integer(warmup_epochs, "warmup_epochs", minimum=0)
        self._finite(min_delta_loss, "min_delta_loss")
        self._finite(min_delta_acc, "min_delta_acc")

        self.reference_lr = reference_lr
        self.reference_batch_size = reference_batch_size
        self.alpha = alpha
        self.lr_factor = lr_factor
        self.plateau_patience = plateau_patience
        self.worsening_patience = worsening_patience
        self.cooldown_epochs = cooldown_epochs
        self.warmup_epochs = warmup_epochs
        self.min_delta_loss = min_delta_loss
        self.min_delta_acc = min_delta_acc
        self.group_ratios = [lr / initial_lrs[0] for lr in initial_lrs]
        self.best_val_loss = None
        self.best_val_accuracy = None
        self.plateau_counter = 0
        self.worsening_counter = 0
        self.cooldown_counter = 0
        self.decay_multiplier = 1.0
        self.lr_history = []
        self.last_info = None

    @staticmethod
    def _finite(value, name, positive=False):
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")

    @staticmethod
    def _integer(value, name, minimum):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def get_lrs(self):
        """Return all parameter-group LRs; this does not change any state."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _base_lr(self, batch_size):
        self._integer(batch_size, "batch_size", minimum=1)
        base = self.reference_lr * (batch_size / self.reference_batch_size) ** self.alpha
        self._finite(base, "computed base LR", positive=True)
        for ratio in self.group_ratios:
            self._finite(base * ratio, "computed group base LR", positive=True)
        return base

    def _warmup_factor(self, epoch):
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    def _apply_lr(self, base, epoch=None):
        if epoch is None:
            epoch = len(self.lr_history) + 1
        warmup_factor = self._warmup_factor(epoch)
        for group, ratio in zip(self.optimizer.param_groups, self.group_ratios):
            group["lr"] = max(
                self.min_lr, base * ratio * self.decay_multiplier * warmup_factor,
            )
        return self.get_lrs()[0]

    def set_batch_size(self, batch_size):
        """Apply batch scaling without consuming validation or cooldown steps.

        Use before the first epoch. Existing decay history is preserved.
        Fixed mode is a no-op. Returns the first parameter group's LR.
        """
        if self.mode == "fixed":
            return self.get_lrs()[0]
        return self._apply_lr(self._base_lr(batch_size))

    def update(self, val_loss, val_accuracy=None, batch_size=None):
        """Process one validation result and return the next epoch's first LR.

        Loss > best_loss + min_delta_loss is worsening; otherwise a loss
        that does not significantly improve is plateauing. Accuracy progress
        resets both counters. Changing between plateau and worsening resets
        the other counter, so patience always counts consecutive epochs.

        Cooldown suppresses feedback decay for exactly cooldown_epochs calls,
        but still tracks best metrics and applies batch scaling. At the LR
        floor, no extra decay is accumulated when all groups are at the floor.
        """
        val_loss = float(val_loss)
        if not math.isfinite(val_loss):
            raise ValueError("val_loss must be finite")
        if self.mode == "fixed":
            return self.get_lrs()[0]
        if val_accuracy is None or batch_size is None:
            raise ValueError("adaptive mode requires val_accuracy and batch_size")
        val_accuracy = float(val_accuracy)
        if not math.isfinite(val_accuracy) or not 0 <= val_accuracy <= 1:
            raise ValueError("val_accuracy must be a fraction in [0, 1]")
        base = self._base_lr(batch_size)

        loss_improved = self.best_val_loss is None or val_loss < self.best_val_loss - self.min_delta_loss
        acc_improved = self.best_val_accuracy is None or val_accuracy > self.best_val_accuracy + self.min_delta_acc
        if loss_improved:
            self.best_val_loss = val_loss
        if acc_improved:
            self.best_val_accuracy = val_accuracy

        epoch = len(self.lr_history) + 1
        reason = "keep"
        if epoch <= self.warmup_epochs:
            # Track best metrics, but do not accumulate decay evidence during warmup.
            self.plateau_counter = self.worsening_counter = 0
            reason = "warmup"
        elif self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.plateau_counter = self.worsening_counter = 0
            reason = "cooldown"
        elif loss_improved or acc_improved:
            self.plateau_counter = self.worsening_counter = 0
        elif val_loss > self.best_val_loss + self.min_delta_loss:
            self.worsening_counter += 1
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
            self.worsening_counter = 0

        trigger = None
        if self.plateau_counter >= self.plateau_patience:
            trigger = "plateau_decay"
        elif self.worsening_counter >= self.worsening_patience:
            trigger = "worsening_decay"
        if trigger is not None:
            above_floor = any(base * ratio * self.decay_multiplier > self.min_lr for ratio in self.group_ratios)
            if above_floor:
                self.decay_multiplier *= self.lr_factor
                self.cooldown_counter = self.cooldown_epochs
                reason = trigger
            self.plateau_counter = self.worsening_counter = 0

        lr = self._apply_lr(base, epoch=epoch + 1)
        self.last_info = {
            "epoch": epoch,
            "warmup_active": epoch <= self.warmup_epochs,
            "next_warmup_factor": self._warmup_factor(epoch + 1),
            "batch_size": batch_size,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "lr_base": base,
            "decay_multiplier": self.decay_multiplier,
            "actual_learning_rate": lr,
            "lrs": self.get_lrs(),
            "plateau_counter": self.plateau_counter,
            "worsening_counter": self.worsening_counter,
            "cooldown_counter": self.cooldown_counter,
            "lr_change_reason": reason,
        }
        self.lr_history.append(dict(self.last_info))
        return lr

    def step(self, val_loss, val_accuracy=None, batch_size=None):
        """Return whether any LR decreased; fixed step(val_loss) is unchanged.

        Adaptive mode now needs accuracy and batch size. For the numerical LR
        and diagnostic reasons, use update() and last_info instead.
        """
        old_lrs = self.get_lrs()
        self.update(val_loss, val_accuracy, batch_size)
        return any(new < old for new, old in zip(self.get_lrs(), old_lrs))
