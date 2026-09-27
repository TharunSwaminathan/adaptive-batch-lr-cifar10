
from collections import deque
import math


class AdaptiveBatchController:
    """
    Adaptive batch-size controller for the E2 experiment.

    The controller increases the training batch size when:

    1. validation loss has plateaued, and
    2. gradient-norm variation is sufficiently stable.

    Batch size only increases:

        32 -> 64 -> 128

    No batch decrease is performed in E2.
    """

    def __init__(
        self,
        batch_sizes,
        initial_batch_size,
        cv_window=3,
        stability_threshold=0.25,
        plateau_patience=3,
        min_delta=0.01,
        cooldown_epochs=2,
    ):
        if not batch_sizes:
            raise ValueError(
                "batch_sizes must not be empty."
            )

        batch_sizes = sorted(
            set(batch_sizes)
        )

        if any(
            batch_size <= 0
            for batch_size in batch_sizes
        ):
            raise ValueError(
                "All batch sizes must be positive."
            )

        if (
            initial_batch_size
            not in batch_sizes
        ):
            raise ValueError(
                "initial_batch_size must be "
                "included in batch_sizes."
            )

        if cv_window <= 0:
            raise ValueError(
                "cv_window must be positive."
            )

        if stability_threshold < 0:
            raise ValueError(
                "stability_threshold must be "
                "non-negative."
            )

        if plateau_patience <= 0:
            raise ValueError(
                "plateau_patience must be positive."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative."
            )

        if cooldown_epochs < 0:
            raise ValueError(
                "cooldown_epochs must be "
                "non-negative."
            )

        self.batch_sizes = batch_sizes

        self.current_batch_size = (
            initial_batch_size
        )

        self.cv_window = cv_window

        self.stability_threshold = (
            stability_threshold
        )

        self.plateau_patience = (
            plateau_patience
        )

        self.min_delta = min_delta

        self.cooldown_epochs = (
            cooldown_epochs
        )

        self.cv_history = deque(
            maxlen=cv_window
        )

        self.best_val_loss = float(
            "inf"
        )

        self.no_improve_epochs = 0

        self.cooldown_remaining = 0


    def _next_batch_size(self):
        """
        Return the next larger allowed batch size.

        If already at the maximum, return the
        current batch size.
        """

        current_index = (
            self.batch_sizes.index(
                self.current_batch_size
            )
        )

        if (
            current_index
            >= len(self.batch_sizes) - 1
        ):
            return self.current_batch_size

        return self.batch_sizes[
            current_index + 1
        ]


    def step(
        self,
        val_loss,
        gradient_norm_cv,
    ):
        """
        Update controller state after one completed epoch.

        Parameters
        ----------
        val_loss:
            Validation loss from the current epoch.

        gradient_norm_cv:
            Within-epoch coefficient of variation
            of minibatch gradient norms.

        Returns
        -------
        dict
            Decision information for logging and
            experiment analysis.
        """

        if not math.isfinite(
            val_loss
        ):
            raise ValueError(
                "val_loss must be finite."
            )

        if (
            not math.isfinite(
                gradient_norm_cv
            )
            or gradient_norm_cv < 0
        ):
            raise ValueError(
                "gradient_norm_cv must be a "
                "finite non-negative value."
            )

        previous_batch_size = (
            self.current_batch_size
        )

        # ---------------------------------------------
        # Update gradient-stability history
        # ---------------------------------------------

        self.cv_history.append(
            gradient_norm_cv
        )

        if (
            len(self.cv_history)
            == self.cv_window
        ):
            stability_score = (
                sum(self.cv_history)
                / len(self.cv_history)
            )

            is_stable = (
                stability_score
                <= self.stability_threshold
            )

        else:
            stability_score = None

            is_stable = False

        # ---------------------------------------------
        # Update validation plateau state
        # ---------------------------------------------

        significant_improvement = (
            val_loss
            < self.best_val_loss
            - self.min_delta
        )

        if significant_improvement:
            self.best_val_loss = (
                val_loss
            )

            self.no_improve_epochs = 0

        else:
            self.no_improve_epochs += 1

        plateau_detected = (
            self.no_improve_epochs
            >= self.plateau_patience
        )

        # ---------------------------------------------
        # Cooldown
        # ---------------------------------------------

        in_cooldown = (
            self.cooldown_remaining > 0
        )

        if in_cooldown:
            self.cooldown_remaining -= 1

        # ---------------------------------------------
        # Batch-size decision
        # ---------------------------------------------

        next_batch_size = (
            self.current_batch_size
        )

        batch_changed = False

        reason = "hold"

        at_max_batch_size = (
            self.current_batch_size
            == self.batch_sizes[-1]
        )

        if at_max_batch_size:
            reason = "maximum_batch_size"

        elif in_cooldown:
            reason = "cooldown"

        elif (
            stability_score is None
        ):
            reason = (
                "insufficient_gradient_history"
            )

        elif (
            plateau_detected
            and is_stable
        ):
            next_batch_size = (
                self._next_batch_size()
            )

            self.current_batch_size = (
                next_batch_size
            )

            batch_changed = (
                next_batch_size
                != previous_batch_size
            )

            if batch_changed:
                self.no_improve_epochs = 0

                self.cooldown_remaining = (
                    self.cooldown_epochs
                )

                reason = (
                    "plateau_and_stable"
                )

        elif not plateau_detected:
            reason = (
                "validation_improving"
            )

        elif not is_stable:
            reason = (
                "gradients_not_stable"
            )

        return {
            "previous_batch_size": (
                previous_batch_size
            ),

            "next_batch_size": (
                self.current_batch_size
            ),

            "batch_changed": (
                batch_changed
            ),

            "stability_score": (
                stability_score
            ),

            "is_stable": (
                is_stable
            ),

            "plateau_detected": (
                plateau_detected
            ),

            "no_improve_epochs": (
                self.no_improve_epochs
            ),

            "cooldown_remaining": (
                self.cooldown_remaining
            ),

            "reason": reason,
        }