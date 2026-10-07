import json
import math
import time
from pathlib import Path

import pandas as pd
import torch


class Trainer:
    """
    Shared training and validation engine.

    This trainer is used by the fixed baseline now and will later
    be reused by the adaptive batch-size and learning-rate experiments.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        results_dir,
        checkpoint_dir,
        run_name,
        run_metadata=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name
        self.run_metadata = run_metadata or {}

        self.history = []
        self.optimizer_updates = 0

        self.best_val_loss = float("inf")
        self.best_epoch = None

    def _synchronize_device(self):
        """
        Synchronize CUDA before/after timed sections.

        CUDA operations are asynchronous, so synchronization gives
        more meaningful wall-clock timing.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _compute_gradient_norm(model):
        """
        Compute the global L2 norm of all model gradients.

        One gradient norm is recorded for each optimizer update.
        """
        total_squared_norm = None

        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()
            squared_norm = torch.sum(grad * grad)

            if total_squared_norm is None:
                total_squared_norm = squared_norm
            else:
                total_squared_norm = total_squared_norm + squared_norm

        if total_squared_norm is None:
            return 0.0

        return torch.sqrt(total_squared_norm).item()

    @staticmethod
    def _gradient_statistics(gradient_norms, epsilon=1e-12):
        """
        Calculate epoch-level statistics from per-batch gradient norms.

        CV = standard deviation / (mean + epsilon)

        We refer to this value as the gradient-norm stability measure.
        A lower CV means the gradient norms varied less during the epoch.
        """
        if not gradient_norms:
            return {
                "mean": 0.0,
                "std": 0.0,
                "cv": 0.0,
            }

        mean = sum(gradient_norms) / len(gradient_norms)

        variance = (
            sum(
                (value - mean) ** 2
                for value in gradient_norms
            )
            / len(gradient_norms)
        )

        std = math.sqrt(variance)

        cv = std / (mean + epsilon)

        return {
            "mean": mean,
            "std": std,
            "cv": cv,
        }

    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total_samples = 0

        gradient_norms = []

        for images, labels in train_loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            gradient_norm = self._compute_gradient_norm(
                self.model
            )

            gradient_norms.append(
                gradient_norm
            )

            self.optimizer.step()

            self.optimizer_updates += 1

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )

            total_samples += batch_size

        train_loss = (
            running_loss / total_samples
        )

        train_accuracy = (
            100.0
            * correct
            / total_samples
        )

        gradient_stats = self._gradient_statistics(
            gradient_norms
        )

        return {
            "loss": train_loss,
            "accuracy": train_accuracy,
            "gradient_norm_mean": gradient_stats["mean"],
            "gradient_norm_std": gradient_stats["std"],
            "gradient_norm_cv": gradient_stats["cv"],
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in val_loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )

            total_samples += batch_size

        val_loss = (
            running_loss / total_samples
        )

        val_accuracy = (
            100.0
            * correct
            / total_samples
        )

        return {
            "loss": val_loss,
            "accuracy": val_accuracy,
        }

    def save_checkpoint(
        self,
        epoch,
        val_loss,
    ):
        """
        Save the checkpoint with the lowest validation loss.
        """
        checkpoint_path = (
            self.checkpoint_dir
            / f"{self.run_name}_best.pt"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "optimizer_updates": self.optimizer_updates,
            "run_metadata": self.run_metadata,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    def save_history(self):
        """
        Save epoch-level training history as CSV.
        """
        results_path = (
            self.results_dir
            / f"{self.run_name}.csv"
        )

        dataframe = pd.DataFrame(
            self.history
        )

        dataframe.to_csv(
            results_path,
            index=False,
        )

        return results_path

    def save_metadata(
        self,
        extra_metadata=None,
    ):
        """
        Save experiment configuration and hardware/software
        information as JSON.
        """
        metadata = dict(
            self.run_metadata
        )

        if extra_metadata:
            metadata.update(
                extra_metadata
            )

        metadata_path = (
            self.results_dir
            / f"{self.run_name}_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

        return metadata_path

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        batch_size,
        batch_controller=None,
        train_loader_factory=None,
        lr_controller=None,
    ):
        """
        Train and validate the model for multiple epochs.

        Optional controllers make decisions after validation and apply them
        to the following epoch.
        """
        if batch_controller is not None:
            if train_loader_factory is None:
                raise ValueError(
                    "train_loader_factory is required when "
                    "batch_controller is used."
                )

            if batch_controller.current_batch_size != batch_size:
                raise ValueError(
                    "batch_size must match the controller's "
                    "initial batch size."
                )

        if lr_controller is not None:
            if lr_controller.optimizer is not self.optimizer:
                raise ValueError(
                    "LR controller must use the trainer's optimizer."
                )

            lr_controller.set_batch_size(batch_size)

        current_train_loader = train_loader
        current_batch_size = batch_size

        if batch_controller is not None and lr_controller is not None:
            print("\nStarting adaptive batch + adaptive LR training")
        elif batch_controller is not None:
            print("\nStarting adaptive batch training")
        elif lr_controller is not None:
            print("\nStarting adaptive LR training")
        else:
            print("\nStarting fixed baseline training")

        print("=" * 78)

        self.save_metadata({"status": "started"})
        self._synchronize_device()
        training_start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self._synchronize_device()
            epoch_start_time = time.perf_counter()

            train_metrics = self.train_one_epoch(current_train_loader)
            val_metrics = self.validate(val_loader)

            self._synchronize_device()
            epoch_time = time.perf_counter() - epoch_start_time

            current_lr = self.optimizer.param_groups[0]["lr"]

            batch_decision = None
            next_batch_size = current_batch_size

            if batch_controller is not None:
                batch_decision = batch_controller.step(
                    val_loss=val_metrics["loss"],
                    gradient_norm_cv=train_metrics["gradient_norm_cv"],
                )
                next_batch_size = batch_decision["next_batch_size"]

            lr_decision = None

            if lr_controller is not None:
                next_lr = lr_controller.update(
                    val_loss=val_metrics["loss"],
                    val_accuracy=val_metrics["accuracy"] / 100.0,
                    batch_size=next_batch_size,
                )

                if lr_controller.mode == "adaptive":
                    lr_decision = dict(lr_controller.last_info)
            else:
                next_lr = current_lr

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "gradient_norm_mean": train_metrics["gradient_norm_mean"],
                "gradient_norm_std": train_metrics["gradient_norm_std"],
                "gradient_norm_cv": train_metrics["gradient_norm_cv"],
                "learning_rate": current_lr,
                "batch_size": current_batch_size,
                "optimizer_updates": self.optimizer_updates,
                "epoch_time_seconds": epoch_time,
                "elapsed_seconds": time.perf_counter() - training_start_time,
            }

            if batch_decision is not None:
                epoch_record.update(
                    {
                        "next_batch_size": batch_decision["next_batch_size"],
                        "batch_changed": batch_decision["batch_changed"],
                        "batch_stability_score": batch_decision["stability_score"],
                        "batch_is_stable": batch_decision["is_stable"],
                        "batch_plateau_detected": batch_decision["plateau_detected"],
                        "batch_no_improve_epochs": batch_decision["no_improve_epochs"],
                        "batch_cooldown_remaining": batch_decision["cooldown_remaining"],
                        "batch_controller_reason": batch_decision["reason"],
                    }
                )

            if lr_controller is not None:
                # Record the batch size passed to the LR controller for the next epoch.
                epoch_record["lr_batch_size_input"] = next_batch_size
                epoch_record["next_learning_rate"] = next_lr

                if lr_decision is not None:
                    for key in (
                        "lr_base",
                        "decay_multiplier",
                        "actual_learning_rate",
                        "plateau_counter",
                        "worsening_counter",
                        "cooldown_counter",
                        "lr_change_reason",
                        "warmup_active",
                        "next_warmup_factor",
                    ):
                        epoch_record[key] = lr_decision[key]

            self.history.append(epoch_record)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_metrics["loss"],
                )

                best_marker = "  <-- best"
            else:
                best_marker = ""

            self.save_history()

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
                f"{best_marker}"
            )

            print(
                f"             "
                f"Grad Mean: {train_metrics['gradient_norm_mean']:.4f} | "
                f"Grad Std: {train_metrics['gradient_norm_std']:.4f} | "
                f"Grad CV: {train_metrics['gradient_norm_cv']:.4f}"
            )

            print(
                f"             "
                f"LR: {current_lr:.6f} | "
                f"Batch: {current_batch_size} | "
                f"Updates: {self.optimizer_updates:,} | "
                f"Time: {epoch_time:.2f}s"
            )

            if batch_decision is not None:
                stability_score = batch_decision["stability_score"]

                stability_text = (
                    "N/A"
                    if stability_score is None
                    else f"{stability_score:.4f}"
                )

                print(
                    f"             "
                    f"Batch Controller | "
                    f"Stability: {stability_text} | "
                    f"Plateau: {batch_decision['plateau_detected']} | "
                    f"Next Batch: {batch_decision['next_batch_size']} | "
                    f"Reason: {batch_decision['reason']}"
                )

            if lr_controller is not None:
                if lr_decision is not None:
                    print(
                        f"             "
                        f"LR Controller | "
                        f"Next LR: {next_lr:.6f} | "
                        f"Reason: {lr_decision['lr_change_reason']}"
                    )
                else:
                    print(
                        f"             "
                        f"LR Controller | "
                        f"Next LR: {next_lr:.6f}"
                    )

            if (
                batch_decision is not None
                and batch_decision["batch_changed"]
            ):
                if epoch < epochs:
                    current_train_loader = train_loader_factory(
                        next_batch_size
                    )

                current_batch_size = next_batch_size

        self._synchronize_device()

        total_training_time = (
            time.perf_counter() - training_start_time
        )

        self.total_training_seconds = total_training_time

        results_path = self.save_history()

        if batch_controller is not None:
            batch_history_path = self.results_dir / "batch_history.json"
            batch_history_path.write_text(
                json.dumps(batch_controller.decision_history, indent=2, allow_nan=False),
                encoding="utf-8",
            )

        completion_metadata = {
            "status": "completed",
            "epochs_completed": len(self.history),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_val_loss,
            "total_optimizer_updates": self.optimizer_updates,
            "total_training_time_seconds": total_training_time,
        }

        if batch_controller is not None:
            completion_metadata.update(
                {
                    "final_batch_size_used": self.history[-1]["batch_size"],
                    "final_batch_size_selected": (
                        batch_controller.current_batch_size
                    ),
                }
            )

        if lr_controller is not None:
            completion_metadata.update(
                {
                    "final_learning_rate_used": (
                        self.history[-1]["learning_rate"]
                    ),
                    "final_learning_rate_selected": (
                        self.optimizer.param_groups[0]["lr"]
                    ),
                }
            )

        metadata_path = self.save_metadata(completion_metadata)

        print("=" * 78)
        print("Training complete")
        print(f"Epochs completed: {len(self.history)}")
        print(f"Best checkpoint epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total optimizer updates: {self.optimizer_updates:,}")
        print(
            f"Total training time: "
            f"{total_training_time:.2f} seconds"
        )

        if batch_controller is not None:
            print(
                f"Final batch size used: "
                f"{self.history[-1]['batch_size']}"
            )

        if lr_controller is not None:
            print(
                f"Final learning rate used: "
                f"{self.history[-1]['learning_rate']:.6f}"
            )

        print(f"Results saved to: {results_path}")
        print(f"Metadata saved to: {metadata_path}")
        print("=" * 78)

        return self.history
