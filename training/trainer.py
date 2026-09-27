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
    ):
        """
        Train and validate the model for multiple epochs.
        """
        print(
            "\nStarting fixed baseline training"
        )

        print("=" * 78)

        # Save initial configuration before training begins.
        self.save_metadata(
            {
                "status": "started",
            }
        )

        self._synchronize_device()

        training_start_time = (
            time.perf_counter()
        )

        for epoch in range(
            1,
            epochs + 1,
        ):
            self._synchronize_device()

            epoch_start_time = (
                time.perf_counter()
            )

            train_metrics = self.train_one_epoch(
                train_loader
            )

            val_metrics = self.validate(
                val_loader
            )

            self._synchronize_device()

            epoch_time = (
                time.perf_counter()
                - epoch_start_time
            )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "gradient_norm_mean": train_metrics[
                    "gradient_norm_mean"
                ],
                "gradient_norm_std": train_metrics[
                    "gradient_norm_std"
                ],
                "gradient_norm_cv": train_metrics[
                    "gradient_norm_cv"
                ],
                "learning_rate": current_lr,
                "batch_size": batch_size,
                "optimizer_updates": self.optimizer_updates,
                "epoch_time_seconds": epoch_time,
            }

            self.history.append(
                epoch_record
            )

            if (
                val_metrics["loss"]
                < self.best_val_loss
            ):
                self.best_val_loss = (
                    val_metrics["loss"]
                )

                self.best_epoch = epoch

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_metrics["loss"],
                )

                best_marker = "  <-- best"

            else:
                best_marker = ""

            # Save after every epoch so partial runs are not lost.
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
                f"Grad Mean: "
                f"{train_metrics['gradient_norm_mean']:.4f} | "
                f"Grad Std: "
                f"{train_metrics['gradient_norm_std']:.4f} | "
                f"Grad CV: "
                f"{train_metrics['gradient_norm_cv']:.4f}"
            )

            print(
                f"             "
                f"LR: {current_lr:.6f} | "
                f"Batch: {batch_size} | "
                f"Updates: {self.optimizer_updates:,} | "
                f"Time: {epoch_time:.2f}s"
            )

        self._synchronize_device()

        total_training_time = (
            time.perf_counter()
            - training_start_time
        )

        results_path = self.save_history()

        metadata_path = self.save_metadata(
            {
                "status": "completed",
                "best_epoch": self.best_epoch,
                "best_validation_loss": self.best_val_loss,
                "total_optimizer_updates": self.optimizer_updates,
                "total_training_time_seconds": total_training_time,
            }
        )

        print("=" * 78)
        print("Training complete")

        print(
            f"Best epoch: "
            f"{self.best_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{self.best_val_loss:.4f}"
        )

        print(
            f"Total optimizer updates: "
            f"{self.optimizer_updates:,}"
        )

        print(
            f"Total training time: "
            f"{total_training_time:.2f} seconds"
        )

        print(
            f"Results saved to: "
            f"{results_path}"
        )

        print(
            f"Metadata saved to: "
            f"{metadata_path}"
        )

        print("=" * 78)

        return self.history