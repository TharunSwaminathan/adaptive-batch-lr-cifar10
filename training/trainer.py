
import math
import time
from pathlib import Path

import pandas as pd
import torch


class Trainer:
    """
    Shared training and validation engine.

    This trainer is intentionally independent of the
    adaptive batch-size and learning-rate controllers.

    Later, all four experiments will use this same trainer.
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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.run_name = run_name

        self.history = []

        # Counts the total number of optimizer updates.
        self.optimizer_updates = 0

        self.best_val_loss = float("inf")

        self.best_epoch = None


    def _synchronize_device(self):
        """
        Synchronize CUDA before timing operations.

        GPU work is asynchronous, so synchronization gives
        more meaningful wall-clock measurements.
        """

        if self.device.type == "cuda":
            torch.cuda.synchronize()


    @staticmethod
    def _compute_gradient_norm(model):
        """
        Compute the global L2 norm of the gradients.

        This is the quantity we will later use to construct
        our gradient-norm stability measure.
        """

        squared_norm_sum = 0.0

        for parameter in model.parameters():

            if parameter.grad is not None:

                grad_norm = (
                    parameter.grad
                    .detach()
                    .norm(2)
                    .item()
                )

                squared_norm_sum += (
                    grad_norm ** 2
                )

        return math.sqrt(
            squared_norm_sum
        )


    def train_one_epoch(
        self,
        train_loader,
    ):
        """
        Train the model for one complete epoch.
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


            # ---------------------------------------------
            # Reset gradients
            # ---------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )


            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            outputs = self.model(
                images
            )


            loss = self.criterion(
                outputs,
                labels,
            )


            # ---------------------------------------------
            # Backward pass
            # ---------------------------------------------

            loss.backward()


            # ---------------------------------------------
            # Record gradient norm
            # ---------------------------------------------

            gradient_norm = (
                self._compute_gradient_norm(
                    self.model
                )
            )

            gradient_norms.append(
                gradient_norm
            )


            # ---------------------------------------------
            # Update model parameters
            # ---------------------------------------------

            self.optimizer.step()

            self.optimizer_updates += 1


            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            batch_size = labels.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )


            total_samples += (
                batch_size
            )


        train_loss = (
            running_loss
            / total_samples
        )


        train_accuracy = (
            100.0
            * correct
            / total_samples
        )


        if gradient_norms:

            average_gradient_norm = (
                sum(gradient_norms)
                / len(gradient_norms)
            )

        else:

            average_gradient_norm = 0.0


        return {
            "loss": train_loss,
            "accuracy": train_accuracy,
            "gradient_norm": (
                average_gradient_norm
            ),
        }


    @torch.no_grad()
    def validate(
        self,
        val_loader,
    ):
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


            outputs = self.model(
                images
            )


            loss = self.criterion(
                outputs,
                labels,
            )


            batch_size = labels.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )


            total_samples += (
                batch_size
            )


        val_loss = (
            running_loss
            / total_samples
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
        Save the best validation-loss checkpoint.
        """

        checkpoint_path = (
            self.checkpoint_dir
            / f"{self.run_name}_best.pt"
        )


        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "val_loss": val_loss,
            "optimizer_updates": (
                self.optimizer_updates
            ),
        }


        torch.save(
            checkpoint,
            checkpoint_path,
        )


        return checkpoint_path


    def save_history(self):
        """
        Save training history to CSV.
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


    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        batch_size,
    ):
        """
        Train and validate for multiple epochs.
        """

        print(
            "\nStarting fixed baseline training"
        )

        print(
            "=" * 70
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


            # ---------------------------------------------
            # Training
            # ---------------------------------------------

            train_metrics = (
                self.train_one_epoch(
                    train_loader
                )
            )


            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            val_metrics = (
                self.validate(
                    val_loader
                )
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


            # ---------------------------------------------
            # Save epoch statistics
            # ---------------------------------------------

            epoch_record = {

                "epoch": epoch,

                "train_loss": (
                    train_metrics["loss"]
                ),

                "train_accuracy": (
                    train_metrics[
                        "accuracy"
                    ]
                ),

                "val_loss": (
                    val_metrics["loss"]
                ),

                "val_accuracy": (
                    val_metrics[
                        "accuracy"
                    ]
                ),

                "gradient_norm": (
                    train_metrics[
                        "gradient_norm"
                    ]
                ),

                "learning_rate": (
                    current_lr
                ),

                "batch_size": (
                    batch_size
                ),

                "optimizer_updates": (
                    self.optimizer_updates
                ),

                "epoch_time_seconds": (
                    epoch_time
                ),
            }


            self.history.append(
                epoch_record
            )


            # ---------------------------------------------
            # Save best model
            # ---------------------------------------------

            if (
                val_metrics["loss"]
                < self.best_val_loss
            ):

                self.best_val_loss = (
                    val_metrics["loss"]
                )

                self.best_epoch = epoch


                checkpoint_path = (
                    self.save_checkpoint(
                        epoch=epoch,
                        val_loss=(
                            val_metrics[
                                "loss"
                            ]
                        ),
                    )
                )


                best_marker = (
                    "  <-- best"
                )

            else:

                checkpoint_path = None

                best_marker = ""


            # ---------------------------------------------
            # Save CSV after every epoch
            # ---------------------------------------------

            self.save_history()


            # ---------------------------------------------
            # Console output
            # ---------------------------------------------

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: "
                f"{train_metrics['loss']:.4f} | "
                f"Train Acc: "
                f"{train_metrics['accuracy']:.2f}% | "
                f"Val Loss: "
                f"{val_metrics['loss']:.4f} | "
                f"Val Acc: "
                f"{val_metrics['accuracy']:.2f}%"
                f"{best_marker}"
            )


            print(
                f"             "
                f"Grad Norm: "
                f"{train_metrics['gradient_norm']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Updates: {self.optimizer_updates:,} | "
                f"Time: {epoch_time:.2f}s"
            )


        self._synchronize_device()


        total_training_time = (
            time.perf_counter()
            - training_start_time
        )


        results_path = (
            self.save_history()
        )


        print(
            "=" * 70
        )

        print(
            "Training complete"
        )

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
            "=" * 70
        )


        return self.history