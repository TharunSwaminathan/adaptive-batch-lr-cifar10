import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer


class StubBatchController:
    def __init__(self):
        self.current_batch_size = 32

    def step(self, val_loss, gradient_norm_cv):
        self.current_batch_size = 64
        return {
            "previous_batch_size": 32,
            "next_batch_size": 64,
            "batch_changed": True,
            "stability_score": 0.10,
            "is_stable": True,
            "plateau_detected": True,
            "no_improve_epochs": 0,
            "cooldown_remaining": 2,
            "reason": "plateau_and_stable",
        }


class SpyLRController:
    mode = "adaptive"

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.seen_batch_size = None
        self.last_info = {}

    def set_batch_size(self, batch_size):
        return self.optimizer.param_groups[0]["lr"]

    def update(self, val_loss, val_accuracy, batch_size):
        self.seen_batch_size = batch_size
        next_lr = 0.01 * (batch_size / 32) ** 0.5
        self.optimizer.param_groups[0]["lr"] = next_lr
        self.last_info = {
            "lr_base": next_lr,
            "decay_multiplier": 1.0,
            "actual_learning_rate": next_lr,
            "plateau_counter": 0,
            "worsening_counter": 0,
            "cooldown_counter": 0,
            "lr_change_reason": "keep",
            "warmup_active": False,
            "next_warmup_factor": 1.0,
        }
        return next_lr


class E4CouplingTests(unittest.TestCase):
    def test_trainer_passes_next_batch_size_to_lr_controller(self):
        x = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        y = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch_controller = StubBatchController()
        lr_controller = SpyLRController(optimizer)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                device=torch.device("cpu"),
                results_dir=Path(tmp) / "results",
                checkpoint_dir=Path(tmp) / "checkpoints",
                run_name="coupling_test",
            )
            history = trainer.fit(
                train_loader=loader,
                val_loader=loader,
                epochs=1,
                batch_size=32,
                batch_controller=batch_controller,
                train_loader_factory=lambda _: loader,
                lr_controller=lr_controller,
            )

        self.assertEqual(lr_controller.seen_batch_size, 64)
        self.assertEqual(history[0]["next_batch_size"], 64)
        self.assertEqual(history[0]["lr_batch_size_input"], 64)
        self.assertAlmostEqual(history[0]["next_learning_rate"], 0.014142135623730952)


if __name__ == "__main__":
    unittest.main()
