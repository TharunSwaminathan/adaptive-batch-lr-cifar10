import unittest

import torch

from experiments.primary_protocol import (
    E4_CONFIG,
    PRIMARY_CONFIGS,
    batch_scaled_lr,
    protocol_fingerprint,
    validate_primary_protocol,
)
from training.lr_controller import LRController


class PrimaryProtocolTests(unittest.TestCase):
    def test_all_primary_runs_are_40_epochs(self):
        self.assertTrue(validate_primary_protocol())
        self.assertEqual({cfg["epochs"] for cfg in PRIMARY_CONFIGS.values()}, {40})

    def test_e4_is_fully_specified(self):
        self.assertEqual(E4_CONFIG["batch"]["allowed"], [32, 64, 128])
        self.assertEqual(E4_CONFIG["learning_rate"]["reference_lr"], 0.01)
        self.assertEqual(E4_CONFIG["learning_rate"]["alpha"], 0.5)
        self.assertEqual(E4_CONFIG["learning_rate"]["warmup_epochs"], 0)
        self.assertEqual(E4_CONFIG["coupling"]["lr_batch_input"], "next_batch_size")

    def test_e4_batch_scaled_lr_examples(self):
        self.assertAlmostEqual(batch_scaled_lr(64), 0.014142135623730952, places=14)
        self.assertAlmostEqual(
            batch_scaled_lr(64, decay_multiplier=0.5),
            0.007071067811865476,
            places=14,
        )

    def test_protocol_fingerprint_is_stable_and_experiment_specific(self):
        first = protocol_fingerprint(PRIMARY_CONFIGS["E1"])
        second = protocol_fingerprint(dict(PRIMARY_CONFIGS["E1"]))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, protocol_fingerprint(PRIMARY_CONFIGS["E4"]))

    def test_real_lr_controller_uses_batch_scaling(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        cfg = E4_CONFIG["learning_rate"]
        controller = LRController(
            optimizer,
            mode="adaptive",
            reference_batch_size=cfg["reference_batch_size"],
            reference_lr=cfg["reference_lr"],
            alpha=cfg["alpha"],
            lr_factor=cfg["factor"],
            plateau_patience=cfg["plateau_patience"],
            worsening_patience=cfg["worsening_patience"],
            cooldown_epochs=cfg["cooldown_epochs"],
            warmup_epochs=cfg["warmup_epochs"],
            min_lr=cfg["min_lr"],
            min_delta_loss=cfg["min_delta_loss"],
            min_delta_acc=cfg["min_delta_acc"],
        )
        self.assertAlmostEqual(controller.set_batch_size(64), 0.014142135623730952)
        controller.decay_multiplier = 0.5
        self.assertAlmostEqual(controller.set_batch_size(64), 0.007071067811865476)


if __name__ == "__main__":
    unittest.main()
