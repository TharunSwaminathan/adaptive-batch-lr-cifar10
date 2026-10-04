"""Canonical 40-epoch E1-E4 protocol for the combined team project.

This module is the single source of truth for the primary comparison.
E4 is fully specified independently; it is not created by merging E2/E3.
"""

import hashlib
import json

PRIMARY_SEED = 42
PRIMARY_EPOCHS = 40

E1_CONFIG = {
    "experiment": "E1",
    "label": "Fixed batch + fixed LR",
    "epochs": 40,
    "seed": 42,
    "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
    "batch": {"policy": "fixed", "size": 32},
    "learning_rate": {"policy": "fixed", "value": 0.01},
}

E2_CONFIG = {
    "experiment": "E2",
    "label": "Adaptive batch + fixed LR",
    "epochs": 40,
    "seed": 42,
    "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
    "batch": {
        "policy": "adaptive",
        "initial": 32,
        "allowed": [32, 64, 128],
        "growth_policy": "grow_only",
        "cv_window": 3,
        "stability_threshold": 0.25,
        "plateau_patience": 3,
        "min_delta": 0.01,
        "cooldown_epochs": 2,
    },
    "learning_rate": {"policy": "fixed", "value": 0.01},
}

E3_CONFIG = {
    "experiment": "E3",
    "label": "Fixed batch + adaptive LR",
    "epochs": 40,
    "seed": 42,
    "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
    "batch": {"policy": "fixed", "size": 32},
    "learning_rate": {
        "policy": "adaptive",
        "reference_lr": 0.01,
        "reference_batch_size": 32,
        "alpha": 0.5,
        "factor": 0.5,
        "plateau_patience": 5,
        "worsening_patience": 3,
        "cooldown_epochs": 2,
        "warmup_epochs": 0,
        "min_lr": 1e-5,
        "min_delta_loss": 1e-3,
        "min_delta_acc": 0.002,
    },
}

E4_CONFIG = {
    "experiment": "E4",
    "label": "Adaptive batch + adaptive LR",
    "epochs": 40,
    "seed": 42,
    "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
    "batch": {
        "policy": "adaptive",
        "initial": 32,
        "allowed": [32, 64, 128],
        "growth_policy": "grow_only",
        "cv_window": 3,
        "stability_threshold": 0.25,
        "plateau_patience": 3,
        "min_delta": 0.01,
        "cooldown_epochs": 2,
    },
    "learning_rate": {
        "policy": "adaptive",
        "reference_lr": 0.01,
        "reference_batch_size": 32,
        "alpha": 0.5,
        "factor": 0.5,
        "plateau_patience": 5,
        "worsening_patience": 3,
        "cooldown_epochs": 2,
        "warmup_epochs": 0,
        "min_lr": 1e-5,
        "min_delta_loss": 1e-3,
        "min_delta_acc": 0.002,
    },
    "coupling": {
        "decision_order": ["batch", "learning_rate"],
        "lr_batch_input": "next_batch_size",
    },
}

PRIMARY_CONFIGS = {
    "E1": E1_CONFIG,
    "E2": E2_CONFIG,
    "E3": E3_CONFIG,
    "E4": E4_CONFIG,
}



def protocol_fingerprint(config):
    """Return a stable SHA-256 fingerprint for one experiment config."""
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def batch_scaled_lr(
    batch_size,
    reference_lr=0.01,
    reference_batch_size=32,
    alpha=0.5,
    decay_multiplier=1.0,
):
    base = reference_lr * (batch_size / reference_batch_size) ** alpha
    return base * decay_multiplier


def validate_primary_protocol():
    for name, cfg in PRIMARY_CONFIGS.items():
        assert cfg["experiment"] == name
        assert cfg["epochs"] == PRIMARY_EPOCHS == 40
        assert cfg["seed"] == PRIMARY_SEED == 42
        assert cfg["optimizer"] == {
            "name": "SGD",
            "momentum": 0.9,
            "weight_decay": 5e-4,
        }

    assert E1_CONFIG["batch"] == {"policy": "fixed", "size": 32}
    assert E1_CONFIG["learning_rate"] == {"policy": "fixed", "value": 0.01}

    assert E2_CONFIG["batch"]["allowed"] == [32, 64, 128]
    assert E2_CONFIG["learning_rate"] == {"policy": "fixed", "value": 0.01}

    assert E3_CONFIG["learning_rate"]["alpha"] == 0.5
    assert E3_CONFIG["learning_rate"]["warmup_epochs"] == 0

    # E4 is explicitly and independently specified.
    assert E4_CONFIG["batch"]["initial"] == 32
    assert E4_CONFIG["batch"]["allowed"] == [32, 64, 128]
    assert E4_CONFIG["learning_rate"]["reference_lr"] == 0.01
    assert E4_CONFIG["learning_rate"]["reference_batch_size"] == 32
    assert E4_CONFIG["learning_rate"]["alpha"] == 0.5
    assert E4_CONFIG["learning_rate"]["factor"] == 0.5
    assert E4_CONFIG["learning_rate"]["warmup_epochs"] == 0
    assert E4_CONFIG["learning_rate"]["min_lr"] == 1e-5
    assert E4_CONFIG["coupling"] == {
        "decision_order": ["batch", "learning_rate"],
        "lr_batch_input": "next_batch_size",
    }

    assert abs(batch_scaled_lr(64) - 0.014142135623730952) < 1e-12
    assert abs(batch_scaled_lr(64, decay_multiplier=0.5) - 0.007071067811865476) < 1e-12
    return True


if __name__ == "__main__":
    validate_primary_protocol()
    print("Primary 40-epoch E1-E4 protocol validation: PASS")
    print("E4 batch 64 LR base:", batch_scaled_lr(64))
    print("E4 batch 64 + 0.5 decay:", batch_scaled_lr(64, decay_multiplier=0.5))
