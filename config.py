from pathlib import Path
import random

import numpy as np
import torch


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data" / "downloads"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

SEED = 42


def set_seed(seed=SEED):
    """
    Set random seeds used by Python, NumPy, and PyTorch.

    This helps make experimental runs reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Improve reproducibility for CUDA operations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

NUM_CLASSES = 10

# CIFAR-10 contains 50,000 original training images.
#
# Because of our available computing resources, we will use
# a fixed stratified subset for training and validation.
#
# These values must remain identical across all experiments.

TRAIN_SIZE = 20_000
VAL_SIZE = 5_000

# We keep the full official CIFAR-10 test set.
TEST_SIZE = 10_000

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()


# ---------------------------------------------------------
# Pilot training settings
#
# These are DEVELOPMENT settings.
# They are not yet the final experimental hyperparameters.
# ---------------------------------------------------------

INITIAL_BATCH_SIZE = 32

INITIAL_LEARNING_RATE = 0.1

# Main experiment training budget.
#
# All primary experiments will use the same epoch budget
# so that their training behavior can be compared fairly.
EPOCHS =30

OPTIMIZER = "sgd"

MOMENTUM = 0.9

WEIGHT_DECAY = 5e-4


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

# GroupNorm is currently preferred for the main experiment
# because its normalization statistics do not depend on
# minibatch size.
#
# Available options:
#
# "group"
# "batch"

NORMALIZATION = "group"

GROUP_NORM_GROUPS = 8


# ---------------------------------------------------------
# Batch-size experiment range
# ---------------------------------------------------------

# Reduced range because this project is being conducted
# with limited computing resources.

BATCH_SIZE_OPTIONS = [
    16,
    32,
    64,
    128,
]

MIN_BATCH_SIZE = min(BATCH_SIZE_OPTIONS)

MAX_BATCH_SIZE = max(BATCH_SIZE_OPTIONS)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


def get_device_name():
    """
    Return a human-readable description of the device.
    """

    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)

    if DEVICE.type == "mps":
        return "Apple MPS"

    return "CPU"


def print_config():
    """
    Print the important experiment settings.
    """

    print("=" * 60)

    print(
        "Adaptive Batch Size + Learning Rate Project"
    )

    print("=" * 60)

    print(
        f"Device:              {DEVICE}"
    )

    print(
        f"Hardware:            {get_device_name()}"
    )

    print(
        f"PyTorch version:     {torch.__version__}"
    )

    print(
        f"Random seed:         {SEED}"
    )

    print(
        f"Training samples:    {TRAIN_SIZE:,}"
    )

    print(
        f"Validation samples:  {VAL_SIZE:,}"
    )

    print(
        f"Test samples:        {TEST_SIZE:,}"
    )

    print(
        f"Initial batch size:  {INITIAL_BATCH_SIZE}"
    )

    print(
        f"Initial LR:          {INITIAL_LEARNING_RATE}"
    )

    print(
    f"Experiment epochs:   {EPOCHS}"
    )

    print(
        f"Normalization:       {NORMALIZATION}"
    )

    print(
        f"Batch options:       {BATCH_SIZE_OPTIONS}"
    )

    print("=" * 60)