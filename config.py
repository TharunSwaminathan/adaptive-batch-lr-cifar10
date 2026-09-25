from pathlib import Path

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


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

NUM_CLASSES = 10

TRAIN_SIZE = 45_000
VAL_SIZE = 5_000
TEST_SIZE = 10_000

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()


# ---------------------------------------------------------
# Initial baseline settings
#
# These are pilot settings, not final experimental values.
# We will freeze the final values after baseline testing.
# ---------------------------------------------------------

INITIAL_BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.01

EPOCHS = 50

OPTIMIZER = "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

# Recommended for the batch-size study because GroupNorm
# does not depend on minibatch statistics.
#
# Options:
#   "group"
#   "batch"
#
NORMALIZATION = "group"

GROUP_NORM_GROUPS = 8


# ---------------------------------------------------------
# Adaptive batch-size limits
#
# We define these now but DO NOT use adaptive training yet.
# ---------------------------------------------------------

BATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256]

MIN_BATCH_SIZE = min(BATCH_SIZE_OPTIONS)
MAX_BATCH_SIZE = max(BATCH_SIZE_OPTIONS)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def print_config():
    """Print the important experiment settings."""

    print("=" * 60)
    print("Adaptive Batch Size + Learning Rate Project")
    print("=" * 60)

    print(f"Device:              {DEVICE}")
    print(f"Random seed:         {SEED}")
    print(f"Training samples:    {TRAIN_SIZE}")
    print(f"Validation samples:  {VAL_SIZE}")
    print(f"Test samples:        {TEST_SIZE}")
    print(f"Initial batch size:  {INITIAL_BATCH_SIZE}")
    print(f"Initial LR:          {INITIAL_LEARNING_RATE}")
    print(f"Epochs:              {EPOCHS}")
    print(f"Normalization:       {NORMALIZATION}")

    print("=" * 60)
