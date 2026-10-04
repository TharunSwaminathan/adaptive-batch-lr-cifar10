from pathlib import Path
import random

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "downloads"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

SEED = 42
PRIMARY_EPOCHS = 40
INITIAL_BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.01
OPTIMIZER = "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# GroupNorm avoids making feature normalization depend on minibatch size.
NORMALIZATION = "group"
GROUP_NORM_GROUPS = 8

# CIFAR-10 uses the same fixed 20k/5k subset design as the original study.
NUM_CLASSES = 10
TRAIN_SIZE = 20_000
VAL_SIZE = 5_000
TEST_SIZE = 10_000
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Primary adaptive-batch choices.
BATCH_SIZE_OPTIONS = [32, 64, 128]

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch for controlled experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_name():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)
    if DEVICE.type == "mps":
        return "Apple MPS"
    return "CPU"


def ensure_output_directories():
    for path in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
