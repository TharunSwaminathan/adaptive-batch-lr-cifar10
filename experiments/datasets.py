"""Dataset selection shared by experiment runners."""
from data.cifar10 import CIFAR10DataModule
from data.cifar100 import CIFAR100DataModule

DATASETS = {
    "cifar10": (CIFAR10DataModule, 10, "CIFAR-10"),
    "cifar100": (CIFAR100DataModule, 100, "CIFAR-100"),
}

def dataset_settings(name):
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASETS[name]

def dataset_run_name(name, run_name):
    dataset_settings(name)
    return run_name if name == "cifar10" else f"{name}_{run_name}"

def add_dataset_argument(parser):
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10",
                        help="Dataset to train/evaluate (default: cifar10).")
