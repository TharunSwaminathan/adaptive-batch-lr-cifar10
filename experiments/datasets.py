"""Dataset selection shared by the primary experiment runners."""

from data.cifar10 import CIFAR10DataModule
from data.cifar100 import CIFAR100DataModule

DATASETS = {
    "cifar10": (CIFAR10DataModule, 10, "CIFAR-10"),
    "cifar100": (CIFAR100DataModule, 100, "CIFAR-100"),
}


def dataset_settings(name):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


def add_dataset_argument(parser):
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="cifar10",
        help="Dataset for this run (default: cifar10).",
    )
