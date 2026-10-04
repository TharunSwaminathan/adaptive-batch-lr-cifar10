import random

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DATA_DIR, NUM_WORKERS, PIN_MEMORY, SEED


# ---------------------------------------------------------
# CIFAR-100 settings
# ---------------------------------------------------------

TRAIN_SIZE = 45_000
VAL_SIZE = 5_000
NUM_CLASSES = 100

# These values were calculated from the original CIFAR-100
# 50,000-image training partition before the internal 45k/5k split.
CIFAR100_MEAN = (
    0.507075,
    0.486549,
    0.440918,
)

CIFAR100_STD = (
    0.267334,
    0.256438,
    0.276150,
)


def seed_worker(worker_id):
    """Give each DataLoader worker a reproducible random seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR100DataModule:
    """
    CIFAR-100 data pipeline used by the project extension.

    The original 50,000-image training partition is split into
    45,000 training images and 5,000 validation images using a
    fixed stratified split. The official 10,000-image test set is
    kept separate for final evaluation.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_indices = None
        self.val_indices = None

        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(SEED)

        self._prepare_datasets()

    def _prepare_datasets(self):
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR100_MEAN,
                    CIFAR100_STD,
                ),
            ]
        )

        evaluation_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR100_MEAN,
                    CIFAR100_STD,
                ),
            ]
        )

        full_train_augmented = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        full_train_evaluation = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=False,
            transform=evaluation_transform,
        )

        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=evaluation_transform,
        )

        targets = np.array(full_train_augmented.targets)
        all_indices = np.arange(len(targets))

        train_indices, val_indices = train_test_split(
            all_indices,
            train_size=TRAIN_SIZE,
            test_size=VAL_SIZE,
            stratify=targets,
            random_state=SEED,
            shuffle=True,
        )

        self.train_indices = train_indices.tolist()
        self.val_indices = val_indices.tolist()

        self.train_dataset = Subset(
            full_train_augmented,
            self.train_indices,
        )

        self.val_dataset = Subset(
            full_train_evaluation,
            self.val_indices,
        )

        self._print_dataset_information(targets)

    def get_train_loader(self, batch_size):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
            generator=self.train_generator,
        )

    def get_val_loader(self, batch_size=256):
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
        )

    def get_test_loader(self, batch_size=256):
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
        )

    def _print_dataset_information(self, targets):
        train_targets = targets[np.array(self.train_indices)]
        val_targets = targets[np.array(self.val_indices)]

        train_counts = np.bincount(
            train_targets,
            minlength=NUM_CLASSES,
        )

        val_counts = np.bincount(
            val_targets,
            minlength=NUM_CLASSES,
        )

        print("\nCIFAR-100 dataset ready")
        print("-" * 60)
        print(f"Training samples:   {len(self.train_dataset):,}")
        print(f"Validation samples: {len(self.val_dataset):,}")
        print(f"Test samples:       {len(self.test_dataset):,}")
        print("\nTraining samples per class:")
        print(train_counts.tolist())
        print("\nValidation samples per class:")
        print(val_counts.tolist())
        print("-" * 60)
