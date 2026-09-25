import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE,
)


def seed_worker(worker_id):
    """
    Make DataLoader workers reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR10DataModule:
    """
    Handles CIFAR-10 downloading, preprocessing,
    reproducible splitting, and DataLoader creation.

    The training DataLoader can be rebuilt with a different
    batch size. This will be useful later when we implement
    adaptive batch-size scheduling.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_indices = None
        self.val_indices = None

        # One persistent generator helps keep shuffling reproducible
        # without restarting the exact same sequence every epoch.
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(SEED)

        self._prepare_datasets()

    def _prepare_datasets(self):
        """
        Download CIFAR-10 and create reproducible
        training/validation splits.
        """

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD,
                ),
            ]
        )

        evaluation_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD,
                ),
            ]
        )

        # Training copy uses augmentation.
        full_train_augmented = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        # Separate copy prevents augmentation from being
        # accidentally applied to the validation set.
        full_train_evaluation = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=evaluation_transform,
        )

        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=evaluation_transform,
        )

        split_generator = torch.Generator()
        split_generator.manual_seed(SEED)

        indices = torch.randperm(
            len(full_train_augmented),
            generator=split_generator,
        ).tolist()

        self.train_indices = indices[:TRAIN_SIZE]

        self.val_indices = indices[
            TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE
        ]

        self.train_dataset = Subset(
            full_train_augmented,
            self.train_indices,
        )

        self.val_dataset = Subset(
            full_train_evaluation,
            self.val_indices,
        )

        self._print_dataset_information()

    def get_train_loader(self, batch_size):
        """
        Create or rebuild the training DataLoader.

        Later, the adaptive batch controller will call this
        whenever the batch size changes.
        """

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
        """
        Validation batch size does not need to match
        the training batch size because gradients are
        not computed during validation.
        """

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
        )

    def get_test_loader(self, batch_size=256):
        """
        Test DataLoader.
        """

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
        )

    def _print_dataset_information(self):
        print("\nCIFAR-10 dataset ready")
        print("-" * 40)
        print(
            f"Training samples:   {len(self.train_dataset):,}"
        )
        print(
            f"Validation samples: {len(self.val_dataset):,}"
        )
        print(
            f"Test samples:       {len(self.test_dataset):,}"
        )
        print("-" * 40)
