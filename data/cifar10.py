import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CIFAR10DataModule:
    """Fixed CIFAR-10 subset used by the primary study.

    Train/validation are prepared immediately. The official test set is loaded
    lazily only when get_test_loader() is called.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.train_dataset = None
        self.train_eval_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_indices = None
        self.val_indices = None
        self.class_names = None
        self.train_generator = torch.Generator().manual_seed(SEED)
        self._prepare_train_validation()

    @staticmethod
    def _train_transform():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    @staticmethod
    def _eval_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def _prepare_train_validation(self):
        train_aug = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self._train_transform(),
        )
        train_eval = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self._eval_transform(),
        )
        self.class_names = tuple(train_eval.classes)

        targets = np.asarray(train_aug.targets)
        indices = np.arange(len(targets))
        selected, _ = train_test_split(
            indices,
            train_size=TRAIN_SIZE + VAL_SIZE,
            stratify=targets,
            random_state=SEED,
            shuffle=True,
        )
        selected_targets = targets[selected]
        train_idx, val_idx = train_test_split(
            selected,
            train_size=TRAIN_SIZE,
            test_size=VAL_SIZE,
            stratify=selected_targets,
            random_state=SEED,
            shuffle=True,
        )
        self.train_indices = train_idx.tolist()
        self.val_indices = val_idx.tolist()
        self.train_dataset = Subset(train_aug, self.train_indices)
        self.train_eval_dataset = Subset(train_eval, self.train_indices)
        self.val_dataset = Subset(train_eval, self.val_indices)

    def _ensure_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self._eval_transform(),
            )
        return self.test_dataset

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

    def get_train_eval_loader(self, batch_size=256):
        """Return the training subset with evaluation-only transforms."""
        return DataLoader(
            self.train_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
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
        dataset = self._ensure_test_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            worker_init_fn=seed_worker,
        )
