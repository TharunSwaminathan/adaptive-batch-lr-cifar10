import random

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import (
    datasets,
    transforms,
)

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
    Give each DataLoader worker a reproducible random seed.
    """

    worker_seed = (
        torch.initial_seed() % 2**32
    )

    np.random.seed(worker_seed)

    random.seed(worker_seed)


class CIFAR10DataModule:
    """
    Handles CIFAR-10 downloading, preprocessing,
    fixed subset selection, train/validation splitting,
    and DataLoader creation.

    The training DataLoader can later be rebuilt with
    different batch sizes for the adaptive batch-size
    experiments.
    """

    def __init__(
        self,
        data_dir=DATA_DIR,
    ):

        self.data_dir = data_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_indices = None
        self.val_indices = None

        # Persistent generator used for training shuffle.
        self.train_generator = torch.Generator()

        self.train_generator.manual_seed(
            SEED
        )

        self._prepare_datasets()


    def _prepare_datasets(self):
        """
        Download CIFAR-10 and create a fixed,
        stratified training/validation subset.
        """

        # -------------------------------------------------
        # Training transformations
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Validation / test transformations
        # -------------------------------------------------

        evaluation_transform = transforms.Compose(
            [
                transforms.ToTensor(),

                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD,
                ),
            ]
        )


        # -------------------------------------------------
        # Load CIFAR-10
        # -------------------------------------------------

        # Training version includes augmentation.
        full_train_augmented = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )


        # Separate non-augmented copy for validation.
        #
        # This prevents random crop / flip from being
        # accidentally applied during validation.
        full_train_evaluation = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=evaluation_transform,
        )


        # Keep the complete official CIFAR-10 test set.
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=evaluation_transform,
        )


        # -------------------------------------------------
        # Fixed stratified subset
        # -------------------------------------------------

        targets = np.array(
            full_train_augmented.targets
        )

        all_indices = np.arange(
            len(targets)
        )


        # First select the fixed 25,000-image subset
        # that our project will use.
        #
        # 20,000 train
        # +
        # 5,000 validation

        selected_indices, _ = train_test_split(
            all_indices,

            train_size=(
                TRAIN_SIZE + VAL_SIZE
            ),

            stratify=targets,

            random_state=SEED,

            shuffle=True,
        )


        selected_targets = targets[
            selected_indices
        ]


        # -------------------------------------------------
        # Split selected subset into train + validation
        # -------------------------------------------------

        (
            train_indices,
            val_indices,
        ) = train_test_split(

            selected_indices,

            train_size=TRAIN_SIZE,

            test_size=VAL_SIZE,

            stratify=selected_targets,

            random_state=SEED,

            shuffle=True,
        )


        self.train_indices = (
            train_indices.tolist()
        )

        self.val_indices = (
            val_indices.tolist()
        )


        # -------------------------------------------------
        # Build datasets
        # -------------------------------------------------

        self.train_dataset = Subset(
            full_train_augmented,
            self.train_indices,
        )


        self.val_dataset = Subset(
            full_train_evaluation,
            self.val_indices,
        )


        self._print_dataset_information(
            targets
        )


    def get_train_loader(
        self,
        batch_size,
    ):
        """
        Create the training DataLoader.

        This method intentionally accepts batch_size
        dynamically because later the adaptive controller
        will rebuild the loader when the batch size changes.
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


    def get_val_loader(
        self,
        batch_size=256,
    ):
        """
        Validation loader.

        Validation batch size does not need to equal the
        training batch size because no gradient updates
        occur during validation.
        """

        return DataLoader(
            self.val_dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=NUM_WORKERS,

            pin_memory=PIN_MEMORY,

            worker_init_fn=seed_worker,
        )


    def get_test_loader(
        self,
        batch_size=256,
    ):
        """
        Test DataLoader using the complete official
        CIFAR-10 test set.
        """

        return DataLoader(
            self.test_dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=NUM_WORKERS,

            pin_memory=PIN_MEMORY,

            worker_init_fn=seed_worker,
        )


    def _print_dataset_information(
        self,
        targets,
    ):
        """
        Print dataset sizes and class distributions.
        """

        train_targets = targets[
            np.array(
                self.train_indices
            )
        ]

        val_targets = targets[
            np.array(
                self.val_indices
            )
        ]


        train_counts = np.bincount(
            train_targets,
            minlength=10,
        )


        val_counts = np.bincount(
            val_targets,
            minlength=10,
        )


        print(
            "\nCIFAR-10 dataset ready"
        )

        print(
            "-" * 60
        )

        print(
            f"Training samples:   "
            f"{len(self.train_dataset):,}"
        )

        print(
            f"Validation samples: "
            f"{len(self.val_dataset):,}"
        )

        print(
            f"Test samples:       "
            f"{len(self.test_dataset):,}"
        )


        print(
            "\nTraining samples per class:"
        )

        print(
            train_counts.tolist()
        )


        print(
            "Validation samples per class:"
        )

        print(
            val_counts.tolist()
        )

        print(
            "-" * 60
        )