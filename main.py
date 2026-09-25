import torch

from config import (
    DEVICE,
    INITIAL_BATCH_SIZE,
    print_config,
)

from data.cifar10 import CIFAR10DataModule
from models.custom_cnn import CustomCNN


def main():

    print_config()

    data = CIFAR10DataModule()

    train_loader = data.get_train_loader(
        INITIAL_BATCH_SIZE
    )

    model = CustomCNN().to(DEVICE)

    images, labels = next(iter(train_loader))

    images = images.to(DEVICE)

    outputs = model(images)

    print("\nSanity check")
    print("-" * 40)

    print(
        "Input batch shape:",
        images.shape,
    )

    print(
        "Label shape:",
        labels.shape,
    )

    print(
        "Model output shape:",
        outputs.shape,
    )

    print("-" * 40)

    assert outputs.shape == (
        INITIAL_BATCH_SIZE,
        10,
    )

    print("PASS: dataset and CNN are connected correctly.")


if __name__ == "__main__":
    main()
