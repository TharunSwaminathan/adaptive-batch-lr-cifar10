import torch.nn as nn

from config import (
    GROUP_NORM_GROUPS,
    NORMALIZATION,
    NUM_CLASSES,
)


def normalization_layer(num_channels):
    """
    Return the normalization layer selected in config.py.

    GroupNorm is recommended for the main batch-size study
    because its statistics do not depend on minibatch size.
    """

    if NORMALIZATION == "batch":
        return nn.BatchNorm2d(num_channels)

    if NORMALIZATION == "group":
        return nn.GroupNorm(
            num_groups=GROUP_NORM_GROUPS,
            num_channels=num_channels,
        )

    raise ValueError(
        f"Unsupported normalization type: {NORMALIZATION}"
    )


class CustomCNN(nn.Module):
    """
    CNN used for the main CIFAR-10 experiments.

    Architecture:
        Conv(32)
        Conv(64)
        MaxPool
        Conv(128)
        MaxPool
        Global Average Pool
        Linear(10)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            # -------------------------------------------------
            # Stage 1
            # -------------------------------------------------

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(32),

            nn.ReLU(inplace=True),

            # -------------------------------------------------
            # Stage 2
            # -------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # -------------------------------------------------
            # Stage 3
            # -------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            normalization_layer(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Linear(
            128,
            num_classes,
        )

    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = x.flatten(1)

        x = self.classifier(x)

        return x
