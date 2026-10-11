import torch.nn as nn

from config import NUM_CLASSES
from models.custom_cnn import normalization_layer


class DeeperCNN(nn.Module):
    """Six-convolution CNN for CIFAR images.

    Each stage defaults to two Conv-Norm-ReLU blocks followed by max pooling.
    Default stage widths are 32, 64, and 128 channels. Global average pooling and
    a linear classifier produce logits for ``num_classes`` classes.
    Dropout before the classifier defaults to p=0.2.
    Normalization follows the same configuration as CustomCNN.
    """

    def __init__(self, num_classes=NUM_CLASSES, channels=(32, 64, 128), dropout_p=0.2, convs_per_stage=2):
        super().__init__()

        if len(channels) != 3 or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in channels
        ):
            raise ValueError("channels must contain three positive integer widths")
        if (not isinstance(convs_per_stage, int)
                or isinstance(convs_per_stage, bool) or convs_per_stage <= 0):
            raise ValueError("convs_per_stage must be a positive integer")
        self.channels = tuple(channels)
        self.conv_layers = len(self.channels) * convs_per_stage

        layers = []
        in_channels = 3
        for out_channels in self.channels:
            for _ in range(convs_per_stage):
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        normalization_layer(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.channels[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class DeeperCNNWide(DeeperCNN):
    """Twelve-convolution CNN with four convolutions per stage.

    Stage widths are 64, 128, and 256. Pooling after each stage reduces
    32x32 inputs to a 256x4x4 feature map, followed by GAP and dropout.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.2):
        super().__init__(
            num_classes=num_classes, channels=(64, 128, 256),
            dropout_p=dropout_p, convs_per_stage=4,
        )
