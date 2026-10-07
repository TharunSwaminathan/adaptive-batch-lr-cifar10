"""Shared model selection for experiment runners."""

from models.custom_cnn import CustomCNN
from models.deeper_cnn import DeeperCNN, DeeperCNNWide

MODELS = {
    "custom_cnn": CustomCNN,
    "deeper_cnn": DeeperCNN,
    "deeper_cnn_wide": DeeperCNNWide,
}


def create_model(name, num_classes):
    if name not in MODELS:
        raise ValueError(f"Unsupported model: {name}")
    return MODELS[name](num_classes=num_classes)


def add_model_argument(parser):
    parser.add_argument(
        "--model", choices=tuple(MODELS), default="custom_cnn",
        help="CNN architecture (default: custom_cnn; deeper_cnn has six convolutions; deeper_cnn_wide has twelve).",
    )


def model_run_name(name, run_name):
    if name not in MODELS:
        raise ValueError(f"Unsupported model: {name}")
    return run_name if name == "custom_cnn" else f"{name}_{run_name}"
