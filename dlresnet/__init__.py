"""dlresnet: ResNet-50 from scratch and via transfer learning with Keras 3 / TensorFlow.

Importing the package registers the custom layers and LR schedules, so saved ``.keras`` models
load with plain ``keras.models.load_model(path)`` (no ``custom_objects`` needed).
"""

from __future__ import annotations

import os

# Defaults only (user settings win): data.py/train.py rely on the TensorFlow backend.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dlresnet.config import ExperimentConfig, load_config  # noqa: E402
from dlresnet.layers import BottleneckBlock, CaffePreprocess  # noqa: E402
from dlresnet.model import (  # noqa: E402
    build_model,
    build_resnet50,
    build_transfer_model,
    unfreeze_top_layers,
)
from dlresnet.schedules import StepDecay, WarmupCosine  # noqa: E402

__version__ = "1.0.0"

__all__ = [
    "BottleneckBlock",
    "CaffePreprocess",
    "ExperimentConfig",
    "StepDecay",
    "WarmupCosine",
    "__version__",
    "build_model",
    "build_resnet50",
    "build_transfer_model",
    "load_config",
    "unfreeze_top_layers",
]
