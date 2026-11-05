"""Describe a saved model (shape, classes, parameter counts) to verify training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from dlresnet.evaluate import load_class_names, resolve_model_path


def _count(weights: list[Any]) -> int:
    return int(sum(int(np.prod(w.shape)) for w in weights))


def describe(model_path: str | Path) -> dict[str, Any]:
    """Return input shape, number of classes, class names and parameter counts of a model."""
    path = resolve_model_path(model_path)
    model = keras.models.load_model(path, compile=False)
    num_classes = int(model.output_shape[-1])
    trainable = _count(model.trainable_weights)
    non_trainable = _count(model.non_trainable_weights)
    return {
        "model_path": str(path),
        "name": model.name,
        "input_shape": [int(d) for d in model.input_shape[1:]],
        "num_classes": num_classes,
        "class_names": load_class_names(path, num_classes),
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "num_layers": len(model.layers),
        "file_size_bytes": path.stat().st_size,
    }
