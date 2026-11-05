"""Single-image inference with the same resize as training and in-model preprocessing.

The notebook divided by 255 at prediction time while training used Caffe mean subtraction, and
loaded a bird image for the ant/bee model. Here the image goes in as raw RGB ``[0, 255]`` and the
model's own ``CaffePreprocess`` layer handles normalisation; labels come from ``class_names.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import keras
import numpy as np


def load_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image as float32 ``H x W x 3`` RGB in ``[0, 255]``.

    Resizing uses bilinear interpolation without antialiasing, the same op that
    ``keras.utils.image_dataset_from_directory`` applies during training.
    """
    img = keras.utils.load_img(str(path), color_mode="rgb")
    arr = keras.utils.img_to_array(img, dtype="float32")
    resized = keras.ops.image.resize(
        arr, tuple(int(s) for s in image_size), interpolation="bilinear", antialias=False
    )
    return np.asarray(keras.ops.convert_to_numpy(resized), dtype=np.float32)


def predict(
    model: keras.Model,
    class_names: list[str],
    paths: Sequence[str | Path],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k predictions per image: ``[{"path", "predictions": [{"label", "prob"}, ...]}]``."""
    if not paths:
        return []
    image_size = (int(model.input_shape[1]), int(model.input_shape[2]))
    batch = np.stack([load_image(p, image_size) for p in paths])
    probs = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
    k = max(1, min(int(top_k), probs.shape[1]))
    results = []
    for path, row in zip(paths, probs, strict=True):
        order = np.argsort(-row)[:k]
        results.append(
            {
                "path": str(path),
                "predictions": [{"label": class_names[i], "prob": float(row[i])} for i in order],
            }
        )
    return results
