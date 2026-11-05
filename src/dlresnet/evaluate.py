"""Load trained runs and evaluate them (loss, accuracy, top-5, confusion matrix)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keras
import numpy as np

import dlresnet.layers  # noqa: F401  (registers custom layers for load_model)
import dlresnet.schedules  # noqa: F401  (registers custom LR schedules for load_model)

CLASS_NAMES_FILE = "class_names.json"


def resolve_model_path(run_dir_or_model_path: str | Path) -> Path:
    """Return the ``.keras`` file for a run dir (``model.keras``, else ``best.keras``) or file."""
    path = Path(run_dir_or_model_path)
    if path.is_dir():
        for name in ("model.keras", "best.keras"):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"No model.keras or best.keras in run directory {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix != ".keras":
        raise ValueError(f"Expected a native Keras 3 '.keras' file, got {path.name}")
    return path


def load_class_names(model_path: Path, num_classes: int | None = None) -> list[str] | None:
    """Read ``class_names.json`` next to the model; ``None`` if absent."""
    names_file = model_path.parent / CLASS_NAMES_FILE
    if not names_file.is_file():
        return None
    names = [str(n) for n in json.loads(names_file.read_text())]
    if num_classes is not None and len(names) != num_classes:
        raise ValueError(
            f"{names_file} lists {len(names)} classes but the model outputs {num_classes}"
        )
    return names


def load_run(
    run_dir_or_model_path: str | Path, compile: bool = False
) -> tuple[keras.Model, list[str]]:
    """Load a trained model and its class names (falls back to ``"0".."n-1"`` if missing)."""
    model_path = resolve_model_path(run_dir_or_model_path)
    model = keras.models.load_model(model_path, compile=compile)
    num_classes = int(model.output_shape[-1])
    names = load_class_names(model_path, num_classes)
    if names is None:
        names = [str(i) for i in range(num_classes)]
    return model, names


def evaluate_model(model: keras.Model, dataset: Any, class_names: list[str]) -> dict[str, Any]:
    """Evaluate on a batched ``(images, int_labels)`` dataset in a single prediction pass.

    Returns a JSON-serialisable dict with loss, accuracy, top5 (if > 5 classes), the confusion
    matrix (rows = true class, columns = predicted class) and per-class accuracy.
    """
    num_classes = len(class_names)
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for images, y in dataset:
        probs.append(np.asarray(model.predict_on_batch(images), dtype=np.float64))
        labels.append(np.asarray(y).astype(np.int64).reshape(-1))
    if not labels:
        raise ValueError("Cannot evaluate on an empty dataset")
    y_true = np.concatenate(labels)
    p = np.concatenate(probs)
    y_pred = p.argmax(axis=1)

    eps = 1e-7
    loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], eps, 1.0))))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    support = confusion.sum(axis=1)
    per_class = {
        name: (float(confusion[i, i] / support[i]) if support[i] else None)
        for i, name in enumerate(class_names)
    }
    metrics: dict[str, Any] = {
        "num_samples": int(len(y_true)),
        "loss": loss,
        "accuracy": float(np.mean(y_pred == y_true)),
    }
    if num_classes > 5:
        top5 = np.argsort(-p, axis=1)[:, :5]
        metrics["top5"] = float(np.mean(np.any(top5 == y_true[:, None], axis=1)))
    metrics["per_class_accuracy"] = per_class
    metrics["class_names"] = list(class_names)
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def evaluate_dir(
    model_path: str | Path,
    data_dir: str | Path,
    image_size: tuple[int, int] | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a saved model on ``data_dir/<class>/<images>`` (class order from training)."""
    from dlresnet.data import image_folder_dataset

    model, class_names = load_run(model_path)
    if image_size is None:
        image_size = (int(model.input_shape[1]), int(model.input_shape[2]))
    dataset, _ = image_folder_dataset(
        data_dir, image_size, batch_size, shuffle=False, class_names=class_names
    )
    metrics = evaluate_model(model, dataset, class_names)
    return {
        "model": str(resolve_model_path(model_path)),
        "data_dir": str(data_dir),
        **metrics,
    }
