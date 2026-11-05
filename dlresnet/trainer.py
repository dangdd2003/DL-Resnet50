"""End-to-end training: data -> model -> fit (+ optional fine-tuning) -> artifacts."""

from __future__ import annotations

import importlib.util
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import tensorflow as tf

from dlresnet.config import ExperimentConfig
from dlresnet.data import build_augmentation, load_datasets
from dlresnet.evaluation import CLASS_NAMES_FILE, evaluate_model
from dlresnet.model import build_model, unfreeze_top_layers
from dlresnet.plotting import plot_history
from dlresnet.schedules import build_lr_schedule, build_optimizer


@dataclass
class TrainResult:
    run_dir: Path
    model_path: Path
    history: dict[str, list[float]]
    test_metrics: dict[str, Any] | None


class LearningRateLogger(keras.callbacks.Callback):
    """Adds the optimizer's current learning rate to the epoch logs (history/CSV)."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is not None:
            lr = self.model.optimizer.learning_rate
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(lr))


def _compile(model: keras.Model, optimizer: keras.optimizers.Optimizer, num_classes: int) -> None:
    metrics: list[keras.metrics.Metric] = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if num_classes > 5:
        metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"))
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics,
    )


def _resolve_precision_policy(requested: bool) -> str:
    """Return the Keras dtype policy name, falling back to float32 when no GPU is visible."""
    if not requested:
        return "float32"
    if tf.config.list_physical_devices("GPU"):
        return "mixed_float16"
    warnings.warn(
        "train.mixed_precision=true but no GPU is visible; training in float32 instead.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "float32"


def _merge(history: dict[str, list[float]], new: dict[str, list[Any]]) -> None:
    for key, values in new.items():
        history.setdefault(key, []).extend(float(v) for v in values)


def train(cfg: ExperimentConfig) -> TrainResult:
    """Train according to ``cfg`` and write all artifacts into ``cfg.run_dir``.

    Artifacts: ``config.yaml``, ``class_names.json``, ``best.keras`` (best val accuracy),
    ``model.keras`` (final), ``history.json``/``history.csv``/``history.png``, ``tb/`` (only if
    the ``tensorboard`` package is installed) and
    ``metrics.json`` when a test split is configured.
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    keras.utils.set_random_seed(cfg.data.seed)
    keras.mixed_precision.set_global_policy(_resolve_precision_policy(cfg.train.mixed_precision))
    strategy = (
        tf.distribute.MirroredStrategy() if cfg.train.distribute else tf.distribute.get_strategy()
    )

    data = load_datasets(cfg.data)
    num_classes = len(data.class_names)
    (run_dir / CLASS_NAMES_FILE).write_text(json.dumps(data.class_names, indent=2))

    with strategy.scope():
        augmentation = build_augmentation(cfg.data.augmentation, seed=cfg.data.seed)
        model = build_model(cfg.model, data.input_shape, num_classes, augmentation)
        schedule = build_lr_schedule(cfg.train, data.steps_per_epoch, cfg.train.epochs)
        _compile(model, build_optimizer(cfg.train, schedule), num_classes)

    best_path = run_dir / "best.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        best_path, monitor="val_accuracy", mode="max", save_best_only=True
    )

    def callbacks(append_csv: bool) -> list[keras.callbacks.Callback]:
        cbs: list[keras.callbacks.Callback] = [
            LearningRateLogger(),  # first, so later callbacks see "learning_rate" in logs
            checkpoint,
            keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=append_csv),
        ]
        if importlib.util.find_spec("tensorboard") is not None:  # optional (not a TF dep anymore)
            cbs.append(keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb")))
        if cfg.train.early_stopping_patience:
            cbs.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=cfg.train.early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        return cbs

    history: dict[str, list[float]] = {}
    fit = model.fit(
        data.train,
        validation_data=data.val,
        epochs=cfg.train.epochs,
        callbacks=callbacks(False),
        shuffle=False,  # the tf.data pipeline already reshuffles every epoch
    )
    _merge(history, fit.history)

    if cfg.train.fine_tune_epochs > 0:
        with strategy.scope():
            n_trainable = unfreeze_top_layers(model, cfg.train.fine_tune_layers)
            schedule = build_lr_schedule(
                cfg.train,
                data.steps_per_epoch,
                cfg.train.fine_tune_epochs,
                base_lr=cfg.train.fine_tune_learning_rate,
            )
            _compile(model, build_optimizer(cfg.train, schedule), num_classes)
        print(f"Fine-tuning: {n_trainable} backbone layers trainable")
        start = len(history.get("loss", []))
        fit = model.fit(
            data.train,
            validation_data=data.val,
            initial_epoch=start,
            epochs=start + cfg.train.fine_tune_epochs,
            callbacks=callbacks(True),
            shuffle=False,
        )
        _merge(history, fit.history)

    model_path = run_dir / "model.keras"
    model.save(model_path)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, run_dir / "history.png")

    test_metrics = None
    if data.test is not None:
        test_metrics = evaluate_model(model, data.test, data.class_names)
        (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))

    return TrainResult(
        run_dir=run_dir, model_path=model_path, history=history, test_metrics=test_metrics
    )
