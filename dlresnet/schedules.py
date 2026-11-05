"""Stateless learning-rate schedules (functions of the optimizer step) and optimizer factory.

The notebooks used a ``LearningRateScheduler`` callback that multiplied the *current* LR by 0.1
on every epoch divisible by 7; re-running ``fit()`` restarted the epoch counter and compounded
the decay. Here the LR is a pure function of ``optimizer.iterations``, so it cannot compound.
"""

from __future__ import annotations

import math
from typing import Any

import keras
from keras import ops

from dlresnet.config import TrainConfig

# Variables whose names match these regexes skip weight decay ("no bias decay"):
# BatchNorm gamma/beta and biases. He et al., 2019, "Bag of Tricks", arXiv:1812.01187.
NO_DECAY_VAR_NAMES: tuple[str, ...] = ("gamma", "beta", "bias")


def _warmup(step: Any, base_lr: float, warmup_steps: int) -> Any:
    return base_lr * (step + 1.0) / float(max(warmup_steps, 1))


@keras.saving.register_keras_serializable(package="dlresnet")
class StepDecay(keras.optimizers.schedules.LearningRateSchedule):
    """``base_lr * drop_factor ** floor(epoch / drop_every_epochs)`` with optional linear warmup.

    ``epoch = floor(step / steps_per_epoch)``. During the first ``warmup_epochs`` epochs the LR
    ramps linearly: ``base_lr * (step + 1) / (warmup_epochs * steps_per_epoch)``
    (Goyal et al., 2017, arXiv:1706.02677).
    """

    def __init__(
        self,
        base_lr: float,
        steps_per_epoch: int,
        drop_every_epochs: int = 7,
        drop_factor: float = 0.1,
        warmup_epochs: int = 0,
    ) -> None:
        if steps_per_epoch < 1 or drop_every_epochs < 1:
            raise ValueError("steps_per_epoch and drop_every_epochs must be >= 1")
        self.base_lr = float(base_lr)
        self.steps_per_epoch = int(steps_per_epoch)
        self.drop_every_epochs = int(drop_every_epochs)
        self.drop_factor = float(drop_factor)
        self.warmup_epochs = int(warmup_epochs)

    def __call__(self, step: Any) -> Any:
        step = ops.cast(step, "float32")
        epoch = ops.floor(step / float(self.steps_per_epoch))
        n_drops = ops.floor(epoch / float(self.drop_every_epochs))
        lr = self.base_lr * ops.power(ops.convert_to_tensor(self.drop_factor, "float32"), n_drops)
        if self.warmup_epochs > 0:
            warmup_steps = self.warmup_epochs * self.steps_per_epoch
            lr = ops.where(step < warmup_steps, _warmup(step, self.base_lr, warmup_steps), lr)
        return ops.cast(lr, "float32")

    def get_config(self) -> dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "drop_every_epochs": self.drop_every_epochs,
            "drop_factor": self.drop_factor,
            "warmup_epochs": self.warmup_epochs,
        }


@keras.saving.register_keras_serializable(package="dlresnet")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup over ``warmup_steps`` then cosine decay from ``base_lr`` to
    ``alpha * base_lr`` at ``total_steps`` (constant afterwards)."""

    def __init__(
        self, base_lr: float, total_steps: int, warmup_steps: int = 0, alpha: float = 0.0
    ) -> None:
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)

    def __call__(self, step: Any) -> Any:
        step = ops.cast(step, "float32")
        decay_steps = float(max(self.total_steps - self.warmup_steps, 1))
        progress = ops.clip((step - self.warmup_steps) / decay_steps, 0.0, 1.0)
        cosine = 0.5 * (1.0 + ops.cos(math.pi * progress))
        lr = self.base_lr * (self.alpha + (1.0 - self.alpha) * cosine)
        if self.warmup_steps > 0:
            lr = ops.where(
                step < self.warmup_steps, _warmup(step, self.base_lr, self.warmup_steps), lr
            )
        return ops.cast(lr, "float32")

    def get_config(self) -> dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


def build_lr_schedule(
    cfg: TrainConfig, steps_per_epoch: int, epochs: int, base_lr: float | None = None
) -> keras.optimizers.schedules.LearningRateSchedule | float:
    """Schedule selected by ``cfg.schedule`` (``base_lr`` defaults to ``cfg.learning_rate``)."""
    lr = float(cfg.learning_rate if base_lr is None else base_lr)
    steps_per_epoch = max(int(steps_per_epoch), 1)
    warmup_steps = max(int(cfg.warmup_epochs), 0) * steps_per_epoch
    if cfg.schedule == "step":
        return StepDecay(
            lr,
            steps_per_epoch,
            drop_every_epochs=cfg.drop_every_epochs,
            drop_factor=cfg.drop_factor,
            warmup_epochs=cfg.warmup_epochs,
        )
    if cfg.schedule == "cosine":
        return WarmupCosine(lr, epochs * steps_per_epoch, warmup_steps=warmup_steps)
    if cfg.schedule == "constant":
        if warmup_steps > 0:  # warmup, then flat (alpha=1 makes the cosine part constant)
            return WarmupCosine(lr, epochs * steps_per_epoch, warmup_steps=warmup_steps, alpha=1.0)
        return lr
    raise ValueError(f"Unknown schedule {cfg.schedule!r}")


def build_optimizer(cfg: TrainConfig, learning_rate: Any) -> keras.optimizers.Optimizer:
    """Optimizer selected by ``cfg.optimizer``; always uses ``learning_rate=`` (never ``lr=``).

    If the optimizer applies weight decay (SGD with ``weight_decay > 0``, or AdamW), BatchNorm
    gamma/beta and biases are excluded from it ("no bias decay", see ``NO_DECAY_VAR_NAMES``).
    """
    opt: keras.optimizers.Optimizer
    decays = False
    if cfg.optimizer == "sgd":
        opt = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay if cfg.weight_decay > 0 else None,
        )
        decays = cfg.weight_decay > 0
    elif cfg.optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif cfg.optimizer == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=cfg.weight_decay)
        decays = True
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")
    if decays:
        opt.exclude_from_weight_decay(var_names=list(NO_DECAY_VAR_NAMES))
    return opt
