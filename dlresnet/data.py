"""Image-folder datasets (``tf.data``) and Keras 3 augmentation pipeline.

Images are yielded as float32 RGB in ``[0, 255]``; preprocessing lives inside the model
(:class:`dlresnet.layers.CaffePreprocess`), so the data pipeline never has to match it by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import tensorflow as tf

from dlresnet.config import AugmentationConfig, DataConfig


@dataclass
class Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset | None
    class_names: list[str]
    input_shape: tuple[int, int, int]
    steps_per_epoch: int


def image_folder_dataset(
    directory: str | Path,
    image_size: tuple[int, int],
    batch_size: int | None,
    *,
    shuffle: bool,
    seed: int | None = None,
    class_names: list[str] | None = None,
) -> tuple[tf.data.Dataset, list[str]]:
    """Load ``directory/<class>/<image>`` as a batched ``(images, int_labels)`` dataset."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Dataset directory not found: {directory}")
    ds = keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=tuple(image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return ds, list(ds.class_names)


def load_datasets(cfg: DataConfig) -> Datasets:
    """Load the train/val (and optional test) splits under ``cfg.data_dir``."""
    root = Path(cfg.data_dir)
    splits = {"train": cfg.train_split, "val": cfg.val_split}
    if cfg.test_split:
        splits["test"] = cfg.test_split
    for role, split in splits.items():
        if not (root / split).is_dir():
            raise ValueError(
                f"Missing {role} split directory: {root / split} "
                f"(expected {root}/<split>/<class_name>/<image files>)"
            )

    def load(split: str, shuffle: bool) -> tuple[tf.data.Dataset, list[str]]:
        return image_folder_dataset(
            root / split, cfg.image_size, cfg.batch_size, shuffle=shuffle, seed=cfg.seed
        )

    if cfg.cache:
        # Cache decoded, unshuffled images first, then reshuffle every epoch. Caching the
        # pipeline returned by image_dataset_from_directory (which shuffles internally) would
        # freeze the first epoch's order. Intended for datasets that fit in memory.
        train, class_names = image_folder_dataset(
            root / cfg.train_split, cfg.image_size, None, shuffle=False
        )
        num_images = int(train.cardinality().numpy())
        train = (
            train.cache()
            .shuffle(num_images, seed=cfg.seed, reshuffle_each_iteration=True)
            .batch(cfg.batch_size)
        )
    else:
        train, class_names = load(cfg.train_split, shuffle=True)
    val, val_names = load(cfg.val_split, shuffle=False)
    test, test_names = load(cfg.test_split, shuffle=False) if cfg.test_split else (None, None)
    for role, names in (("val", val_names), ("test", test_names)):
        if names is not None and names != class_names:
            raise ValueError(
                f"Class names of the {role} split differ from the train split: "
                f"train={class_names}, {role}={names}"
            )

    steps_per_epoch = int(train.cardinality().numpy())

    def finish(ds: tf.data.Dataset | None) -> tf.data.Dataset | None:
        if ds is None:
            return None
        if cfg.cache:
            ds = ds.cache()  # val/test are not shuffled, so a fixed order is correct
        return ds.prefetch(tf.data.AUTOTUNE)

    return Datasets(
        train=train.prefetch(tf.data.AUTOTUNE),
        val=finish(val),
        test=finish(test),
        class_names=class_names,
        input_shape=(int(cfg.image_size[0]), int(cfg.image_size[1]), 3),
        steps_per_epoch=steps_per_epoch,
    )


def build_augmentation(cfg: AugmentationConfig, seed: int | None = None) -> keras.Sequential | None:
    """Random augmentation layers (active only in training mode); ``None`` if all disabled."""

    def s(offset: int) -> int | None:
        return None if seed is None else seed + offset

    layers: list[keras.layers.Layer] = []
    if cfg.horizontal_flip:
        layers.append(keras.layers.RandomFlip("horizontal", seed=s(0), name="random_flip"))
    if cfg.rotation_degrees > 0:
        layers.append(
            keras.layers.RandomRotation(
                cfg.rotation_degrees / 360.0, fill_mode="reflect", seed=s(1), name="random_rotation"
            )
        )
    if cfg.translation > 0:
        layers.append(
            keras.layers.RandomTranslation(
                cfg.translation,
                cfg.translation,
                fill_mode="reflect",
                seed=s(2),
                name="random_translation",
            )
        )
    if cfg.zoom > 0:
        layers.append(
            keras.layers.RandomZoom(cfg.zoom, fill_mode="reflect", seed=s(3), name="random_zoom")
        )
    if cfg.contrast > 0:
        layers.append(
            keras.layers.RandomContrast(
                cfg.contrast, value_range=(0, 255), seed=s(4), name="random_contrast"
            )
        )
    if not layers:
        return None
    return keras.Sequential(layers, name="augmentation")
