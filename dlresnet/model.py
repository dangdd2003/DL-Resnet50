"""Model builders: ResNet-50 from scratch (single source of truth) and transfer learning."""

from __future__ import annotations

from typing import Any

import keras

from dlresnet.config import ModelConfig
from dlresnet.layers import BottleneckBlock, CaffePreprocess

# (bottleneck width, number of blocks, stride of the first block) for conv2_x .. conv5_x.
RESNET50_STAGES: tuple[tuple[int, int, int], ...] = (
    (64, 3, 1),
    (128, 4, 2),
    (256, 6, 2),
    (512, 3, 2),
)


def _head(x: Any, num_classes: int, dropout: float) -> Any:
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout, name="dropout")(x)
    # float32 output keeps softmax/loss numerically stable under mixed precision.
    return keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)


def _stem(inputs: Any, augmentation: keras.Sequential | None, include_preprocessing: bool) -> Any:
    x = inputs
    if augmentation is not None:
        x = augmentation(x)
    if include_preprocessing:
        x = CaffePreprocess(name="caffe_preprocess")(x)
    return x


def build_resnet50(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    zero_init_residual: bool = True,
    stride_in_3x3: bool = False,
    dropout: float = 0.0,
    augmentation: keras.Sequential | None = None,
    include_preprocessing: bool = True,
    name: str = "resnet50",
) -> keras.Model:
    """Build ResNet-50 (He et al., 2016) with the functional API.

    Input -> [augmentation] -> [CaffePreprocess] -> conv7x7/2 -> BN -> ReLU -> maxpool3x3/2
    -> bottleneck stages [3, 4, 6, 3] -> global average pool -> [dropout] -> softmax.

    With ``input_shape=(224, 224, 3)`` and ``num_classes=1000`` this has 25,557,032 trainable and
    53,120 non-trainable (BatchNorm moving statistics) parameters.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    inputs = keras.Input(shape=tuple(input_shape), name="image")
    x = _stem(inputs, augmentation, include_preprocessing)

    x = keras.layers.Conv2D(
        64,
        7,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="conv1",
    )(x)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1_bn")(x)
    x = keras.layers.ReLU(name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    for stage_idx, (filters, blocks, stride) in enumerate(RESNET50_STAGES, start=2):
        for block_idx in range(blocks):
            x = BottleneckBlock(
                filters,
                stride=stride if block_idx == 0 else 1,
                zero_init_residual=zero_init_residual,
                stride_in_3x3=stride_in_3x3,
                name=f"conv{stage_idx}_block{block_idx + 1}",
            )(x)

    outputs = _head(x, num_classes, dropout)
    return keras.Model(inputs, outputs, name=name)


def build_transfer_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    *,
    weights: str | None = "imagenet",
    freeze_backbone: bool = True,
    dropout: float = 0.0,
    augmentation: keras.Sequential | None = None,
    name: str = "resnet50_transfer",
) -> keras.Model:
    """ImageNet ResNet-50 backbone (``keras.applications``) with a new softmax head.

    The backbone is called with ``training=False`` so its BatchNorm layers stay in inference mode
    even after (partial) unfreezing, as recommended for fine-tuning.
    """
    inputs = keras.Input(shape=tuple(input_shape), name="image")
    x = _stem(inputs, augmentation, include_preprocessing=True)
    backbone = keras.applications.ResNet50(
        include_top=False, weights=weights, input_shape=tuple(input_shape), name="backbone"
    )
    backbone.trainable = not freeze_backbone
    x = backbone(x, training=False)
    outputs = _head(x, num_classes, dropout)
    return keras.Model(inputs, outputs, name=name)


def unfreeze_top_layers(model: keras.Model, n: int) -> int:
    """Unfreeze the last ``n`` layers of the ``"backbone"`` sub-model (BatchNorm stays frozen).

    Returns the number of backbone layers that now have trainable weights. Recompile the model
    afterwards for the change to take effect.
    """
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    layers = backbone.layers
    cutoff = max(len(layers) - max(int(n), 0), 0)
    for i, layer in enumerate(layers):
        trainable = i >= cutoff and not isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = trainable
    return sum(1 for layer in layers if layer.trainable and layer.trainable_weights)


def build_model(
    cfg: ModelConfig,
    input_shape: tuple[int, int, int],
    num_classes: int,
    augmentation: keras.Sequential | None = None,
) -> keras.Model:
    """Build the architecture selected by ``cfg.arch``."""
    if cfg.arch == "resnet50":
        return build_resnet50(
            input_shape,
            num_classes,
            zero_init_residual=cfg.zero_init_residual,
            stride_in_3x3=cfg.stride_in_3x3,
            dropout=cfg.dropout,
            augmentation=augmentation,
        )
    if cfg.arch == "resnet50_transfer":
        return build_transfer_model(
            input_shape,
            num_classes,
            weights=cfg.weights,
            freeze_backbone=cfg.freeze_backbone,
            dropout=cfg.dropout,
            augmentation=augmentation,
        )
    raise ValueError(f"Unknown model.arch {cfg.arch!r}")
