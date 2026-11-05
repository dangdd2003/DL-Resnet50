"""Custom Keras 3 layers: in-model Caffe preprocessing and the ResNet-50 bottleneck block.

Both layers are registered for native ``.keras`` serialisation, so a saved model can be loaded
with plain ``keras.models.load_model(path)`` after ``import dlresnet``.
"""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

# ImageNet channel means in BGR order, exactly as used by
# ``keras.applications.resnet50.preprocess_input`` (mode="caffe").
CAFFE_BGR_MEAN: tuple[float, float, float] = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package="dlresnet")
class CaffePreprocess(keras.layers.Layer):
    """RGB ``[0, 255]`` -> BGR, minus the ImageNet channel mean (no scaling).

    Numerically identical to ``keras.applications.resnet50.preprocess_input``. Putting it inside
    the model means training, evaluation and inference can never use different preprocessing.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: Any) -> Any:
        x = ops.cast(inputs, self.compute_dtype)
        x = ops.flip(x, axis=-1)  # RGB -> BGR
        mean = ops.convert_to_tensor(CAFFE_BGR_MEAN, dtype=self.compute_dtype)
        return x - mean

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return tuple(input_shape)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@keras.saving.register_keras_serializable(package="dlresnet")
class BottleneckBlock(keras.layers.Layer):
    """ResNet v1 bottleneck block (He et al., 2016, arXiv:1512.03385).

    ``conv1x1(f) -> BN -> ReLU -> conv3x3(f) -> BN -> ReLU -> conv1x1(4f) -> BN``, then
    ``+ shortcut`` and a single ReLU *after* the addition (the notebooks applied a ReLU before
    the add, which turns the residual branch into a non-negative-only correction).

    Args:
        filters: Bottleneck width; the block outputs ``4 * filters`` channels.
        stride: Spatial stride of the block (1 or 2).
        project_shortcut: Force (True) / forbid (False) a 1x1 conv + BN projection shortcut.
            ``None`` decides in ``build()``: project iff ``stride != 1`` or the input channel count
            differs from ``4 * filters``.
        zero_init_residual: Initialise the last BN's gamma to zero so the block starts as identity
            (Goyal et al., 2017, arXiv:1706.02677).
        stride_in_3x3: ``False`` = original v1 (stride on the first 1x1, like
            ``keras.applications.ResNet50``); ``True`` = "ResNet v1.5" (stride on the 3x3).
        bn_momentum: BatchNormalization momentum.
        bn_epsilon: BatchNormalization epsilon.
    """

    expansion = 4

    def __init__(
        self,
        filters: int,
        stride: int = 1,
        project_shortcut: bool | None = None,
        zero_init_residual: bool = True,
        stride_in_3x3: bool = False,
        bn_momentum: float = 0.9,
        bn_epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.stride = int(stride)
        self.project_shortcut = project_shortcut
        self.zero_init_residual = bool(zero_init_residual)
        self.stride_in_3x3 = bool(stride_in_3x3)
        self.bn_momentum = float(bn_momentum)
        self.bn_epsilon = float(bn_epsilon)
        self.out_channels = self.expansion * self.filters

        s1, s3 = (1, self.stride) if self.stride_in_3x3 else (self.stride, 1)
        self.conv1 = self._conv(self.filters, 1, s1, "conv1")
        self.bn1 = self._bn("bn1")
        self.conv2 = self._conv(self.filters, 3, s3, "conv2")
        self.bn2 = self._bn("bn2")
        self.conv3 = self._conv(self.out_channels, 1, 1, "conv3")
        self.bn3 = self._bn("bn3", gamma_zero=self.zero_init_residual)
        # Projection shortcut needs the input channel count, so it is created in build().
        self.shortcut_conv: keras.layers.Conv2D | None = None
        self.shortcut_bn: keras.layers.BatchNormalization | None = None

    def _conv(self, filters: int, kernel: int, stride: int, name: str) -> keras.layers.Conv2D:
        return keras.layers.Conv2D(
            filters,
            kernel,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            dtype=self.dtype_policy,
            name=name,
        )

    def _bn(self, name: str, gamma_zero: bool = False) -> keras.layers.BatchNormalization:
        return keras.layers.BatchNormalization(
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            gamma_initializer="zeros" if gamma_zero else "ones",
            dtype=self.dtype_policy,
            name=name,
        )

    def _needs_projection(self, in_channels: int) -> bool:
        if self.project_shortcut is not None:
            return bool(self.project_shortcut)
        return self.stride != 1 or in_channels != self.out_channels

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        input_shape = tuple(input_shape)
        in_channels = int(input_shape[-1])
        if self._needs_projection(in_channels):
            self.shortcut_conv = self._conv(self.out_channels, 1, self.stride, "shortcut_conv")
            self.shortcut_bn = self._bn("shortcut_bn")
            self.shortcut_conv.build(input_shape)
            self.shortcut_bn.build(self.shortcut_conv.compute_output_shape(input_shape))
        elif self.stride != 1:
            raise ValueError("An identity shortcut requires stride == 1.")
        # Build every sub-layer explicitly so all weights exist before a .keras load.
        shape = input_shape
        for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)):
            conv.build(shape)
            shape = conv.compute_output_shape(shape)
            bn.build(shape)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        x = ops.relu(self.bn1(self.conv1(inputs), training=training))
        x = ops.relu(self.bn2(self.conv2(x), training=training))
        x = self.bn3(self.conv3(x), training=training)  # no ReLU before the residual add
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs), training=training)
        else:
            shortcut = inputs
        return ops.relu(x + ops.cast(shortcut, x.dtype))

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        n, h, w, _ = tuple(input_shape)

        def down(d: int | None) -> int | None:
            return None if d is None else -(-d // self.stride)  # ceil, "same" padding

        return (n, down(h), down(w), self.out_channels)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "project_shortcut": self.project_shortcut,
                "zero_init_residual": self.zero_init_residual,
                "stride_in_3x3": self.stride_in_3x3,
                "bn_momentum": self.bn_momentum,
                "bn_epsilon": self.bn_epsilon,
            }
        )
        return config
