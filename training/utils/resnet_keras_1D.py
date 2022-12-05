# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is the original resnet code from the tensorflow repository, modified to
# work with 1D data and use group normalization instead of batch normalization.

"""Contains definitions of ResNet and ResNet-RS models."""

from typing import Callable, Optional

# Import libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers.normalizations import GroupNormalization

layers = tf.keras.layers

# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
RESNET_SPECS = {
    10: [
        ("residual", 64, 1),
        ("residual", 128, 1),
        ("residual", 256, 1),
        ("residual", 512, 1),
    ],
    18: [
        ("residual", 64, 2),
        ("residual", 128, 2),
        ("residual", 256, 2),
        ("residual", 512, 2),
    ],
    34: [
        ("residual", 64, 3),
        ("residual", 128, 4),
        ("residual", 256, 6),
        ("residual", 512, 3),
    ],
    50: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 4),
        ("bottleneck", 256, 6),
        ("bottleneck", 512, 3),
    ],
    101: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 4),
        ("bottleneck", 256, 23),
        ("bottleneck", 512, 3),
    ],
    152: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 8),
        ("bottleneck", 256, 36),
        ("bottleneck", 512, 3),
    ],
    200: [
        ("bottleneck", 64, 3),
        ("bottleneck", 128, 24),
        ("bottleneck", 256, 36),
        ("bottleneck", 512, 3),
    ],
    270: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 29),
        ("bottleneck", 256, 53),
        ("bottleneck", 512, 4),
    ],
    350: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 36),
        ("bottleneck", 256, 72),
        ("bottleneck", 512, 4),
    ],
    420: [
        ("bottleneck", 64, 4),
        ("bottleneck", 128, 44),
        ("bottleneck", 256, 87),
        ("bottleneck", 512, 4),
    ],
    "custom_10": [
        ("residual", 16, 1),
        ("residual", 32, 1),
        ("residual", 64, 1),
        ("residual", 128, 1),
    ],
}


def make_divisible(
    value: float,
    divisor: int,
    min_value: Optional[float] = None,
    round_down_protect: bool = True,
) -> int:
    """This is to ensure that all layers have channels that are divisible by 8.
    Args:
      value: A `float` of original value.
      divisor: An `int` of the divisor that need to be checked upon.
      min_value: A `float` of  minimum value threshold.
      round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.
    Returns:
      The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ResidualBlock(tf.keras.layers.Layer):
    """A residual block."""

    def __init__(
        self,
        filters,
        strides,
        use_projection=False,
        resnetd_shortcut=False,
        kernel_initializer="VarianceScaling",
        kernel_regularizer=None,
        bias_regularizer=None,
        activation="relu",
        use_explicit_padding: bool = False,
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        bn_trainable=True,
        square_kernel_size_and_stride=False,
        disable_batchnorm=False,
        use_group_norm=False,
        **kwargs,
    ):
        """Initializes a residual block with BN after convolutions.
        Args:
          filters: An `int` number of filters for the first two convolutions. Note
            that the third and final convolution will use 4 times as many filters.
          strides: An `int` block stride. If greater than 1, this block will
            ultimately downsample the input.
          use_projection: A `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
          resnetd_shortcut: A `bool` if True, apply the resnetd style modification
            to the shortcut connection. Not implemented in residual blocks.
          kernel_initializer: A `str` of kernel_initializer for convolutional
            layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv1D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
            Default to None.
          activation: A `str` name of the activation function.
          use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
            inputs so that the output dimensions are the same as if 'SAME' padding
            were used.
          use_sync_bn: A `bool`. If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        super(ResidualBlock, self).__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._use_projection = use_projection
        self._resnetd_shortcut = resnetd_shortcut
        self._use_explicit_padding = use_explicit_padding
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._square_kernel_size_and_stride = square_kernel_size_and_stride
        self._disable_batchnorm = disable_batchnorm
        self._use_group_norm = use_group_norm

        if use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        elif not use_group_norm:
            self._norm = tf.keras.layers.BatchNormalization
        else:
            self._norm = GroupNormalization

        if tf.keras.backend.image_data_format() == "channels_last":
            self._bn_axis = -1
        else:
            self._bn_axis = 1
        self._activation_fn = keras.layers.Activation(activation)
        self._bn_trainable = bn_trainable

        self.sub_layers = []

    def build(self, input_shape):
        if self._use_projection:
            self._shortcut = tf.keras.layers.Conv1D(
                filters=self._filters,
                kernel_size=1,
                strides=self._strides,
                use_bias=False,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )
            self.sub_layers.append(self._shortcut)

            if not self._use_group_norm:
                self._norm0 = self._norm(
                    axis=self._bn_axis,
                    momentum=self._norm_momentum,
                    epsilon=self._norm_epsilon,
                    trainable=self._bn_trainable,
                )
            else:
                self._norm0 = self._norm()

            self.sub_layers.append(self._norm0)

        conv1_padding = "same"
        # explicit padding here is added for centernet
        if self._use_explicit_padding:
            self._pad = keras.layers.ZeroPadding1D(padding=(1,))
            self.sub_layers.append(self._pad)
            conv1_padding = "valid"

        self._conv1 = tf.keras.layers.Conv1D(
            filters=self._filters,
            kernel_size=3 if not self._square_kernel_size_and_stride else 3**2,
            strides=self._strides,
            padding=conv1_padding,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self.sub_layers.append(self._conv1)

        if not self._use_group_norm:
            self._norm1 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable,
            )
        else:
            self._norm1 = self._norm()

        self.sub_layers.append(self._norm1)

        self._conv2 = tf.keras.layers.Conv1D(
            filters=self._filters,
            kernel_size=3 if not self._square_kernel_size_and_stride else 3**2,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self.sub_layers.append(self._conv2)

        if not self._use_group_norm:
            self._norm2 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable,
            )
        else:
            self._norm2 = self._norm()

        self.sub_layers.append(self._norm2)

        super(ResidualBlock, self).build(input_shape)

    def get_config(self):
        config = {
            "filters": self._filters,
            "strides": self._strides,
            "use_projection": self._use_projection,
            "resnetd_shortcut": self._resnetd_shortcut,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "activation": self._activation,
            "use_explicit_padding": self._use_explicit_padding,
            "use_sync_bn": self._use_sync_bn,
            "norm_momentum": self._norm_momentum,
            "norm_epsilon": self._norm_epsilon,
            "bn_trainable": self._bn_trainable,
            "square_kernel_size_and_stride": self._square_kernel_size_and_stride,
            "disable_batchnorm": self._disable_batchnorm,
            "use_group_norm": self._use_group_norm,
        }
        base_config = super(ResidualBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            shortcut = self._shortcut(shortcut)
            if not self._disable_batchnorm:
                shortcut = self._norm0(shortcut)

        if self._use_explicit_padding:
            inputs = self._pad(inputs)
        x = self._conv1(inputs)
        if not self._disable_batchnorm:
            x = self._norm1(x)
        x = self._activation_fn(x)

        x = self._conv2(x)
        if not self._disable_batchnorm:
            x = self._norm2(x)

        return self._activation_fn(x + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
    """A standard bottleneck block."""

    def __init__(
        self,
        filters,
        strides,
        dilation_rate=1,
        use_projection=False,
        resnetd_shortcut=False,
        kernel_initializer="VarianceScaling",
        kernel_regularizer=None,
        bias_regularizer=None,
        activation="relu",
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        bn_trainable=True,
        square_kernel_size_and_stride=False,
        disable_batchnorm=False,
        use_group_norm=False,
        **kwargs,
    ):
        """Initializes a standard bottleneck block with BN after convolutions.
        Args:
          filters: An `int` number of filters for the first two convolutions. Note
            that the third and final convolution will use 4 times as many filters.
          strides: An `int` block stride. If greater than 1, this block will
            ultimately downsample the input.
          dilation_rate: An `int` dilation_rate of convolutions. Default to 1.
          use_projection: A `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
          resnetd_shortcut: A `bool`. If True, apply the resnetd style modification
            to the shortcut connection.
          kernel_initializer: A `str` of kernel_initializer for convolutional
            layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv1D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
            Default to None.
          activation: A `str` name of the activation function.
          use_sync_bn: A `bool`. If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        super(BottleneckBlock, self).__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._dilation_rate = dilation_rate
        self._use_projection = use_projection
        self._resnetd_shortcut = resnetd_shortcut
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._square_kernel_size_and_stride = square_kernel_size_and_stride
        self._disable_batchnorm = disable_batchnorm
        self._use_group_norm = use_group_norm

        if use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        elif not use_group_norm:
            self._norm = tf.keras.layers.BatchNormalization
        else:
            self._norm = GroupNormalization

        if tf.keras.backend.image_data_format() == "channels_last":
            self._bn_axis = -1
        else:
            self._bn_axis = 1
        self._bn_trainable = bn_trainable

        self.sub_layers = []

    def build(self, input_shape):
        if self._use_projection:
            if self._resnetd_shortcut:
                self._shortcut0 = tf.keras.layers.AveragePooling1D(
                    pool_size=2, strides=self._strides, padding="same"
                )
                self.sub_layers.append(self._shortcut0)
                self._shortcut1 = tf.keras.layers.Conv1D(
                    filters=self._filters * 4,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    kernel_initializer=self._kernel_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                )
                self.sub_layers.append(self._shortcut1)
            else:
                self._shortcut = tf.keras.layers.Conv1D(
                    filters=self._filters * 4,
                    kernel_size=1,
                    strides=self._strides,
                    use_bias=False,
                    kernel_initializer=self._kernel_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                )
                self.sub_layers.append(self._shortcut)

            if not self._use_group_norm:
                self._norm0 = self._norm(
                    axis=self._bn_axis,
                    momentum=self._norm_momentum,
                    epsilon=self._norm_epsilon,
                    trainable=self._bn_trainable,
                )
            else:
                self._norm0 = self._norm()

            self.sub_layers.append(self._norm0)

        self._conv1 = tf.keras.layers.Conv1D(
            filters=self._filters,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self.sub_layers.append(self._conv1)

        if not self._use_group_norm:
            self._norm1 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable,
            )
        else:
            self._norm1 = self._norm()

        self.sub_layers.append(self._norm1)
        self._activation1 = keras.layers.Activation(self._activation)
        self.sub_layers.append(self._activation1)

        self._conv2 = tf.keras.layers.Conv1D(
            filters=self._filters,
            kernel_size=3 if not self._square_kernel_size_and_stride else 3**2,
            strides=self._strides,
            dilation_rate=self._dilation_rate,
            padding="same",
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self.sub_layers.append(self._conv2)

        if not self._use_group_norm:
            self._norm2 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable,
            )
        else:
            self._norm2 = self._norm()

        self.sub_layers.append(self._norm2)
        self._activation2 = keras.layers.Activation(self._activation)
        self.sub_layers.append(self._activation2)

        self._conv3 = tf.keras.layers.Conv1D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self.sub_layers.append(self._conv3)

        if not self._use_group_norm:
            self._norm3 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable,
            )
        else:
            self._norm3 = self._norm()

        self.sub_layers.append(self._norm3)
        self._activation3 = keras.layers.Activation(self._activation)
        self.sub_layers.append(self._activation3)

        self._add = tf.keras.layers.Add()
        self.sub_layers.append(self._add)

        super(BottleneckBlock, self).build(input_shape)

    def get_config(self):
        config = {
            "filters": self._filters,
            "strides": self._strides,
            "dilation_rate": self._dilation_rate,
            "use_projection": self._use_projection,
            "resnetd_shortcut": self._resnetd_shortcut,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "activation": self._activation,
            "use_sync_bn": self._use_sync_bn,
            "norm_momentum": self._norm_momentum,
            "norm_epsilon": self._norm_epsilon,
            "bn_trainable": self._bn_trainable,
            "square_kernel_size_and_stride": self._square_kernel_size_and_stride,
            "disable_batchnorm": self._disable_batchnorm,
            "use_group_norm": self._use_group_norm,
        }
        base_config = super(BottleneckBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            if self._resnetd_shortcut:
                shortcut = self._shortcut0(shortcut)
                shortcut = self._shortcut1(shortcut)
            else:
                shortcut = self._shortcut(shortcut)
            if not self._disable_batchnorm:
                shortcut = self._norm0(shortcut)

        x = self._conv1(inputs)
        if not self._disable_batchnorm:
            x = self._norm1(x)
        x = self._activation1(x)

        x = self._conv2(x)
        if not self._disable_batchnorm:
            x = self._norm2(x)
        x = self._activation2(x)

        x = self._conv3(x)
        if not self._disable_batchnorm:
            x = self._norm3(x)

        x = self._add([x, shortcut])
        return self._activation3(x)


class ResNet(tf.keras.Model):
    """Creates ResNet and ResNet-RS family models.
    This implements the Deep Residual Network from:
      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
      Deep Residual Learning for Image Recognition.
      (https://arxiv.org/pdf/1512.03385) and
      Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas,
      Tsung-Yi Lin, Jonathon Shlens, Barret Zoph.
      Revisiting ResNets: Improved Training and Scaling Strategies.
      (https://arxiv.org/abs/2103.07579).
    """

    def __init__(
        self,
        model_id: int,
        input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
            shape=[None, None, None, 3]
        ),
        depth_multiplier: float = 1.0,
        stem_type: str = "v0",
        resnetd_shortcut: bool = False,
        replace_stem_max_pool: bool = False,
        scale_stem: bool = True,
        activation: str = "relu",
        use_sync_bn: bool = False,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        kernel_initializer: str = "VarianceScaling",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bn_trainable: bool = True,
        square_kernel_size_and_stride=False,
        disable_batchnorm=False,
        use_group_norm=False,
        **kwargs,
    ):
        """Initializes a ResNet model.
        Args:
          model_id: An `int` of the depth of ResNet backbone model.
          input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
          depth_multiplier: A `float` of the depth multiplier to uniformaly scale up
            all layers in channel size. This argument is also referred to as
            `width_multiplier` in (https://arxiv.org/abs/2103.07579).
          stem_type: A `str` of stem type of ResNet. Default to `v0`. If set to
            `v1`, use ResNet-D type stem (https://arxiv.org/abs/1812.01187).
          resnetd_shortcut: A `bool` of whether to use ResNet-D shortcut in
            downsampling blocks.
          replace_stem_max_pool: A `bool` of whether to replace the max pool in stem
            with a stride-2 conv,
          scale_stem: A `bool` of whether to scale stem layers.
          activation: A `str` name of the activation function.
          use_sync_bn: If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A small `float` added to variance to avoid dividing by zero.
          kernel_initializer: A str for kernel initializer of convolutional layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv1D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv1D.
            Default to None.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        self._model_id = model_id
        self._input_specs = input_specs
        self._depth_multiplier = depth_multiplier
        self._stem_type = stem_type
        self._resnetd_shortcut = resnetd_shortcut
        self._replace_stem_max_pool = replace_stem_max_pool
        self._scale_stem = scale_stem
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        if use_sync_bn:
            self._norm = layers.experimental.SyncBatchNormalization
        elif not use_group_norm:
            self._norm = layers.BatchNormalization
        else:
            self._norm = GroupNormalization
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._bn_trainable = bn_trainable
        self._square_kernel_size_and_stride = square_kernel_size_and_stride
        self._disable_batchnorm = disable_batchnorm
        self._use_group_norm = use_group_norm

        if tf.keras.backend.image_data_format() == "channels_last":
            bn_axis = -1
        else:
            bn_axis = 1

        # Build ResNet.
        inputs = tf.keras.Input(shape=input_specs.shape[1:])

        stem_depth_multiplier = self._depth_multiplier if scale_stem else 1.0
        if stem_type == "v0":
            x = layers.Conv1D(
                filters=int(64 * stem_depth_multiplier),
                kernel_size=7,
                strides=2,
                use_bias=False,
                padding="same",
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )(inputs)
            if not self._disable_batchnorm:
                if not self._use_group_norm:
                    x = self._norm(
                        axis=bn_axis,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=bn_trainable,
                    )(x)
                else:
                    x = self._norm()(x)
            x = keras.layers.Activation(activation)(x)
        elif stem_type == "v1":
            x = layers.Conv1D(
                filters=int(32 * stem_depth_multiplier),
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding="same",
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )(inputs)
            if not self._disable_batchnorm:
                if not self._use_group_norm:
                    x = self._norm(
                        axis=bn_axis,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=bn_trainable,
                    )(x)
                else:
                    x = self._norm()(x)

            x = keras.layers.Activation(activation)(x)
            x = layers.Conv1D(
                filters=int(32 * stem_depth_multiplier),
                kernel_size=3,
                strides=1,
                use_bias=False,
                padding="same",
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )(x)
            if not self._disable_batchnorm:
                if not self._use_group_norm:
                    x = self._norm(
                        axis=bn_axis,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=bn_trainable,
                    )(x)
                else:
                    x = self._norm()(x)

            x = keras.layers.Activation(activation)(x)
            x = layers.Conv1D(
                filters=int(64 * stem_depth_multiplier),
                kernel_size=3,
                strides=1,
                use_bias=False,
                padding="same",
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )(x)
            if not self._disable_batchnorm:
                if not self._use_group_norm:
                    x = self._norm(
                        axis=bn_axis,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=bn_trainable,
                    )(x)
                else:
                    x = self._norm()(x)

            x = keras.layers.Activation(activation)(x)
        else:
            raise ValueError("Stem type {} not supported.".format(stem_type))

        if replace_stem_max_pool:
            x = layers.Conv1D(
                filters=int(64 * self._depth_multiplier),
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding="same",
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
            )(x)
            if not self._disable_batchnorm:
                if not self._use_group_norm:
                    x = self._norm(
                        axis=bn_axis,
                        momentum=norm_momentum,
                        epsilon=norm_epsilon,
                        trainable=bn_trainable,
                    )(x)
                else:
                    x = self._norm()(x)

            x = keras.layers.Activation(activation)(x)
        else:
            x = layers.MaxPool1D(pool_size=3, strides=2, padding="same")(x)

        endpoints = {}
        for i, spec in enumerate(RESNET_SPECS[model_id]):
            if spec[0] == "residual":
                block_fn = ResidualBlock
            elif spec[0] == "bottleneck":
                block_fn = BottleneckBlock
            else:
                raise ValueError("Block fn `{}` is not supported.".format(spec[0]))
            x = self._block_group(
                inputs=x,
                filters=int(spec[1] * self._depth_multiplier),
                strides=(
                    1
                    if i == 0
                    else (2 if not square_kernel_size_and_stride else 2**2)
                ),
                block_fn=block_fn,
                block_repeats=spec[2],
                name="block_group_l{}".format(i + 2),
            )
            endpoints[str(i + 2)] = x

        self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

        super(ResNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

    def _block_group(
        self,
        inputs: tf.Tensor,
        filters: int,
        strides: int,
        block_fn: Callable[..., tf.keras.layers.Layer],
        block_repeats: int = 1,
        name: str = "block_group",
    ):
        """Creates one group of blocks for the ResNet model.
        Args:
          inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
          filters: An `int` number of filters for the first convolution of the
            layer.
          strides: An `int` stride to use for the first convolution of the layer.
            If greater than 1, this layer will downsample the input.
          block_fn: The type of block group. Either `nn_blocks.ResidualBlock` or
            `nn_blocks.BottleneckBlock`.
          block_repeats: An `int` number of blocks contained in the layer.
          name: A `str` name for the block.
        Returns:
          The output `tf.Tensor` of the block layer.
        """
        x = block_fn(
            filters=filters,
            strides=strides,
            use_projection=True,
            resnetd_shortcut=self._resnetd_shortcut,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=self._activation,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            bn_trainable=self._bn_trainable,
            square_kernel_size_and_stride=self._square_kernel_size_and_stride,
            disable_batchnorm=self._disable_batchnorm,
            use_group_norm=self._use_group_norm,
        )(inputs)

        for _ in range(1, block_repeats):
            x = block_fn(
                filters=filters,
                strides=1,
                use_projection=False,
                resnetd_shortcut=self._resnetd_shortcut,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activation=self._activation,
                use_sync_bn=self._use_sync_bn,
                norm_momentum=self._norm_momentum,
                norm_epsilon=self._norm_epsilon,
                bn_trainable=self._bn_trainable,
                square_kernel_size_and_stride=self._square_kernel_size_and_stride,
                disable_batchnorm=self._disable_batchnorm,
                use_group_norm=self._use_group_norm,
            )(x)

        return tf.keras.layers.Activation("linear", name=name)(x)

    def get_config(self):
        config_dict = {
            "model_id": self._model_id,
            "depth_multiplier": self._depth_multiplier,
            "stem_type": self._stem_type,
            "resnetd_shortcut": self._resnetd_shortcut,
            "replace_stem_max_pool": self._replace_stem_max_pool,
            "activation": self._activation,
            "scale_stem": self._scale_stem,
            "use_sync_bn": self._use_sync_bn,
            "norm_momentum": self._norm_momentum,
            "norm_epsilon": self._norm_epsilon,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "bn_trainable": self._bn_trainable,
            "square_kernel_size_and_stride": self._square_kernel_size_and_stride,
            "disable_batchnorm": self._disable_batchnorm,
            "use_group_norm": self._use_group_norm,
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @property
    def output_specs(self):
        """A dict of {level: TensorShape} pairs for the model output."""
        return self._output_specs


if __name__ == "__main__":

    resnet_model = ResNet(10, keras.layers.InputSpec(shape=(None, 8501, 1)))

    predictions = keras.layers.Flatten()(resnet_model.layers[-1].output)
    predictions = keras.layers.Dense(100)(predictions)

    model = keras.Model(resnet_model.inputs, outputs=predictions)

    model.summary()

    keras.utils.plot_model(model, show_shapes=True)
