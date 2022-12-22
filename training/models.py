import tensorflow.keras as keras
import tensorflow as tf
from training.utils.resnet_keras_1D import ResNet

# All models that were used in out publication.
# See the publication for a detailed description and illustrations.


def build_model_park_small(
    number_of_input_values=8501,
    number_of_output_labels=230,
    use_dropout=False,
    lr=0.001,
):
    """Small 7-label version of CNN by Park et al.

    Args:
        number_of_input_values (int, optional): Number of input values. Defaults to 8501.
        number_of_output_labels (int, optional): Number of output labels. Defaults to 230.
        use_dropout (bool, optional): Whether or not to turn on dropout. Defaults to False.
        lr (float, optional): learning rate of the Adam optimizer. Defaults to 0.001.

    Returns:
        model: keras model
    """

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            80,
            100,
            strides=5,
            padding="same",
            input_shape=(number_of_input_values, 1),
        )
    )  # add convolution layer
    model.add(keras.layers.Activation("relu"))  # activation

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=2))  # pooling layer

    model.add(keras.layers.Convolution1D(80, 50, strides=5, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))

    model.add(keras.layers.Convolution1D(80, 25, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(700))  # This is the smaller Park version!
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(70))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(number_of_output_labels))

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    model.summary()

    return model


def build_model_park_medium(
    number_of_input_values=8501,
    number_of_output_labels=230,
    use_dropout=False,
    lr=0.001,
):
    """Medium 101-label version of CNN by Park et al.

    Args:
        number_of_input_values (int, optional): Number of input values. Defaults to 8501.
        number_of_output_labels (int, optional): Number of output labels. Defaults to 230.
        use_dropout (bool, optional): Whether or not to turn on dropout. Defaults to False.
        lr (float, optional): learning rate of the Adam optimizer. Defaults to 0.001.

    Returns:
        model: keras model
    """

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            80,
            100,
            strides=5,
            padding="same",
            input_shape=(number_of_input_values, 1),
        )
    )  # add convolution layer
    model.add(keras.layers.Activation("relu"))  # activation

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=2))  # pooling layer

    model.add(keras.layers.Convolution1D(80, 50, strides=5, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))

    model.add(keras.layers.Convolution1D(80, 25, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4040))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(202))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(number_of_output_labels))

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    model.summary()

    return model


def build_model_park_big(
    number_of_input_values=8501,
    number_of_output_labels=230,
    use_dropout=False,
    lr=0.001,
):
    """Big 230-label version of CNN by Park et al.

    Args:
        number_of_input_values (int, optional): Number of input values. Defaults to 8501.
        number_of_output_labels (int, optional): Number of output labels. Defaults to 230.
        use_dropout (bool, optional): Whether or not to turn on dropout. Defaults to False.
        lr (float, optional): learning rate of the Adam optimizer. Defaults to 0.001.

    Returns:
        model: keras model
    """

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            80,
            100,
            strides=5,
            padding="same",
            input_shape=(number_of_input_values, 1),
        )
    )  # add convolution layer
    model.add(keras.layers.Activation("relu"))  # activation

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=2))  # pooling layer

    model.add(keras.layers.Convolution1D(80, 50, strides=5, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))

    model.add(keras.layers.Convolution1D(80, 25, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(2300))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1150))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(number_of_output_labels))

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    model.summary()

    return model


def build_model_park_extended(
    number_of_input_values=8501,
    number_of_output_labels=230,
    use_dropout=False,
    lr=0.001,
):
    """Extended version of CNN by Park et al. This includes one additional convolution,
    more convolution filter channels, and more dense weights in the end.

    Args:
        number_of_input_values (int, optional): Number of input values. Defaults to 8501.
        number_of_output_labels (int, optional): Number of output labels. Defaults to 230.
        use_dropout (bool, optional): Whether or not to turn on dropout. Defaults to False.
        lr (float, optional): learning rate of the Adam optimizer. Defaults to 0.001.

    Returns:
        model: keras model
    """

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            120,
            100,
            strides=2,
            padding="same",
            input_shape=(number_of_input_values, 1),
        )
    )  # add convolution layer
    model.add(keras.layers.Activation("relu"))  # activation

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=2))  # pooling layer

    model.add(keras.layers.Convolution1D(120, 75, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=None))

    model.add(keras.layers.Convolution1D(120, 50, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=None))

    model.add(keras.layers.Convolution1D(120, 25, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=None))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(3000))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2000))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1000))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(number_of_output_labels))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    model.summary()

    return model


def build_model_resnet_i(
    number_of_input_values=8501,
    number_of_output_labels=230,
    lr=0.0003,
    batchnorm_momentum=0.99,
    i=10,
    disable_batchnorm=False,
    use_group_norm=True,
    add_additional_dense_layer=True,
):
    """Build resnet model.

    Args:
        number_of_input_values (int, optional): Number of input values. Defaults to 8501.
        number_of_output_labels (int, optional): Number of output labels. Defaults to 230.
        lr (float, optional): learning rate of the Adam optimizer. Defaults to 0.001.
        batchnorm_momentum (float, optional): Momentum used for the batch normalization. Defaults to 0.99.
        i (int, optional): Index specifying the size of the resnet.
        Possible values: 10, 18, 34, 50, 101, 152, 200, 270, 350, 420, "custom_10". Defaults to 10.
        disable_batchnorm (bool, optional): Whether or not to disable the batch normalization. Defaults to False.
        use_group_norm (bool, optional): Whether or not to replace batch normalization with group normalization. Defaults to True.
        add_additional_dense_layer (bool, optional): Whether or not to add an additional dense layer in the end. Defaults to True.

    Returns:
        model: keras model
    """

    resnet_model = ResNet(
        i,
        keras.layers.InputSpec(shape=[None, number_of_input_values, 1]),
        square_kernel_size_and_stride=True,
        disable_batchnorm=disable_batchnorm,
        norm_momentum=batchnorm_momentum,
        use_group_norm=use_group_norm,
    )

    # predictions = keras.layers.AveragePooling1D(pool_size=5, strides=5)(
    #    resnet_model.layers[-1].output
    # )
    # predictions = keras.layers.Flatten()(predictions)

    predictions = keras.layers.Flatten()(resnet_model.layers[-1].output)

    if not add_additional_dense_layer:
        predictions = keras.layers.Dense(number_of_output_labels)(predictions)
    else:
        predictions = keras.layers.Dense(256)(predictions)
        predictions = keras.layers.Dense(number_of_output_labels)(predictions)

    model = keras.Model(resnet_model.inputs, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    model.summary()

    return model


if __name__ == "__main__":

    model = build_model_resnet_i(
        8501, 145, 0.001, i=10, use_group_norm=True, add_additional_dense_layer=True
    )
    model = build_model_resnet_i(
        8501, 145, 0.001, i=50, use_group_norm=True, add_additional_dense_layer=True
    )
    model = build_model_resnet_i(
        8501, 145, 0.001, i=101, use_group_norm=True, add_additional_dense_layer=True
    )
