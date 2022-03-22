import tensorflow.keras as keras
import tensorflow.keras.metrics as tfm
import tensorflow as tf


class BinaryAccuracy(tfm.BinaryAccuracy):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(BinaryAccuracy, self).update_state(
                y_true, tf.nn.sigmoid(y_pred), sample_weight
            )
        else:
            super(BinaryAccuracy, self).update_state(y_true, y_pred, sample_weight)


def build_model_park(
    hp=None, number_of_input_values=9018, number_of_output_labels=2, use_dropout=False
):

    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

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

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ],  # here from_logits is not needed, since argmax will be the same
    )

    return model


def build_model_park_medium_size(
    hp=None, number_of_input_values=9018, number_of_output_labels=2, use_dropout=False
):

    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

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

    model.add(
        # keras.layers.Dense(
        #    1 if (number_of_output_labels == 2) else number_of_output_labels
        # )
        keras.layers.Dense(number_of_output_labels)
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # if number_of_output_labels == 2:
    #    model.compile(
    #        optimizer=optimizer,
    #        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    #        metrics=[BinaryAccuracy(from_logits=True)],
    #    )
    # else:
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ],  # here from_logits is not needed, since argmax will be the same
    )

    return model


def build_model_park_huge_size(
    hp=None, number_of_input_values=9018, number_of_output_labels=2, use_dropout=False
):
    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            120,
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

    model.add(keras.layers.Convolution1D(120, 50, strides=5, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))

    model.add(keras.layers.Convolution1D(120, 25, strides=2, padding="same"))
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

    model.add(
        # keras.layers.Dense(
        #    1 if (number_of_output_labels == 2) else number_of_output_labels
        # )
        keras.layers.Dense(number_of_output_labels)
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # if number_of_output_labels == 2:
    #    model.compile(
    #        optimizer=optimizer,
    #        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    #        metrics=[BinaryAccuracy(from_logits=True)],
    #    )
    # else:
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ],  # here from_logits is not needed, since argmax will be the same
    )

    return model


def build_model_park_tiny_size(
    hp=None, number_of_input_values=9018, number_of_output_labels=2, use_dropout=False
):

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            120,
            100,
            strides=5,
            padding="same",
            input_shape=(number_of_input_values, 1),
        )
    )  # add convolution layer
    model.add(keras.layers.Activation("relu"))  # activation

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(500))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(50))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(number_of_output_labels))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ],  # here from_logits is not needed, since argmax will be the same
    )

    return model
