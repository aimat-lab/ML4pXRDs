import tensorflow.keras as keras
import tensorflow.keras.metrics as tfm
import tensorflow as tf
from train_dataset.utils.resnet_v2_1D import ResNetv2
from train_dataset.utils.resnet_keras_1D import ResNet
from train_dataset.utils.transformer_vit import build_model_transformer_vit
from train_dataset.utils.AdamWarmup import AdamWarmup
import numpy as np


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
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
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

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_2_layer_CNN(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
):

    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

    model = keras.models.Sequential()
    model.add(
        keras.layers.Convolution1D(
            30,
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

    model.add(keras.layers.Convolution1D(30, 50, strides=5, padding="same"))
    model.add(keras.layers.Activation("relu"))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=7, strides=7))

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
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_medium_size(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
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

    optimizer = keras.optimizers.Adam(learning_rate=lr)

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
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_huge_size(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
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

    optimizer = keras.optimizers.Adam(learning_rate=lr)

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
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_gigantic_size(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
):
    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

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

    optimizer = keras.optimizers.Adam(learning_rate=lr)

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
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_gigantic_size_more_dense(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
    momentum=0.0,
    optimizer="Adam",
):
    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

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

    model.add(
        # keras.layers.Dense(
        #    1 if (number_of_output_labels == 2) else number_of_output_labels
        # )
        keras.layers.Dense(number_of_output_labels)
    )

    if optimizer == "Adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    elif optimizer == "SGD":
        model.compile(
            optimizer=keras.optimizers.SGD(lr, momentum=momentum),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    else:
        raise Exception("Optimizer not supported.")

    model.summary()

    return model


def build_model_park_gigantic_size_more_dense_bn(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
    momentum=0.0,
    optimizer="Adam",
    bn_momentum=0.99,
):
    # From Park:
    # They actually train for 5000 epochs and batch size 1000 in the original paper

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

    model.add(keras.layers.BatchNormalization(momentum=bn_momentum))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=2))  # pooling layer

    model.add(keras.layers.Convolution1D(120, 75, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.BatchNormalization(momentum=bn_momentum))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=None))

    model.add(keras.layers.Convolution1D(120, 50, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.BatchNormalization(momentum=bn_momentum))

    if use_dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=None))

    model.add(keras.layers.Convolution1D(120, 25, strides=2, padding="same"))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.BatchNormalization(momentum=bn_momentum))

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

    model.add(
        # keras.layers.Dense(
        #    1 if (number_of_output_labels == 2) else number_of_output_labels
        # )
        keras.layers.Dense(number_of_output_labels)
    )

    if optimizer == "Adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    elif optimizer == "SGD":
        model.compile(
            optimizer=keras.optimizers.SGD(lr, momentum=momentum),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    else:
        raise Exception("Optimizer not supported.")

    model.summary()

    return model


def build_model_park_original_spg(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
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
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_park_tiny_size(
    hp=None,
    number_of_input_values=9018,
    number_of_output_labels=2,
    use_dropout=False,
    lr=0.001,
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

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],  # here from_logits is not needed, since argmax will be the same
    )

    model.summary()

    return model


def build_model_resnet_50_old(
    hp=None,
    number_of_input_values=8501,
    number_of_output_labels=2,
    dropout_rate=False,
    lr=0.0003,
):

    model_width = 16  # Width of the Initial Layer, subsequent layers start from here

    Model = ResNetv2(
        number_of_input_values,
        1,
        model_width,
        problem_type="Regression",  # this just yields a linear last layer (no activation) => from_logits can be used
        output_nums=number_of_output_labels,
        pooling="avg",
        dropout_rate=dropout_rate,
    ).ResNet50()

    Model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    Model.summary()

    return Model


def build_model_resnet_i(
    hp=None,
    number_of_input_values=8501,
    number_of_output_labels=2,
    lr=0.0003,
    momentum=0.0,  # only used for SGD
    optimizer="Adam",
    batchnorm_momentum=0.99,
    i=10,
    disable_batchnorm=False,
):

    # resnet_model = ResNet(
    #    10, keras.layers.InputSpec(shape=(None, number_of_input_values, 1))
    # )

    resnet_model = ResNet(
        i,
        keras.layers.InputSpec(shape=[None, number_of_input_values, 1]),
        square_kernel_size_and_stride=True,
        disable_batchnorm=disable_batchnorm,
        norm_momentum=batchnorm_momentum,
    )

    # predictions = keras.layers.AveragePooling1D(pool_size=5, strides=5)(
    #    resnet_model.layers[-1].output
    # )
    # predictions = keras.layers.Flatten()(predictions)

    predictions = keras.layers.Flatten()(resnet_model.layers[-1].output)
    predictions = keras.layers.Dense(number_of_output_labels)(predictions)

    model = keras.Model(resnet_model.inputs, outputs=predictions)

    # keras.utils.plot_model(model, show_shapes=True)

    if optimizer == "Adam":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    elif optimizer == "SGD":
        model.compile(
            optimizer=keras.optimizers.SGD(lr, momentum=momentum),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            ],
        )
    else:
        raise Exception("Optimizer not supported.")

    model.summary()

    return model


if __name__ == "__main__":

    # print("Gigantic size_more_dense")
    # model = build_model_park_gigantic_size_more_dense_bn(
    #    None, 8501, 145, False, 0.0001
    # )  # huge + one CNN layer + different strides
    # model.save("test")
    # model = keras.models.load_model("test")

    # print("Resnet 10")
    # model = build_model_resnet_i(None, 8501, 145, 0.0001, 0, "Adam", i=10)

    print("Resnet 50")
    model = build_model_resnet_i(None, 8501, 145, 0.0001, 0, "Adam", i=50)

    for i, layer in enumerate(model.layers):
        print(layer._name)
        if "batch_normalization" in layer._name:
            layer.momentum = 0.5

        if hasattr(layer, "sub_layers"):
            for sub_layer in layer.sub_layers:
                print("                   ", sub_layer._name)
                if "batch_normalization" in sub_layer._name:
                    layer.momentum = 0.5

    exit()

    if True:

        print("Tiny size")
        model = build_model_park_tiny_size(
            None, 8501, 145, False, 0.0001
        )  # only one conv layer but with more filters (120 instead of 80)
        # model.save("test")
        # model = keras.models.load_model(
        #    "test",
        #    custom_objects={
        #        "AdamWarmup": AdamWarmup
        #    },  # this works, even though it doesn't use it
        # )

        print("7-label version")
        model = build_model_park(None, 8501, 145, False, 0.0001)  # 7-label version
        # model.save("test")
        # model = keras.models.load_model("test")

        print("7-label version with 2 CNN layers")
        model = build_model_park_2_layer_CNN(
            None, 8501, 145, False, 0.0001
        )  # 7-label version with only 2 CNN layers and some less filters
        # model.save("test")
        # model = keras.models.load_model("test")

        print("Medium size")
        model = build_model_park_medium_size(
            None, 8501, 145, False, 0.0001
        )  # 101-label version
        # model.save("test")
        # model = keras.models.load_model("test")

        print("Original 230-label")
        model = build_model_park_original_spg(
            None, 8501, 145, False, 0.0001
        )  # 230-label version
        # model.save("test")
        # model = keras.models.load_model("test")

        print("Huge size")
        model = build_model_park_huge_size(
            None, 8501, 145, False, 0.0001
        )  # my version: original 230-label + more filters
        # model.save("test")
        # model = keras.models.load_model("test")

        print("Gigantic size")
        model = build_model_park_gigantic_size(
            None, 8501, 145, False, 0.0001
        )  # huge + one CNN layer + different strides
        # model.save("test")
        # model = keras.models.load_model("test")

        print("Gigantic size_more_dense")
        model = build_model_park_gigantic_size_more_dense(
            None, 8501, 145, False, 0.0001
        )  # huge + one CNN layer + different strides
        # model.save("test")
        # model = keras.models.load_model("test")

    print("Resnet 10")
    model = build_model_resnet_i(None, 8501, 145, 0.0001, 0, "Adam", i=10)

    print("Resnet 50")
    model = build_model_resnet_i(None, 8501, 145, 0.0001, 0, "Adam", i=50)
    # model.save("test")
    # model = keras.models.load_model("test")

    if False:
        print("ViT")
        model = build_model_transformer_vit(None, 8501, 145, 0.0001, 600, 1500)
        # model.save("test")
        # model = keras.models.load_model("test", custom_objects={"AdamWarmup": AdamWarmup})
