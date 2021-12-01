import sys

from tensorflow.python.keras.losses import BinaryCrossentropy

sys.path.append("../")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
from datetime import datetime
import os

sys.path.append("../dataset_simulations/")

from keras_tuner import BayesianOptimization
import keras_tuner
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from dataset_simulations.random_simulation import Simulation
from sklearn.utils import class_weight
import random
import math
import pickle
import gc
from sklearn.metrics import classification_report
from custom_loss import CustomSmoothedWeightedCCE

tag = (
    "test"  # additional tag that will be added to the tuner folder and training folder
)
mode = "narrow"  # possible: narrow and random
model_str = "conv_narrow"  # possible: conv, fully_connected, Lee (CNN-3), conv_narrow, Park, random
model_is_binary = False

train_on_this = None  # supply a model_str that contains a hyperparameter optimization to train on (using best parameters)

number_of_values_initial = 9018
simulated_range = np.linspace(0, 90, number_of_values_initial)

NO_workers = 1
queue_size = 20

if mode == "narrow":

    # only use a restricted range of the simulated patterns
    start_x = 10
    end_x = 50
    step = 1
    start_index = np.argwhere(simulated_range >= start_x)[0][0]
    end_index = np.argwhere(simulated_range <= end_x)[-1][0]
    used_range = simulated_range[start_index : end_index + 1 : step]
    number_of_values = len(used_range)

    scale_features = True

    tune_hyperparameters = True

    tuner_epochs = 4
    tuner_batch_size = 128

    current_dir = "narrow_19-11-2021_08:12:29_test"  # where to read the best model from
    train_epochs = 100
    train_batch_size = 128

elif mode == "random":

    # only use a restricted range of the simulated patterns

    # TODO: Maybe make this 0?
    start_x = 10
    end_x = 90  # different from above
    step = 1
    start_index = np.argwhere(simulated_range >= start_x)[0][0]
    end_index = np.argwhere(simulated_range <= end_x)[-1][0]
    used_range = simulated_range[start_index : end_index + 1 : step]
    number_of_values = len(used_range)

    scale_features = True

    tune_hyperparameters = False

    tuner_epochs = 4
    # tuner_batch_size = 500
    tuner_batch_size = 64

    current_dir = "narrow_19-11-2021_08:12:29_test"
    train_epochs = 1000
    # train_batch_size = 500
    train_batch_size = 500

if train_on_this is None:
    out_base = (
        "classifier/"
        + mode
        + "_"
        + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        + "_"
        + tag
        + "/"
    )
else:
    out_base = "classifier/" + train_on_this + "/"

if not os.path.exists(out_base):
    os.system("mkdir -p " + out_base)

# Each mode / model needs to define (x_train and y_train) or only x_train if it is a Sequence object (then leave y_train as None)
# Each mode / model needs to define x_val, y_val and x_test and y_test

x_train = None
y_train = None  # can stay None

x_val = None
y_val = None

x_test = None
y_test = None

n_classes = None

class_weights = None  # can stay None

# read and preprocess the data
if mode == "narrow":

    path_to_patterns = "../dataset_simulations/patterns/narrow/"

    sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
    sim.output_dir = "../dataset_simulations/patterns/narrow/"

    sim.load()

    n_patterns_per_crystal = len(sim.sim_patterns[0])

    patterns = sim.sim_patterns
    labels = sim.sim_labels
    variations = sim.sim_variations

    # plt.plot(patterns[124][3])
    # plt.show()

    for i in reversed(range(0, len(patterns))):
        if np.any(np.isnan(variations[i][0])):
            del patterns[i]
            del labels[i]
            del variations[i]

    patterns = np.array(patterns)

    y = []
    for label in labels:
        y.extend([label[0]] * n_patterns_per_crystal)

    x = patterns.reshape((patterns.shape[0] * patterns.shape[1], patterns.shape[2]))
    variations = np.array(variations)
    variations = variations.reshape(
        (variations.shape[0] * variations.shape[1], variations.shape[2])
    )
    x = x[:, start_index : end_index + 1 : step]
    y = np.array(y)

    print(f"Shape of x: {x.shape}")
    print(f"Shape of y: {y.shape}")

    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    assert len(x) == len(y)

    n_classes = len(np.unique(y))

    print("##### Loaded {} training points".format(len(x)))

    # Split into train, validation, test set + shuffle
    x, y, variations = shuffle(x, y, variations, random_state=1234)

    (
        __x_train,
        __x_test,
        __y_train,
        __y_test,
        variations_train,
        variations_test,
    ) = train_test_split(x, y, variations, test_size=0.3)
    (
        __x_test,
        __x_val,
        __y_test,
        __y_val,
        variations_test,
        variations_val,
    ) = train_test_split(__x_test, __y_test, variations_test, test_size=0.5)

    # compute proper class weights
    classes = np.unique(
        __y_train
    )  # only compute class weights based on the majority label
    class_weights_narrow = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=__y_train
    )

    print("Class weights:")
    print(class_weights_narrow)

    if scale_features:
        sc = StandardScaler()
        sc.fit(__x_train)

        with open(os.path.join(out_base, "scaler"), "wb") as file:
            pickle.dump(sc, file)

    # when using conv2d layers, keras needs this format: (n_samples, height, width, channels)
    if "conv" in model_str:
        __x_train = np.expand_dims(__x_train, axis=2)
        __x_test = np.expand_dims(__x_test, axis=2)
        __x_val = np.expand_dims(__x_val, axis=2)

    def calc_std_dev(two_theta, tau, wavelength=1.207930):
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            standard deviation for gaussian kernel
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9  ## shape factor
        wavelength = wavelength * 0.1  ## angstrom to nm
        theta = np.radians(two_theta / 2.0)  ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau)  # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
        return sigma

    def alter_dataset(current_x, current_y, variations, do_plot=False):

        min_peak_height = 0.01

        is_pures = []
        for i, item in enumerate(current_x):
            if do_plot:
                plt.plot(current_x[i, :, 0], label="Original")

            if random.random() < 0.5:  # in 50% of the samples, add additional peaks
                is_pures.append(0)

                for j in range(0, random.randint(1, 5)):
                    mean = random.uniform(start_x, end_x)

                    sigma_peak = calc_std_dev(mean, variations[i][0])

                    peak_height = random.uniform(min_peak_height, 1)
                    peak = (
                        1
                        / (sigma_peak * np.sqrt(2 * np.pi))
                        * np.exp(-1 / (2 * sigma_peak ** 2) * (used_range - mean) ** 2)
                    )

                    peak = peak / np.max(peak) * peak_height

                    current_x[i, :, 0] += peak
            else:
                is_pures.append(1)

            if do_plot:
                plt.plot(current_x[i, :, 0], label="Altered")
                plt.legend()
                plt.show()

        if scale_features:
            current_x[:, :, 0] = sc.transform(current_x[:, :, 0])

        return current_x, [current_y, np.array(is_pures)]

    class NarrowSequence(keras.utils.Sequence):
        def __init__(self, x_train, y_train, variations, batch_size):

            self.x_train = x_train
            self.y_train = y_train

            self.variations = variations
            self.batch_size = batch_size
            self.number_of_batches = math.ceil(len(x_train) / batch_size)

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):

            end_index = (idx + 1) * self.batch_size

            current_x = self.x_train[
                idx * self.batch_size : end_index
                if end_index < len(self.x_train)
                else len(self.x_train)
            ]
            current_y = self.y_train[
                idx * self.batch_size : end_index
                if end_index < len(self.x_train)
                else len(self.x_train)
            ]
            variations = self.variations[
                idx * self.batch_size : end_index
                if end_index < len(self.x_train)
                else len(self.x_train)
            ]

            return alter_dataset(current_x, current_y, variations)

    # encode train and val y as one_hot to work with the custom loss function
    # for y_test this is not needed, since I test it by hand anyways and the custom loss is not used
    __y_train = np.array(tf.one_hot(__y_train, 3))
    __y_val = np.array(tf.one_hot(__y_val, 3))

    x_train = NarrowSequence(
        __x_train,
        __y_train,
        variations,
        tuner_batch_size if tune_hyperparameters else train_batch_size,
    )

    x_test, y_test = alter_dataset(__x_test, __y_test, variations_test)
    x_val, y_val = alter_dataset(__x_val, __y_val, variations_val)

elif mode == "random":

    path_to_patterns = "../dataset_simulations/patterns/random/"

    cluster = os.getenv("CURRENT_CLUSTER")
    if cluster is None:
        raise Exception("Environment variable CURRENT_CLUSTER not set.")

    if cluster == "bwuni":
        sim = Simulation(
            "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/kit/iti/la2559/Databases/ICSD/cif/",
        )
        sim.output_dir = path_to_patterns

    elif cluster == "local":
        sim = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        sim.output_dir = path_to_patterns

    elif cluster == "nano":
        sim = Simulation(
            "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/ws/uvgnh/Databases/ICSD/cif/",
        )
        sim.output_dir = path_to_patterns

    sim.load()

    n_patterns_per_crystal = len(sim.sim_patterns[0])

    patterns = sim.sim_patterns
    labels = sim.sim_labels
    variations = sim.sim_variations

    for i in reversed(range(0, len(patterns))):
        if np.any(np.isnan(variations[i][0])):
            del patterns[i]
            del labels[i]
            del variations[i]

    ys_unique = [14, 104]
    y = []
    for label in labels:
        y.extend([ys_unique.index(label[0])] * n_patterns_per_crystal)

    y = np.array(y)

    # print(np.sum(y == 0))
    # print(np.sum(y == 1))
    # plt.hist(y)
    # plt.show()

    x_1 = []
    for pattern in patterns:
        for sub_pattern in pattern:
            x_1.append(sub_pattern[start_index : end_index + 1 : step])

    variations = np.array(variations)
    variations = variations.reshape(
        (variations.shape[0] * variations.shape[1], variations.shape[2])
    )

    # print(f"Shape of x: {x_1.shape}")
    # print(f"Shape of y: {y.shape}")

    # assert not np.any(np.isnan(x_1))
    # assert not np.any(np.isnan(y))
    # assert len(x_1) == len(y)

    n_classes = len(np.unique(y))

    print("##### Loaded {} training points with {} classes".format(len(x_1), n_classes))

    # Split into train, validation, test set + shuffle
    x_2, y, variations = shuffle(x_1, y, variations, random_state=1234)

    x_train_3, x_test_3, y_train, y_test = train_test_split(x_2, y, test_size=0.3)
    x_test_4, x_val_4, y_test, y_val = train_test_split(x_test_3, y_test, test_size=0.5)

    if scale_features:
        sc = StandardScaler()

        x_train_transformed = sc.fit_transform(x_train_3)
        x_test_transformed = sc.transform(x_test_4)
        x_val_transformed = sc.transform(x_val_4)

        del x_train_3[:]
        del x_train_3
        del x_test_4[:]
        del x_test_4
        del x_val_4[:]
        del x_val_4
        gc.collect()

        with open(os.path.join(out_base, "scaler"), "wb") as file:
            pickle.dump(sc, file)

    if model_str != "fully_connected":
        x_train = np.expand_dims(x_train_transformed, axis=2)
        x_test = np.expand_dims(x_test_transformed, axis=2)
        x_val = np.expand_dims(x_val_transformed, axis=2)

        del x_train_transformed
        del x_test_transformed
        del x_val_transformed
        gc.collect()

    else:

        x_train = x_train_transformed
        x_test = x_test_transformed
        x_val = x_val_transformed

else:
    raise Exception("Data source not recognized.")

# print available devices:
# print(device_lib.list_local_devices())

if model_str == "conv":

    def build_model(hp):  # define model with hyperparameters

        model = tf.keras.models.Sequential()

        # starting_filter_size = hp.Int(
        #    "starting_filter_size", min_value=10, max_value=510, step=20
        # )

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=3)):

            if i == 0:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=210, step=20,
                        ),
                        # int(starting_filter_size * (3 / 4) ** i),
                        hp.Int(
                            "filter_size_" + str(i),
                            min_value=10,
                            max_value=510,
                            step=20,
                        ),
                        input_shape=(number_of_values, 1),
                        activation="relu",
                    )
                )
            else:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=200, step=20,
                        ),
                        # int(starting_filter_size * (3 / 4) ** i),
                        hp.Int(
                            "filter_size_" + str(i),
                            min_value=10,
                            max_value=510,
                            step=20,
                        ),
                        activation="relu",
                    )
                )

            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Flatten())

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=3)):

            model.add(
                tf.keras.layers.Dense(
                    hp.Int(
                        "number_of_dense_units_" + str(i),
                        min_value=32,
                        max_value=2080,
                        step=128,
                    ),
                    activation="relu",
                )
            )
            model.add(
                tf.keras.layers.Dropout(
                    hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)
                )
            )

        model.add(tf.keras.layers.Dense(n_classes))

        optimizer = tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["SparseCategoricalAccuracy"],
        )

        model.summary()

        return model


elif model_str == "conv_narrow":

    def build_model(hp):  # define model with hyperparameters

        inputs = keras.layers.Input(shape=(number_of_values, 1))
        x_nn = inputs

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=2)):

            x_nn = keras.layers.Conv1D(
                hp.Int("number_of_filters", min_value=10, max_value=210, step=20),
                # int(starting_filter_size * (3 / 4) ** i),
                hp.Int("filter_size_" + str(i), min_value=10, max_value=510, step=20,),
                input_shape=(number_of_values, 1),
                activation="relu",
            )(x_nn)

            x_nn = keras.layers.BatchNormalization()(x_nn)
            x_nn = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x_nn)

        x_nn = keras.layers.Flatten()(x_nn)

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=2)):

            x_nn = keras.layers.Dense(
                hp.Int(
                    "number_of_dense_units_" + str(i),
                    min_value=32,
                    max_value=2080,
                    step=128,
                ),
                activation="relu",
            )(x_nn)
            x_nn = keras.layers.Dropout(
                hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)
            )(x_nn)

        outputs_softmax = keras.layers.Dense(n_classes, name="outputs_softmax")(x_nn)
        output_sigmoid = keras.layers.Dense(1, name="output_sigmoid")(x_nn)

        optimizer = keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
        )

        model = keras.Model(inputs=inputs, outputs=[outputs_softmax, output_sigmoid])

        model.compile(
            optimizer=optimizer,
            loss={
                # "outputs_softmax": CustomSmoothedWeightedCCE(
                #    class_weights=class_weights_narrow
                # ),
                "output_sigmoid": BinaryCrossentropy(from_logits=True),
            },
            metrics={
                "outputs_softmax": "CategoricalAccuracy",
                "output_sigmoid": "BinaryAccuracy",
            },
        )

        model.summary()

        return model


elif model_str == "fully_connected":

    def build_model(hp):

        model = tf.keras.models.Sequential()

        for i in range(0, hp.Int("number_of_layers", min_value=1, max_value=15)):

            if i == 0:
                model.add(
                    tf.keras.layers.Dense(
                        hp.Int(
                            "units_" + str(i), min_value=64, max_value=2048, step=64
                        ),
                        activation="relu",
                        kernel_regularizer=regularizers.l2(
                            hp.Float("l2_reg", 0, 0.005, step=0.0001)
                        ),
                        input_shape=(number_of_values,),
                    )
                )

            else:

                model.add(
                    tf.keras.layers.Dense(
                        hp.Int(
                            "units_" + str(i), min_value=64, max_value=2048, step=64
                        ),
                        activation="relu",
                        kernel_regularizer=regularizers.l2(
                            hp.Float("l2_reg", 0, 0.005, step=0.0001)
                        ),
                    )
                )

            model.add(
                tf.keras.layers.Dropout(
                    hp.Float("dropout", 0, 0.5, step=0.1, default=0.2)
                )
            )

        model.add(tf.keras.layers.Dense(n_classes))

        optimizer_str = hp.Choice("optimizer", values=["adam", "adagrad", "SGD"])

        if optimizer_str == "adam":
            optimizer = tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["SparseCategoricalAccuracy"],
        )

        return model


elif model_str == "Lee":

    def build_model(hp=None):

        keep_prob_ = 0.5
        learning_rate_ = 0.001

        inputs = keras.Input(
            shape=(number_of_values, 1), dtype=tf.float32, name="inputs"
        )

        conv1 = keras.layers.Conv1D(
            filters=64,
            kernel_size=20,
            strides=1,
            padding="same",
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(inputs)

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=3, padding="same")(
            conv1
        )

        conv2 = keras.layers.Conv1D(
            filters=64,
            kernel_size=15,
            strides=1,
            padding="same",
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(max_pool_1)

        max_pool_2 = keras.layers.MaxPool1D(pool_size=2, strides=3, padding="same")(
            conv2
        )

        """
        conv3 = keras.layers.Conv1D(
            filters=64,
            kernel_size=10,
            strides=2,
            padding="same",
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(max_pool_2)

        max_pool_3 = keras.layers.MaxPool1D(pool_size=1, strides=2, padding="same")(
            conv3
        )

        flat = keras.layers.Flatten()(max_pool_3)
        """

        flat = keras.layers.Flatten()(max_pool_2)

        drop1 = keras.layers.Dropout(keep_prob_)(flat)

        # TODO: Why do they originally not use activation functions here? Let's better use them:

        dense1 = keras.layers.Dense(
            2500,
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(drop1)

        drop2 = keras.layers.Dropout(keep_prob_)(dense1)

        dense2 = keras.layers.Dense(
            1000,
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(drop2)

        drop3 = keras.layers.Dropout(keep_prob_)(dense2)

        dense3 = keras.layers.Dense(
            n_classes,
            kernel_initializer=keras.initializers.GlorotNormal(seed=None),
            activation="relu",
        )(drop3)

        model = keras.Model(inputs=inputs, outputs=dense3)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["SparseCategoricalAccuracy"],
        )

        model.summary()

        return model


elif model_str == "Park":

    def build_model(hp=None):

        # this is the 7-label version

        model = keras.models.Sequential()
        model.add(
            keras.layers.Convolution1D(
                80,
                100,
                subsample_length=5,
                border_mode="same",
                input_shape=(number_of_values, 1),
            )
        )  # add convolution layer
        model.add(keras.layers.Activation("relu"))  # activation
        model.add(keras.layers.Dropout(0.3))
        model.add(
            keras.layers.AveragePooling1D(pool_length=3, stride=2)
        )  # pooling layer
        model.add(
            keras.layers.Convolution1D(80, 50, subsample_length=5, border_mode="same")
        )
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.AveragePooling1D(pool_length=3, stride=None))
        model.add(
            keras.layers.Convolution1D(80, 25, subsample_length=2, border_mode="same")
        )
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.AveragePooling1D(pool_length=3, stride=None))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(700))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(70))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(7))

        optimizer = keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["SparseCategoricalAccuracy"],
        )
        # they actually train for 5000 epochs and batch size 500

        return model


elif model_str == "random":

    def build_model(hp=None):

        """
        model = keras.models.Sequential()

        model.add(
            keras.layers.Dense(
                1024, activation="relu", input_shape=(number_of_values,),
            )
        )

        model.add(keras.layers.Dense(1024, activation="relu",))

        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Adam()

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["BinaryAccuracy"],
        )

        return model
        """

        # From Park:
        model = keras.models.Sequential()
        model.add(
            keras.layers.Convolution1D(
                80, 100, strides=5, padding="same", input_shape=(number_of_values, 1),
            )
        )  # add convolution layer
        model.add(keras.layers.Activation("relu"))  # activation
        model.add(keras.layers.Dropout(0.3))
        model.add(
            keras.layers.AveragePooling1D(pool_size=3, strides=2)
        )  # pooling layer
        model.add(keras.layers.Convolution1D(80, 50, strides=5, padding="same"))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))
        model.add(keras.layers.Convolution1D(80, 25, strides=2, padding="same"))
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.AveragePooling1D(pool_size=3, strides=None))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(700))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(70))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        optimizer = keras.optimizers.Adam()

        if model_is_binary:
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["BinaryAccuracy"],
            )
            # they actually train for 5000 epochs and batch size 500
        else:
            raise Exception("Non-binary classification not supported here.")

        return model


else:

    raise Exception("Model not recognized.")


class MyTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = tuner_batch_size
        kwargs["epochs"] = tuner_epochs
        super(MyTuner, self).run_trial(trial, *args, **kwargs)

    def on_epoch_end(trial, model, epoch, *args, **kwargs):

        checkpoint_files = glob(
            "tuner/bayesian_opt_"
            + model_str
            + "/trial_*/checkpoints/epoch_*/checkpoint*"
        )

        for file in checkpoint_files:
            os.system("rm " + file)


tuner = MyTuner(
    build_model,
    objective=keras_tuner.Objective(
        # "val_outputs_softmax_categorical_accuracy", direction="max"
        "val_output_sigmoid_binary_accuracy",
        direction="max",
    ),  # TODO: Maybe use a combination of the softmax and sigmoid metric in the future
    max_trials=1000,
    executions_per_trial=1,
    overwrite=False,
    project_name="tuner",
    directory=out_base if tune_hyperparameters else ("classifier/" + current_dir),
    num_initial_points=3 * 9,
)

if tune_hyperparameters:

    tuner.search_space_summary()
    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=[keras.callbacks.TensorBoard(out_base + "tuner_tb")],
        class_weight=class_weights,
        steps_per_epoch=x_train.number_of_batches if y_train is None else None,
        workers=NO_workers,
        max_queue_size=queue_size,
        use_multiprocessing=True,
    )

else:  # build model from best set of hyperparameters

    if not model_str == "Lee" and not model_str == "random":

        best_hp = tuner.get_best_hyperparameters()[0]

        config = best_hp.get_config()
        # config["values"]["dropout"] = 0.3 # modify the dropout rate

        changed_hp = best_hp.from_config(config)

        model = tuner.hypermodel.build(changed_hp)

        print("Model with best hyperparameters:")
        print(changed_hp.get_config())

        model.summary()

    else:  # models that do not have optimizable hyperparameters:

        model = build_model(None)

    # use tensorboard to inspect the graph, write log file periodically:
    log_dir = out_base + "tb"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # periodically save the weights to a checkpoint file:
    checkpoint_path = out_base + "cps" + "/weights{epoch}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True
    )

    model.fit(
        x_train,
        y_train,
        epochs=train_epochs,
        batch_size=train_batch_size,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, cp_callback],
        verbose=2,
        workers=NO_workers,
        max_queue_size=queue_size,
        use_multiprocessing=True,
        steps_per_epoch=x_train.number_of_batches if y_train is None else None,
        class_weight=class_weights,
    )

    # print("\nOn test dataset:")
    # model.evaluate(x_test, y_test, verbose=2)
    # print()

    model.save(out_base + "final")

    # model = keras.models.load_model(
    #    "classifier/narrow_30-11-2021_09:42:59_test/final", compile=False
    # )

    if model_is_binary:

        prob_model = keras.Sequential([model, keras.layers.Activation("sigmoid")])
        predicted_y = np.array(prob_model.predict(x_test))
        predicted_y = predicted_y[:, 0]
        predicted_y = np.where(predicted_y > 0.5, 1, 0)

        print()
        print("Classification report:")
        print(classification_report(y_test, predicted_y))

    elif model_str != "conv_narrow":

        prob_model = keras.Sequential([model, keras.layers.Activation("softmax")])
        predicted_y = np.array(prob_model.predict(x_test))
        predicted_y = np.argmax(predicted_y, axis=1)

        print()
        print("Classification report:")
        print(classification_report(y_test, predicted_y))

    else:

        softmax_activation = keras.layers.Activation("softmax")(
            model.get_layer("outputs_softmax").output
        )
        prob_model_softmax = keras.Model(
            inputs=model.layers[0].output, outputs=softmax_activation
        )
        prediction_softmax = prob_model_softmax.predict(x_test)
        prediction_softmax = np.argmax(prediction_softmax, axis=1)

        print()
        print("Classification report softmax:")
        print(classification_report(y_test[0], prediction_softmax))

        sigmoid_activation = keras.layers.Activation("sigmoid")(
            model.get_layer("output_sigmoid").output
        )
        prob_model_sigmoid = keras.Model(
            inputs=model.layers[0].output, outputs=sigmoid_activation
        )
        prediction_sigmoid = prob_model_sigmoid.predict(x_test)
        prediction_sigmoid = prediction_sigmoid[:, 0]
        prediction_sigmoid = np.where(prediction_sigmoid > 0.5, 1, 0)

        print()
        print("Classification report sigmoid:")
        print(classification_report(y_test[1], prediction_sigmoid))
