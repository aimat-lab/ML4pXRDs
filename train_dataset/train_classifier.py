from re import S
import sys

sys.path.append("../")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
import datetime
import os
from keras_tuner import BayesianOptimization
from keras_tuner import Hyperband
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from dataset_simulations.narrow_simulation import NarrowSimulation
from sklearn.utils import class_weight

additional_tag = (
    ""  # additional tag that will be added to the tuner folder and training folder
)
current_data_source = "narrow"


number_of_values_initial = 9018
simulated_range = np.linspace(0, 90, number_of_values_initial)

# only use a restricted range of the simulated patterns
start_x = 10
end_x = 50
step = 1
start_index = np.argwhere(simulated_range >= start_x)[0][0]
end_index = np.argwhere(simulated_range <= end_x)[-1][0]
used_range = simulated_range[start_index : end_index + 1 : step]
number_of_values = len(used_range)

scale_features = True

model_str = "conv_narrow"  # possible: conv, fully_connected, Lee (CNN-3), conv_narrow
tuner_str = "bayesian"  # possible: hyperband and bayesian
tune_hyperparameters = True

tuner_epochs = 4
tuner_batch_size = 128

# read data
if current_data_source == "narrow":

    path_to_patterns = "../dataset_simulations/patterns/narrow/"

    sim = NarrowSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        output_dir="../dataset_simulations/patterns/narrow/",
    )

    sim.load()

    n_patterns_per_crystal = len(sim.sim_patterns[0])

    patterns = sim.sim_patterns
    labels = sim.sim_labels

    for i in reversed(range(0, len(patterns))):
        if any(x is None for x in patterns[i]):
            del patterns[i]
            del labels[i]

    patterns = np.array(patterns)
    print(patterns.shape)

    y = []
    for label in labels:
        y.extend([label[0]] * n_patterns_per_crystal)

    x = patterns.reshape((patterns.shape[0] * patterns.shape[1], patterns.shape[2]))
    x = x[:, start_index : end_index + 1 : step]
    y = np.array(y)

    class_weights = {}
    classes = np.unique(y)
    class_weight_array = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    )
    for i, weight in enumerate(class_weight_array):
        class_weights[classes[i]] = weight

    print("Class weights:")
    print(class_weight)

    n_classes = len(np.unique(y))

else:
    raise Exception("Data source not recognized.")

# print available devices:
print(device_lib.list_local_devices())

assert not np.any(np.isnan(x))
assert not np.any(np.isnan(y))
assert len(x) == len(y)

print("##### Loaded {} training points".format(len(x)))

# Split into train, validation, test set + shuffle
x, y = shuffle(x, y, random_state=1234)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

# scale features
if scale_features:
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_val = sc.transform(x_val)

# when using conv2d layers, keras needs this format: (n_samples, height, width, channels)
if "conv" in model_str:
    x = np.expand_dims(x, axis=2)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    x_val = np.expand_dims(x_val, axis=2)

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
                            "number_of_filters", min_value=10, max_value=210, step=20
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
                            "number_of_filters", min_value=10, max_value=200, step=20
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
            metrics=["accuracy"],
        )

        model.summary()

        return model


if model_str == "conv_narrow":

    def build_model(hp):  # define model with hyperparameters

        model = tf.keras.models.Sequential()

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=2)):

            if i == 0:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=210, step=20
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
                            "number_of_filters", min_value=10, max_value=200, step=20
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

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=2)):

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
            metrics=["accuracy"],
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
            metrics=["accuracy"],
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

        # TODO: Why do they originally not use activation functions here? Let's better use them.

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
            metrics=["accuracy"],
        )

        model.summary()

        return model


else:

    raise Exception("Model not recognized.")

if tuner_str == "bayesian":

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
        objective="val_accuracy",
        max_trials=1000,
        executions_per_trial=1,
        overwrite=False,
        project_name="bayesian_opt_"
        + model_str
        + (("_" + additional_tag) if additional_tag != "" else ""),
        directory="tuner",
        num_initial_points=3 * 9,
    )

elif tuner_str == "hyperband":

    class MyTuner(Hyperband):
        def run_trial(self, trial, *args, **kwargs):
            kwargs["batch_size"] = tuner_batch_size
            kwargs["epochs"] = tuner_epochs
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    tuner = MyTuner(
        build_model,
        objective="val_accuracy",
        max_epochs=tuner_epochs,
        overwrite=False,
        directory="tuner",
        project_name="hyperband_opt_"
        + model_str
        + (("_" + additional_tag) if additional_tag != "" else ""),
        hyperband_iterations=10000,
    )

if tune_hyperparameters:

    tuner.search_space_summary()
    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=[
            keras.callbacks.TensorBoard(
                "tuner/"
                + ("hyperband_opt_" if tuner_str == "hyperband" else "bayesian_opt_")
                + model_str
                + (("_" + additional_tag) if additional_tag != "" else "")
                + "/tf"
            )
        ],
        class_weight=class_weights,  # TODO: Is it actually using these class weights?
    )

else:  # build model from best set of hyperparameters

    training_outdir = "trainings/" + model_str + "/"

    if not model_str == "Lee":

        best_hp = tuner.get_best_hyperparameters()[0]

        config = best_hp.get_config()
        # config["values"]["dropout"] = 0.3 # modify the dropout rate

        changed_hp = best_hp.from_config(config)

        model = tuner.hypermodel.build(changed_hp)

        print("Model with best hyperparameters:")
        print(changed_hp.get_config())

        model.summary()

    else:

        model = build_model(None)

    # use tensorboard to inspect the graph, write log file periodically:
    out_dir = training_outdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = out_dir + "/log"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # periodically save the weights to a checkpoint file:
    checkpoint_path = out_dir + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True
    )

    model.fit(
        x_train,
        y_train,
        epochs=40,
        batch_size=100,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, cp_callback],
        verbose=2,
    )

    print("\nOn test dataset:")
    model.evaluate(x_test, y_test, verbose=2)
    print()

    model.save(out_dir + "/model")
