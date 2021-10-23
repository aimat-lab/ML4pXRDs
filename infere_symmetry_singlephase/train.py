import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
import datetime
import os
from keras_tuner import BayesianOptimization
from keras_tuner import Hyperband
from glob import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib
import tensorflow.keras as keras

# print available devices:
print(device_lib.list_local_devices())

csv_filenames = glob(r"databases/icsd/*.csv")

number_of_values_initial = 9001
simulated_range = np.linspace(0, 90, number_of_values_initial)
step = 2  # only use every step'th point in pattern
starting_angle = 5  # where to start using the simulated pattern
used_range_beginning = np.where(simulated_range == starting_angle)[0][0]
used_range = simulated_range[used_range_beginning::step]
number_of_values = len(used_range)

n_classes = 14  # TODO: read this directly from the csv

model_str = "Lee"  # possible: conv, fully_connected, Lee (CNN-3)
tune_hyperparameters = False
tuner_str = "bayesian"  # possible: hyperband and bayesian

read_from_csv = False
pickle_database = False  # for future, faster loading

pickle_path = r"databases/icsd/database"

x = None
bravais_str = None
space_group_number = None

if read_from_csv:

    for i, filename in enumerate(csv_filenames):

        print()
        print("Loading csv file {} of {}".format(i + 1, len(csv_filenames)))

        data = pd.read_csv(filename, delimiter=" ", header=None)

        x_more = np.array(data[range(1, number_of_values_initial + 1)])[
            :, used_range_beginning::step
        ]
        if x is None:
            x = x_more
        else:
            x = np.append(x, x_more, axis=0)

        bravais_str_more = np.array(data[number_of_values_initial + 1], dtype=str)
        if bravais_str is None:
            bravais_str = bravais_str_more
        else:
            bravais_str = np.append(bravais_str, bravais_str_more, axis=0)

        space_group_number_more = np.array(
            data[number_of_values_initial + 2], dtype=int
        )
        if space_group_number is None:
            space_group_number = space_group_number_more
        else:
            space_group_number = np.append(
                space_group_number, space_group_number_more, axis=0
            )

else:
    x, bravais_str, space_group_number = pickle.load(open(pickle_path, "rb"))

if pickle_database:
    database = (x, bravais_str, space_group_number)
    pickle.dump(database, open(pickle_path, "wb"))


space_group_number = space_group_number - 1  # make integers zero-based

bravais_labels = [
    "aP",
    "mP",
    "mS",
    "oP",
    "oS",
    "oI",
    "oF",
    "tP",
    "tI",
    "cP",
    "cI",
    "cF",
    "hP",
    "hR",
]
y = np.array([int(bravais_labels.index(name)) for name in bravais_str])

# Plot distribution over bravais lattices:
# plt.bar(*np.unique(y, return_counts=True))
# plt.show()
# Very inbalanced!

# Distribution of space groups:
# plt.bar(*np.unique(space_group_number, return_counts=True))
# plt.show()
# also very very inbalanced!

x = x / 100  # set maximum volume under a peak to 1

"""
# plot as test
xs = np.linspace(0, 90, 9001)[::10]
ys = x[1000]

plt.plot(xs, ys)
plt.show()

exit()
"""

assert not np.any(np.isnan(x))
assert not np.any(np.isnan(y))

print("##### Loaded {} training points".format(len(x)))

# when using conv2d layers, keras needs this format: (n_samples, height, width, channels)

if model_str == "conv":
    x = np.expand_dims(x, axis=2)

# Split into train, validation, test set + shuffle

x, y, bravais_str, space_group_number = shuffle(
    x, y, bravais_str, space_group_number, random_state=1234
)

# Use less training data for hyperparameter optimization
if tune_hyperparameters:
    x = x[0:100000]  # only use 100k patterns for hyper optimization
    y = y[0:100000]
    bravais_str = bravais_str[0:100000]
    space_group_number = space_group_number[0:100000]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

if model_str == "conv":

    def build_model(hp):  # define model with hyperparameters

        model = tf.keras.models.Sequential()

        starting_filter_size = hp.Int(
            "starting_filter_size", min_value=10, max_value=510, step=20
        )

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=4)):

            if i == 0:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=200, step=10
                        ),
                        int(starting_filter_size * (3 / 4) ** i),
                        input_shape=(number_of_values, 1),
                        activation="relu",
                    )
                )
            else:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=200, step=10
                        ),
                        int(starting_filter_size * (3 / 4) ** i),
                        activation="relu",
                    )
                )

            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Flatten())

        # flattended_size = model.layers[-1].get_output_at(0).get_shape().as_list()[1]
        # reduce_factor = hp.Int("reduce_factor", min_value=2, max_value=4)

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=4)):

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

        model.add(tf.keras.layers.Dense(14))

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

        model.add(tf.keras.layers.Dense(14))

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

        drop1 = keras.layers.Dropout(keep_prob_)(flat)

        # TODO: Why do they not use activation functions here?

        dense1 = keras.layers.Dense(
            2500, kernel_initializer=keras.initializers.GlorotNormal(seed=None)
        )(drop1)

        drop2 = keras.layers.Dropout(keep_prob_)(dense1)

        dense2 = keras.layers.Dense(
            1000, kernel_initializer=keras.initializers.GlorotNormal(seed=None)
        )(drop2)

        drop3 = keras.layers.Dropout(keep_prob_)(dense2)

        dense3 = keras.layers.Dense(
            n_classes, kernel_initializer=keras.initializers.GlorotNormal(seed=None)
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
            kwargs["batch_size"] = 100
            kwargs["epochs"] = 4
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
        project_name="bayesian_opt_" + model_str,
        directory="tuner",
        num_initial_points=3 * 9,
    )

elif tuner_str == "hyperband":

    class MyTuner(Hyperband):
        def run_trial(self, trial, *args, **kwargs):
            kwargs["batch_size"] = 500
            kwargs["epochs"] = 10
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    tuner = MyTuner(
        build_model,
        objective="val_accuracy",
        max_epochs=10,
        overwrite=False,
        directory="tuner",
        project_name="hyperband_opt_" + model_str,
        hyperband_iterations=100000,
    )

if tune_hyperparameters:

    tuner.search_space_summary()
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), verbose=2)

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
        epochs=20,
        batch_size=50,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, cp_callback],
    )

    print("\nOn test dataset:")
    model.evaluate(x_test, y_test, verbose=2)
    print()

    model.save(out_dir + "/model")
