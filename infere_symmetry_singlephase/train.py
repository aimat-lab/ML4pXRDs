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

csv_filenames = glob(r"databases/icsd/*.csv")[:]

# number_of_values = 181
number_of_values_initial = 9001
skip_values = 10  # only use one tenth of the dataset
number_of_values = (number_of_values_initial - 1) // skip_values

# TODO: Use different range for training

model_str = "conv"  # possible: conv, fully_connected
tune_hyperparameters = True
tuner_str = "hyperband"  # possible: hyperband and bayesian

read_from_csv = False
pickle_database = False
pickle_path = "databases/icsd/database"

# Use less training data for hyperparameter optimization
# if tune_hyperparameters:
#    filenames = filenames[0:3]

x = None
bravais_str = None
space_group_number = None

if read_from_csv:

    for i, filename in enumerate(csv_filenames):

        print()
        print("Loading csv file {} of {}".format(i + 1, len(csv_filenames)))

        data = pd.read_csv(filename, delimiter=" ", header=None)

        # x_more = np.loadtxt(filename, delimiter=' ', usecols=list(range(1, number_of_values + 1))) # too slow
        x_more = np.array(data[range(1, number_of_values_initial + 1)])[
            :, ::skip_values
        ]

        if x is None:
            x = x_more
        else:
            x = np.append(x, x_more, axis=0)

        # bravais_str_more = np.loadtxt(filename, delimiter=' ', usecols=[number_of_values + 1], dtype=str) # too slow
        bravais_str_more = np.array(data[number_of_values_initial + 1], dtype=str)
        if bravais_str is None:
            bravais_str = bravais_str_more
        else:
            bravais_str = np.append(bravais_str, bravais_str_more, axis=0)

        # space_group_number_more = np.loadtxt(filename, delimiter=' ', usecols=[number_of_values + 2], dtype=int) # too slow
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
# TODO: Check how many land in which class

x = x / 100  # set maximum volume under a peak to 1

"""
# plot as test
xs = np.linspace(0, 90, 9001)[::10]
ys = x[1000]

plt.plot(xs, ys)
plt.show()

exit()
"""

# print(x.shape)

assert not np.any(np.isnan(x))
assert not np.any(np.isnan(y))

print("##### Loaded {} training points".format(len(x)))

# when using conv2d layers, keras needs this format: (n_samples, height, width, channels)

if model_str == "conv":
    x = np.expand_dims(x, axis=2)

# Split into train, validation, test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

if model_str == "conv":

    def build_model(hp):  # define model with hyperparameters

        model = tf.keras.models.Sequential()

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=4)):

            if i == 0:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=200, step=10
                        ),
                        hp.Choice("filter_size", [3, 5, 7, 9]),
                        input_shape=(number_of_values, 1),
                    )
                )
            else:
                model.add(
                    tf.keras.layers.Conv1D(
                        hp.Int(
                            "number_of_filters", min_value=10, max_value=200, step=10
                        ),
                        hp.Choice("filter_size", [3, 5, 7, 9]),
                    )
                )

            model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice("pool_size", [3,5,7,9]), strides=hp.Choice("pool_size", [3,5,7,9])))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Flatten())

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=10)):

            model.add(
                tf.keras.layers.Dense(
                    hp.Int(
                        "number_of_dense_units", min_value=32, max_value=512, step=32
                    ),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(
                        hp.Float("l2_reg", 0, 0.005, step=0.0001)
                    ),
                )
            )
            model.add(
                tf.keras.layers.Dropout(
                    hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)
                )
            )

        tf.keras.layers.Dense(14)

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


elif model_str == "fully_connected":

    def build_model(hp):

        model = tf.keras.models.Sequential()

        for i in range(0, hp.Int("number_of_layers", min_value=1, max_value=15)):

            if i == 0:
                tf.keras.layers.Dense(
                    hp.Int("units_" + str(i), min_value=64, max_value=2048, step=64),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(
                        hp.Float("l2_reg", 0, 0.005, step=0.0001)
                    ),
                    input_shape=(number_of_values,),
                )

            else:

                tf.keras.layers.Dense(
                    hp.Int("units_" + str(i), min_value=64, max_value=2048, step=64),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(
                        hp.Float("l2_reg", 0, 0.005, step=0.0001)
                    ),
                )

            tf.keras.layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.2))

        tf.keras.layers.Dense(14)

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


else:

    raise Exception("Model not recognized.")

# TODO: Try ray tune here
if tuner_str == "bayesian":

    class MyTuner(BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            kwargs["batch_size"] = 100
            kwargs["epochs"] = 10
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

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
    tuner.search(x_train, y_train, validation_data=(x_val, y_val))

else:  # build model from best set of hyperparameters

    training_outdir = "trainings/" + model_str + "/"

    best_hp = tuner.get_best_hyperparameters()[0]

    config = best_hp.get_config()
    # config["values"]["dropout"] = 0.3 # modify the dropout rate

    changed_hp = best_hp.from_config(config)

    model = tuner.hypermodel.build(changed_hp)

    print("Model with best hyperparameters:")
    print(changed_hp.get_config())

    model.summary()

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
        epochs=1000,
        batch_size=200,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback, cp_callback],
    )

    print("\nOn test dataset:")
    model.evaluate(x_test, y_test, verbose=2)
    print()

    model.save(out_dir + "/model")
