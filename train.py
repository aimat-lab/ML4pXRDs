from keras_tuner.engine.hyperparameters import HyperParameter
from keras_tuner.tuners import hyperband
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
import datetime
import os
from keras_tuner import BayesianOptimization
from keras_tuner import Hyperband

filenames = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv", "dataset_6.csv", "dataset_7.csv", "dataset_8.csv"]

model_str = "fully_connected" # possible: conv, lstm, fully_connected
tune_hyperparameters = True
tuner_str = "hyperband" # possible: hyperband and bayesian

if tune_hyperparameters:
    filenames = filenames[0:3]

x = None 
bravais_str = None
space_group_number = None

for filename in filenames:
    x_more = np.loadtxt(filename, delimiter=' ', usecols=list(range(1,182))) 
    if x is None:
        x = x_more
    else:
        x = np.append(x, x_more, axis=0)

    bravais_str_more = np.loadtxt(filename, delimiter=' ', usecols=[182], dtype=str)
    if bravais_str is None:
        bravais_str = bravais_str_more
    else:
        bravais_str = np.append(bravais_str, bravais_str_more, axis=0)

    space_group_number_more = np.loadtxt(filename, delimiter=' ', usecols=[183], dtype=int)
    if space_group_number is None:
        space_group_number = space_group_number_more
    else:
        space_group_number = np.append(space_group_number, space_group_number_more, axis=0)

space_group_number = space_group_number - 1 # make integers zero-based

bravais_labels = ["aP", "mP", "mS", "oP", "oS", "oI", "oF", "tP", "tI", "cP", "cI", "cF", "hP", "hR"]
y = np.array([int(bravais_labels.index(name)) for name in bravais_str])

x = x / 100 # set maximum volume under a peak to 1

assert not np.any(np.isnan(x))
assert not np.any(np.isnan(y))

print("##### Loaded {} training points".format(len(x)))

# when using conv2d layers, keras needs this format: (n_samples, height, width, channels)

if model_str == "lstm" or model_str == "conv":
    x = np.expand_dims(x, axis=2)

# Split into train, validation, test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

print(np.shape(x_train))

if model_str == "lstm":

    model = tf.keras.models.Sequential([

        tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(181, 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(14)
    ])

elif model_str == "conv":

    def build_model(hp):

        model = tf.keras.models.Sequential()

        for i in range(0, hp.Int("number_of_conv_layers", min_value=1, max_value=4)):
            
            if i == 0:
                model.add(tf.keras.layers.Conv1D(hp.Int("number_of_filters", min_value=10, max_value=200, step=10), 
                        hp.Choice("filter_size", [3,5,7,9]), input_shape=(181, 1)))
            else:
                model.add(tf.keras.layers.Conv1D(hp.Int("number_of_filters", min_value=10, max_value=200, step=10), 
                        hp.Choice("filter_size", [3,5,7,9])))

            model.add(tf.keras.layers.BatchNormalization())
            #model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice("pool_size", [3,5,7,9]), strides=hp.Choice("pool_size", [3,5,7,9])))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Flatten())

        for i in range(0, hp.Int("number_of_dense_layers", min_value=1, max_value=10)):

            model.add(tf.keras.layers.Dense(hp.Int("number_of_dense_units", min_value=32, max_value=512, step=32), 
                        activation='relu', kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 0, 0.005, step=0.0001))))
            model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)))

        tf.keras.layers.Dense(14)

        optimizer_str=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        if optimizer_str == 'adam':
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == 'adagrad':
            optimizer=tf.keras.optimizers.Adagrad(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == 'SGD':
            optimizer=tf.keras.optimizers.SGD(
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
                tf.keras.layers.Dense(hp.Int("units_" + str(i), min_value=64, max_value=2048, step=64), activation='relu', 
                                    kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 0, 0.005, step=0.0001)), input_shape=(181,))

            else:

                tf.keras.layers.Dense(hp.Int("units_" + str(i), min_value=64, max_value=2048, step=64), activation='relu', 
                                    kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 0, 0.005, step=0.0001)))

            tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.2))

        tf.keras.layers.Dense(14)

        optimizer_str=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        if optimizer_str == 'adam':
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == 'adagrad':
            optimizer=tf.keras.optimizers.Adagrad(
                hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
            )
        elif optimizer_str == 'SGD':
            optimizer=tf.keras.optimizers.SGD(
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

if tuner_str == "bayesian":

    class MyTuner(BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            kwargs['batch_size'] = 500
            kwargs['epochs'] = 10
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    tuner = MyTuner(
        build_model,
        objective="val_accuracy",
        max_trials=1000,
        executions_per_trial=1,
        overwrite=False,
        project_name="bayesian_opt_" + model_str,
        directory="tuner",
        num_initial_points=3*9,
    )

elif tuner_str == "hyperband":

    class MyTuner(Hyperband):
        def run_trial(self, trial, *args, **kwargs):
            kwargs['batch_size'] = 500
            kwargs['epochs'] = 10
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    tuner = MyTuner(
        build_model,
        objective="val_accuracy",
        max_epochs=10,
        overwrite=False,
        directory="tuner",
        project_name="hyperband_opt_" + model_str,
        hyperband_iterations=1000
    )

if tune_hyperparameters:

    tuner.search_space_summary()
    tuner.search(x_train, y_train, validation_data=(x_val, y_val))

else:

    training_outdir = "trainings/" + model_str + "/"

    best_hp = tuner.get_best_hyperparameters()[0]

    config = best_hp.get_config()
    config["values"]["dropout"] = 0.3

    changed_hp = best_hp.from_config(config)

    model = tuner.hypermodel.build(changed_hp)

    print("Model with best hyperparameters:")
    print(changed_hp.get_config())

    model.summary()

    # use tensorboard to inspect the graph, write log file periodically:
    out_dir = training_outdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = out_dir + "/log"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # periodically save the weights to a checkpoint file:
    checkpoint_path = out_dir + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    verbose=1, save_weights_only=True)


    model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_data=(x_val, y_val), 
    callbacks=[tensorboard_callback, cp_callback])

    print("\nOn test dataset:")
    model.evaluate(x_test,  y_test, verbose=2)
    print()

    model.save(out_dir + "/model")

# TODO: 
# - Implement this also for lstm
# - Can I do the search without saving the model as a checkpoint?