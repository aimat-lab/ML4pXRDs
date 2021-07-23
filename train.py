import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
import datetime
import os

filenames = ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv", "dataset_4.csv", "dataset_5.csv", "dataset_6.csv", "dataset_7.csv", "dataset_8.csv"]
model = "fully_connected" # also possible: lstm, fully_connected

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

if model == "lstm" or model == "conv":
    x = np.expand_dims(x, axis=2)

# Split into train, validation, test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

if model == "lstm":

    model = tf.keras.models.Sequential([

        tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(181, 1)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(14)
    ])

elif model == "conv":

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(16, 40, input_shape=(181, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),

        tf.keras.layers.Conv1D(16, 20, input_shape=(181, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0007)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(14)

    ])

elif model == "fully_connected":

    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0007), input_shape=(181,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0007)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(14, activation='relu', kernel_regularizer=regularizers.l2(0.0007))

    ])

else:
    raise Exception("Model not recognized.")

model.summary()

# negative log probability of the true class
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002,
                beta_1=0.9, beta_2=0.999),
                loss=loss_fn,
                metrics=['accuracy'])

#model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, clipvalue=0.5),
#                loss=loss_fn,
#                metrics=['accuracy'])

# use tensorboard to inspect the graph, write log file periodically:
out_dir = "trainings/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = out_dir + "/log"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# periodically save the weights to a checkpoint file:
checkpoint_path = out_dir + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1, save_weights_only=True)


model.fit(x_train, y_train, epochs=20, batch_size=1000, validation_data=(x_val, y_val), 
callbacks=[tensorboard_callback, cp_callback])

print("\nOn test dataset:")
model.evaluate(x_test,  y_test, verbose=2)
print()

model.save(out_dir + "/model")