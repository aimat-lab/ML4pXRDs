from UNet_1DCNN import UNet
import tensorflow.keras as keras
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import pickle
import os
import sys

sys.path.append("../")
import generate_background_noise_utils

mode = "removal"  # also possible: "info"

training_mode = "train"  # possible: train and test

N = 9018
pattern_x = np.linspace(0, 90, N)

batch_size = 128
number_of_batches = 500
number_of_epochs = 20

print(f"Training with {batch_size * number_of_batches * number_of_epochs} samples")

# only use a restricted range of the simulated patterns
start_x = 10
end_x = 50
start_index = np.argwhere(pattern_x >= start_x)[0][0]
end_index = np.argwhere(pattern_x <= end_x)[-1][0]
pattern_x = pattern_x[start_index : end_index + 1]
N = len(pattern_x)

# TODO: Since we generate the dataset on the fly, there is no standard scaler involved. Is this a problem?

if training_mode == "train":

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches, mode, start_index, end_index):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches
            self.mode = mode
            self.start_index = start_index
            self.end_index = end_index

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):
            batch = generate_background_noise_utils.generate_samples(
                N=self.batch_size, mode=self.mode
            )
            return (
                batch[0][:, self.start_index : self.end_index + 1],
                batch[1][:, self.start_index : self.end_index + 1],
            )

    my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
    model = my_unet.UNet()

    # keras.utils.plot_model(model, show_shapes=True)

    model.summary()

    if mode == "removal":
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    elif mode == "info":
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
    else:
        raise Exception("Mode not supported.")

    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath="unet/" + mode + "_cps" + "/weights{epoch}",
        save_weights_only=True,
        verbose=1,
    )

    model.fit(
        x=CustomSequence(batch_size, number_of_batches, mode, start_index, end_index),
        epochs=number_of_epochs,
        verbose=1,
        max_queue_size=10,
        workers=3,
        use_multiprocessing=True,
        callbacks=[
            cp_callback,
            keras.callbacks.TensorBoard(log_dir="unet/" + mode + "_tb"),
        ],
        steps_per_epoch=number_of_batches,
    )

    model.save("unet/" + mode + "_final")

else:

    model = keras.models.load_model("unet/" + mode + "_final")

    test_batch = generate_background_noise_utils.generate_samples(N=100, mode=mode)
    x_test, y_test = (
        test_batch[0][:, start_index : end_index + 1],
        test_batch[1][:, start_index : end_index + 1],
    )

    if mode == "removal":
        predictions = model.predict(x_test)
    else:
        probability_model = keras.Sequential(
            [model, keras.layers.Activation("sigmoid")]
        )
        predictions = probability_model.predict(x_test[0:20])

    os.system("mkdir -p predictions")
    for i, prediction in enumerate(predictions):
        if mode == "removal":
            plt.plot(pattern_x, prediction)
            plt.plot(pattern_x, x_test[i])
            plt.savefig(f"predictions/prediction_{i}.pdf")
        elif mode == "info":
            plt.scatter(pattern_x, prediction)
            plt.plot(pattern_x, x_test[i])
            plt.savefig(f"predictions/prediction_{i}.pdf")
        else:
            raise Exception("Mode not recognized.")
