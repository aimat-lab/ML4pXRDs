from UNet_1DCNN import UNet
import tensorflow.keras as keras

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

import matplotlib.pyplot as plt
import sys

sys.path.append("../")

import generate_background_noise_utils

# import generate_nackground_noise_utils_old

mode = "removal"  # possible: "info", "removal"
training_mode = "test"  # possible: train and test

N = 9018
pattern_x = np.linspace(0, 90, N)

batch_size = 128
number_of_batches = 500
number_of_epochs = 50
# cache_multiplier = 100

print(f"Training with {batch_size * number_of_batches * number_of_epochs} samples")

# only use a restricted range of the simulated patterns
start_x = 10
end_x = 50
start_index = np.argwhere(pattern_x >= start_x)[0][0]
end_index = np.argwhere(pattern_x <= end_x)[-1][0]
pattern_x = pattern_x[start_index : end_index + 1]
N = len(pattern_x)

print(f"Actual N of used range: {N}")

# with open("unet/scaler", "rb") as file:
#    scaler = pickle.load(file)

if training_mode == "train":

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches, mode, start_index, end_index):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches
            self.mode = mode
            self.start_index = start_index
            self.end_index = end_index

            self.data = None
            self.counter = 0
            # self.cache_multiplier = cache_multiplier

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):
            """
            # print()
            # print(self.counter)
            # print()

            # cache patterns to make it faster
            if self.data is None or self.counter == self.cache_multiplier:
                self.data = generate_background_noise_utils.generate_samples_gp(
                    n_samples=self.batch_size * self.cache_multiplier,
                    mode=self.mode,
                    do_plot=False,
                )
                print()
                print()
                print("New data")
                print()
                print()

                # self.data = generate_nackground_noise_utils_old.generate_samples(
                #    N=self.cache_multiplier * self.batch_size
                # )

                self.counter = 0

            # xs = scaler.transform(batch[0])

            # print(self.counter)

            """

            data = generate_background_noise_utils.generate_samples_gp(
                n_samples=self.batch_size, mode=self.mode, do_plot=False,
            )

            """
            xs = self.data[0][
                self.counter * self.batch_size : (self.counter + 1) * self.batch_size, :
            ]
            ys = self.data[1][
                self.counter * self.batch_size : (self.counter + 1) * self.batch_size, :
            ]
            """

            # self.counter += 1

            return (
                data[0][:, self.start_index : self.end_index + 1],
                data[1][:, self.start_index : self.end_index + 1],
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
            metrics=["binary_crossentropy", "mean_squared_error"],
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
        max_queue_size=500,
        workers=8,
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

    test_batch = generate_background_noise_utils.generate_samples_gp(
        n_samples=100, mode=mode
    )

    # test_batch = generate_nackground_noise_utils_old.generate_samples(N=100, mode=mode)

    # test_xs = scaler.transform(test_batch[0])
    test_xs = test_batch[0]

    x_test, y_test = (
        test_xs[:, start_index : end_index + 1],
        test_batch[1][:, start_index : end_index + 1],
    )

    if mode == "removal":
        predictions = model.predict(x_test)
    else:
        probability_model = keras.Sequential(
            [model, keras.layers.Activation("sigmoid")]
        )
        predictions = probability_model.predict(x_test)

    os.system("mkdir -p predictions")
    for i, prediction in enumerate(predictions):
        if mode == "removal":

            plt.plot(pattern_x, prediction, label="Prediction")

            plt.plot(
                pattern_x,
                test_batch[0][:, start_index : end_index + 1][i],
                label="Input pattern",
            )

            plt.plot(
                pattern_x,
                test_batch[1][:, start_index : end_index + 1][i],
                label="Target",
            )

            plt.legend()

            plt.savefig(f"predictions/prediction_{i}.pdf")

            plt.show()
            plt.figure()
        elif mode == "info":
            plt.scatter(pattern_x, prediction, s=3)
            plt.scatter(pattern_x, y_test[i], s=3)

            plt.plot(pattern_x, x_test[i])
            plt.savefig(f"predictions/prediction_{i}.pdf")
            plt.show()
            plt.figure()
        else:
            raise Exception("Mode not recognized.")

