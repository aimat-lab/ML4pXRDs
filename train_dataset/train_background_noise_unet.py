from UNet_1DCNN import UNet
import tensorflow.keras as keras
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import generate_background_noise_utils
from datetime import datetime

tag = "scherer"
mode = "removal"  # possible: "info", "removal"
training_mode = "train"  # possible: train and test

to_test = "removal_01-12-2021_13-23-10_variable_variance"

start_x = 10.015
end_x = 50.155
N = 2677
pattern_x = np.linspace(start_x, end_x, N)

# UNet only works for certain input sizes
# works: 2672, 2688
pattern_x = pattern_x[0:2672]
N = 2672

batch_size = 128
number_of_batches = 500
number_of_epochs = 100
NO_workers = 16

print(f"Training with {batch_size * number_of_batches * number_of_epochs} samples")

# only use a restricted range of the simulated patterns

print(f"Actual N of used range: {N}")

out_base = (
    "unet/"
    + mode
    + "_"
    + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    + "_"
    + tag
    + "/"
)

# with open("unet/scaler", "rb") as file:
#    scaler = pickle.load(file)

if training_mode == "train":

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches, mode):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches
            self.mode = mode

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
                self.batch_size, (start_x, end_x), n_angles_output=N
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
                data[0],
                data[1],
            )

    # for i in range(1000, 2000):
    # my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
    # try:

    # UNet works without error for N ~ 2^model_depth
    my_unet = UNet(
        length=N,
        model_depth=4,  # height
        num_channel=1,  # input
        model_width=5,  # first conv number of channels, danach immer verdoppeln
        kernel_size=64,
        output_nums=1,
        problem_type="Regression",
    )
    model = my_unet.UNet()

    # print(f"Worked {i}")
    # except:
    # pass
    # exit()

    # length: Input Signal Length
    # model_depth: Depth of the Model
    # model_width: Width of the Input Layer of the Model
    # num_channel: Number of Channels allowed by the Model
    # kernel_size: Kernel or Filter Size of the Convolutional Layers
    # problem_type: Classification (Binary or Multiclass) or Regression
    # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
    # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
    # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
    # alpha: This Parameter is only for MultiResUNet, default value is 1
    # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
    # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer

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
        filepath=out_base + "cps" + "/weights{epoch}",
        save_weights_only=True,
        verbose=1,
    )

    model.fit(
        x=CustomSequence(batch_size, number_of_batches, mode),
        epochs=number_of_epochs,
        verbose=2,
        max_queue_size=500,
        workers=NO_workers,
        use_multiprocessing=True,
        callbacks=[
            cp_callback,
            keras.callbacks.TensorBoard(log_dir=out_base + "tb"),
        ],
        steps_per_epoch=number_of_batches,
    )

    model.save(out_base + "final")

else:

    model = keras.models.load_model("unet/" + to_test + "/final")

    test_batch = generate_background_noise_utils.generate_samples_gp(
        100, (start_x, end_x), n_angles_output=N
    )

    # test_batch = generate_nackground_noise_utils_old.generate_samples(N=100, mode=mode)

    # test_xs = scaler.transform(test_batch[0])
    test_xs = test_batch[0]

    x_test, y_test = test_batch[0], test_batch[1]

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

            plt.plot(pattern_x, prediction[:, 0], label="Prediction")

            plt.plot(
                pattern_x,
                test_batch[0][:][i],
                label="Input pattern",
            )

            plt.plot(
                pattern_x,
                test_batch[1][:][i],
                label="Target",
            )

            plt.plot(
                pattern_x,
                test_batch[0][:][i] - prediction[:, 0],
                label="Background and noise",
                linestyle="dotted",
            )

            plt.legend()

            # plt.savefig(f"predictions/prediction_{i}.pdf")

            plt.show()
            plt.figure()
        elif mode == "info":
            plt.scatter(pattern_x, prediction, s=3)
            plt.scatter(pattern_x, y_test[i], s=3)

            plt.plot(pattern_x, x_test[i])
            # plt.savefig(f"predictions/prediction_{i}.pdf")
            plt.show()
            plt.figure()
        else:
            raise Exception("Mode not recognized.")
