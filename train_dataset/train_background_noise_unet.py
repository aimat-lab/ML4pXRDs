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

tag = "UNetPP"
training_mode = "train"  # possible: train and test

# to_test = "removal_03-12-2021_16-48-30_UNetPP" # pretty damn good
to_test = "09-05-2022_12-24-34_UNetPP"

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

batch_size = 300
number_of_batches = 500
number_of_epochs = 600
NO_workers = 32

print(
    f"Training with {batch_size * number_of_batches * number_of_epochs} samples in total"
)

out_base = "unet/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + tag + "/"

# with open("unet/scaler", "rb") as file:
#    scaler = pickle.load(file)

if training_mode == "train":

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):

            data = generate_background_noise_utils.generate_samples_gp(
                self.batch_size, (start_x, end_x), n_angles_output=N
            )

            return (
                data[0],
                data[1],
            )

    my_unet = UNet(
        length=N,
        model_depth=5,  # height
        num_channel=1,  # input
        model_width=5,  # first conv number of channels, danach immer verdoppeln
        kernel_size=64,
        output_nums=1,
        problem_type="Regression",
    )
    # model = my_unet.UNet()
    model = my_unet.UNetPP()

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

    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

    # cp_callback = keras.callbacks.ModelCheckpoint(
    #    filepath=out_base + "cps" + "/weights{epoch}",
    #    save_weights_only=True,
    #    verbose=1,
    # )

    model.fit(
        x=CustomSequence(batch_size, number_of_batches),
        epochs=number_of_epochs,
        verbose=2,
        max_queue_size=500,
        workers=NO_workers,
        use_multiprocessing=True,
        # callbacks=[cp_callback, keras.callbacks.TensorBoard(log_dir=out_base + "tb"),],
        callbacks=[
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

    # test_xs = scaler.transform(test_batch[0])
    test_xs = test_batch[0]

    x_test, y_test = test_batch[0], test_batch[1]

    if False:
        for i in range(x_test.shape[0]):
            repeated_x = np.repeat(x_test[i, 1700:1701], x_test.shape[1] - 1700)
            repeated_y = np.repeat([0], y_test.shape[1] - 1700)
            x_test[i, :] = np.concatenate((x_test[i, 0:1700], repeated_x))
            y_test[i, :] = np.concatenate((y_test[i, 0:1700], repeated_y))

    predictions = model.predict(x_test)

    os.system("mkdir -p predictions")
    for i, prediction in enumerate(predictions):

        plt.xlabel(r"$2 \theta$")
        plt.ylabel("Intensity")

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
            label="Prediced background and noise",
            linestyle="dotted",
        )

        plt.legend()

        # plt.savefig(f"predictions/prediction_{i}.png")

        plt.show()
        plt.figure()
