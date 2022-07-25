from UNet_1DCNN import UNet
import tensorflow.keras as keras
import os
from dataset_simulations.simulation import Simulation

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import contextlib

sys.path.append("../")
import generate_background_noise_utils
from datetime import datetime
import pickle

tag = "UNetPP"
training_mode = "train"  # possible: train and test

# to_test = "removal_03-12-2021_16-48-30_UNetPP" # pretty damn good
to_test = "06-06-2022_22-15-44_UNetPP"

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

batch_size = 300
number_of_batches = 500
number_of_epochs = 2000

use_distributed_strategy = True
use_ICSD_patterns = False
use_caglioti = True

local = False

if not local:
    NO_workers = 30
    verbosity = 2
else:
    NO_workers = 7
    verbosity = 1

print(
    f"Training with {batch_size * number_of_batches * number_of_epochs} samples in total"
)

out_base = "unet/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + tag + "/"

# with open("unet/scaler", "rb") as file:
#    scaler = pickle.load(file)

if training_mode == "train":

    if use_ICSD_patterns or use_caglioti:

        with open("../dataset_simulations/prepared_training/meta", "rb") as file:

            data = pickle.load(file)

            per_element = data[6]

            counter_per_spg_per_element = data[0]
            if per_element:
                counts_per_spg_per_element_per_wyckoff = data[1]
            else:
                counts_per_spg_per_wyckoff = data[1]
            NO_wyckoffs_prob_per_spg = data[2]
            NO_unique_elements_prob_per_spg = data[3]

            if per_element:
                NO_repetitions_prob_per_spg_per_element = data[4]
            else:
                NO_repetitions_prob_per_spg = data[4]
            denseness_factors_per_spg = data[5]

            statistics_metas = data[7]
            statistics_labels = data[8]
            statistics_match_metas = data[9]
            statistics_match_labels = data[10]
            test_metas = data[11]
            test_labels = data[12]
            corrected_labels = data[13]
            test_match_metas = data[14]
            test_match_pure_metas = data[15]

        path_to_patterns = "../dataset_simulations/patterns/icsd_vecsei/"
        jobid = os.getenv("SLURM_JOB_ID")
        if jobid is not None and jobid != "":
            icsd_sim_statistics = Simulation(
                os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
                os.path.expanduser("~/Databases/ICSD/cif/"),
            )
            icsd_sim_statistics.output_dir = path_to_patterns
        else:  # local
            icsd_sim_statistics = Simulation(
                "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
                "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
            )
            icsd_sim_statistics.output_dir = path_to_patterns

        statistics_match_metas_flat = [item[0] for item in statistics_match_metas]

        icsd_sim_statistics.load(
            load_only_N_patterns_each=1,
            metas_to_load=statistics_match_metas_flat,
            stop=1 if local else None,
        )
        statistics_patterns = [j for i in icsd_sim_statistics.sim_patterns for j in i]
        statistics_angles = icsd_sim_statistics.sim_angles
        statistics_intensities = icsd_sim_statistics.sim_intensities

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches, number_of_epochs):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches
            self._number_of_epochs = number_of_epochs

        def __call__(self):
            """Return next batch using an infinite generator model."""

            for i in range(self.__len__() * self._number_of_epochs):
                yield self.__getitem__(i)

                if (i + 1) % self.__len__() == 0:
                    self.on_epoch_end()

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):

            # while True:
            (
                in_patterns,
                out_patterns,
            ) = generate_background_noise_utils.generate_samples_gp(
                self.batch_size,
                (start_x, end_x),
                n_angles_output=N,
                icsd_patterns=statistics_patterns if use_ICSD_patterns else None,
                icsd_angles=statistics_angles if use_caglioti else None,
                icsd_intensities=statistics_intensities if use_caglioti else None,
            )

            # if np.any(np.isnan(in_patterns)) or np.any(np.isnan(out_patterns)):
            #    print("Encountered NAN")
            #    continue
            # else:
            #    break

            # in_patterns, out_patterns = np.expand_dims(
            #    in_patterns, axis=2
            # ), np.expand_dims(out_patterns, axis=2)

            return (
                in_patterns,
                out_patterns,
            )

    if use_distributed_strategy:
        strategy = tf.distribute.MirroredStrategy()

    with (strategy.scope() if use_distributed_strategy else contextlib.nullcontext()):

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

        if use_distributed_strategy:
            sequence = CustomSequence(batch_size, number_of_batches, number_of_epochs)
            dataset = tf.data.Dataset.from_generator(
                sequence,
                output_types=(tf.float64, tf.float64),
                output_shapes=(
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                ),
            )

        model.fit(
            x=sequence if not use_distributed_strategy else dataset,
            epochs=number_of_epochs,
            verbose=verbosity,
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
