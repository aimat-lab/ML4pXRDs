import time
from UNet_1DCNN import UNet
import tensorflow.keras as keras
import os
from utils.simulation.simulation import Simulation

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
import ray
from ray.util.queue import Queue
import utils.matplotlib_defaults as matplotlib_defaults

figure_double_width_pub = matplotlib_defaults.pub_width

tag = "UNetPP"
training_mode = "test"  # possible: train and test
N_to_plot = 20  # only for test
plot_separately = True
show_plots = False

# to_test = "removal_03-12-2021_16-48-30_UNetPP" # pretty damn good
# to_test = "06-06-2022_22-15-44_UNetPP"
# to_test = "30-07-2022_10-20-17_UNetPP"
# to_test = "31-07-2022_12-40-47"
# to_test = "10-06-2022_13-12-26_UNetPP"
to_test = "05-08-2022_07-59-47"

continue_run = True  # TODO: Change back
pretrained_model_path = (
    "/home/ws/uvgnh/MSc/HEOs_MSc/train_dataset/unet/31-07-2022_12-40-47/final"
)

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

batch_size = 300
number_of_batches = 500
number_of_epochs = 1000

use_distributed_strategy = True
use_ICSD_patterns = False
use_caglioti = True

local = True
if not local:
    NO_workers = 28 + 128
    verbosity = 2
else:
    NO_workers = 7
    verbosity = 1

print(
    f"Training with {batch_size * number_of_batches * number_of_epochs} samples in total"
)

if len(sys.argv) > 1:
    date_time = sys.argv[1]  # get it from the bash script
    out_base = "unet/" + date_time + "/"
else:
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_base = "unet/" + date_time + "_" + tag + "/"

# with open("unet/scaler", "rb") as file:
#    scaler = pickle.load(file)

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
        stop=10 if local else None,
        load_patterns_angles_intensities=True,
        load_only_angles_intensities=use_caglioti,  # only angles and intensities needed in this case!
    )
    if not use_caglioti:
        statistics_patterns = [j for i in icsd_sim_statistics.sim_patterns for j in i]
    else:
        statistics_patterns = None
    statistics_angles = icsd_sim_statistics.sim_angles
    statistics_intensities = icsd_sim_statistics.sim_intensities

    if training_mode == "test":  # read some patterns from the match dataset for testing

        if jobid is not None and jobid != "":
            icsd_sim_test = Simulation(
                os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
                os.path.expanduser("~/Databases/ICSD/cif/"),
            )
            icsd_sim_test.output_dir = path_to_patterns
        else:  # local
            icsd_sim_test = Simulation(
                "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
                "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
            )
            icsd_sim_test.output_dir = path_to_patterns

        test_match_metas_flat = [item[0] for item in test_match_metas]

        icsd_sim_test.load(
            load_only_N_patterns_each=1,
            metas_to_load=test_match_metas_flat,
            stop=10 if local else None,
            load_patterns_angles_intensities=True,
            load_only_angles_intensities=use_caglioti,  # only angles and intensities needed in this case!
        )
        if not use_caglioti:
            test_patterns = [j for i in icsd_sim_test.sim_patterns for j in i]
        else:
            test_patterns = None
        test_angles = icsd_sim_test.sim_angles
        test_intensities = icsd_sim_test.sim_intensities


if training_mode == "train":

    log_wait_timings = []

    class CustomSequence(keras.utils.Sequence):
        def __init__(self, batch_size, number_of_batches, number_of_epochs, queue):
            self.batch_size = batch_size
            self.number_of_batches = number_of_batches
            self._number_of_epochs = number_of_epochs
            self.queue = queue

        def __call__(self):
            """Return next batch using an infinite generator model."""

            for i in range(self.__len__() * self._number_of_epochs):
                yield self.__getitem__(i)

                if (i + 1) % self.__len__() == 0:
                    self.on_epoch_end()

        def __len__(self):
            return self.number_of_batches

        def __getitem__(self, idx):
            start = time.time()
            result = self.queue.get()
            log_wait_timings.append(time.time() - start)
            return result

    # log to tensorboard
    file_writer = tf.summary.create_file_writer(out_base + "metrics")

    ray.init(
        address="localhost:6379" if not local else None,
        include_dashboard=False,
    )

    print()
    print(ray.cluster_resources())
    print()

    queue = Queue(maxsize=100)

    icsd_patterns_handle = ray.put(statistics_patterns)
    statistics_angles_np = [np.array(item) for item in statistics_angles]
    icsd_angles_handle = ray.put(statistics_angles_np)
    statistics_intensities_np = [np.array(item) for item in statistics_intensities]
    icsd_intensities_handle = ray.put(statistics_intensities_np)

    @ray.remote(num_cpus=1, num_gpus=0)
    def batch_generator_queue(
        queue,
        batch_size,
        icsd_patterns_handle,
        icsd_angles_handle,
        icsd_intensities_handle,
    ):

        # icsd_patterns = ray.get(icsd_patterns_handle)
        # icsd_angles = ray.get(icsd_angles_handle)
        # icsd_intensities = ray.get(icsd_intensities_handle)

        while True:
            (
                in_patterns,
                out_patterns,
            ) = generate_background_noise_utils.generate_samples_gp(
                batch_size,
                (start_x, end_x),
                n_angles_output=N,
                icsd_patterns=icsd_patterns_handle,
                icsd_angles=icsd_angles_handle,
                icsd_intensities=icsd_intensities_handle,
                use_caglioti=use_caglioti,
                use_ICSD_patterns=use_ICSD_patterns,
            )
            queue.put((in_patterns, out_patterns))

    for i in range(0, NO_workers):
        batch_generator_queue.remote(
            queue,
            batch_size,
            icsd_patterns_handle,
            icsd_angles_handle,
            icsd_intensities_handle,
        )

    if use_distributed_strategy:
        strategy = tf.distribute.MirroredStrategy()

    with (strategy.scope() if use_distributed_strategy else contextlib.nullcontext()):

        if not continue_run:
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

        else:

            model = keras.models.load_model(pretrained_model_path)

        if use_distributed_strategy:
            sequence = CustomSequence(
                batch_size, number_of_batches, number_of_epochs, queue
            )
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
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False,
            callbacks=[
                keras.callbacks.TensorBoard(log_dir=out_base + "tb"),
            ],
            steps_per_epoch=number_of_batches,
        )

        model.save(out_base + "final")

        ray.shutdown()

        with file_writer.as_default():
            for i, value in enumerate(log_wait_timings):
                tf.summary.scalar("waiting time", data=value, step=i)

else:

    model = keras.models.load_model("unet/" + to_test + "/final")

    test_batch = generate_background_noise_utils.generate_samples_gp(
        2000,
        (start_x, end_x),
        n_angles_output=N,
        use_caglioti=use_caglioti,
        icsd_angles=test_angles,
        icsd_intensities=test_intensities,
        use_ICSD_patterns=use_ICSD_patterns,
        icsd_patterns=test_patterns,
    )

    x_test, y_test = test_batch[0], test_batch[1]

    if False:
        for i in range(x_test.shape[0]):
            repeated_x = np.repeat(x_test[i, 1700:1701], x_test.shape[1] - 1700)
            repeated_y = np.repeat([0], y_test.shape[1] - 1700)
            x_test[i, :] = np.concatenate((x_test[i, 0:1700], repeated_x))
            y_test[i, :] = np.concatenate((y_test[i, 0:1700], repeated_y))

    print("Model evaluation:")
    print(model.evaluate(x_test, y_test))

    predictions = model.predict(x_test)

    os.system("mkdir -p predictions")

    for i, prediction in enumerate(predictions[0:N_to_plot]):

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )

        plt.xlabel(r"$2 \theta$")
        plt.ylabel("Intensity / rel.")

        if not plot_separately:

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
            plt.tight_layout()

            plt.savefig(f"predictions/test_{i}.pdf", bbox_inches="tight")

            if show_plots:
                plt.show()

        else:

            plt.plot(
                pattern_x,
                test_batch[0][:][i],
                label="Input pattern",
            )
            # plt.legend()
            plt.tight_layout()
            plt.savefig(f"predictions/input_{i}.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()

            plt.figure(
                figsize=(
                    figure_double_width_pub * 0.95 * 0.5,
                    figure_double_width_pub * 0.7 * 0.5,
                )
            )
            plt.xlabel(r"$2 \theta$")
            plt.ylabel("Intensity / rel.")
            plt.plot(
                pattern_x,
                test_batch[1][:][i],
                label="Target",
            )
            # plt.legend()
            plt.tight_layout()
            plt.savefig(f"predictions/target_{i}.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()

            plt.figure(
                figsize=(
                    figure_double_width_pub * 0.95 * 0.5,
                    figure_double_width_pub * 0.7 * 0.5,
                )
            )
            plt.xlabel(r"$2 \theta$")
            plt.ylabel("Intensity / rel.")
            plt.plot(pattern_x, prediction[:, 0], label="Prediction")
            # plt.legend()
            plt.tight_layout()
            plt.savefig(f"predictions/prediction_{i}.pdf", bbox_inches="tight")
            if show_plots:
                plt.show()
