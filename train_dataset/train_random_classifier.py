import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
from dataset_simulations.random_simulation_utils import load_wyckoff_statistics
import numpy as np
from models import build_model_park
from datetime import datetime
import os
from sklearn.utils import shuffle
from dataset_simulations.simulation import Simulation
from scipy import interpolate as ip
import ray
from ray.util.queue import Queue
import pickle
import tensorflow as tf
import sys

# tag = "spgs-2-15"
# tag = "4-spgs-no-distance-check"
tag = "2-spgs-new_generation_max_volume"

out_base = sys.argv[1] + "/"
# out_base = (
#    "classifier_spgs/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_" + tag + "/"
# )
os.system("mkdir -p " + out_base)
os.system("mkdir -p " + out_base + "tuner_tb")
os.system("touch " + out_base + tag)

test_every_X_epochs = 1
batches_per_epoch = 1500
# NO_epochs = 1000
NO_epochs = 200

# structures_per_spg = 1 # for all spgs
# structures_per_spg = 5
structures_per_spg = 10 # for (2,15) tuple
NO_corn_sizes = 5
# => 4*5*5=100 batch size (for 4 spgs)
do_distance_checks = False
do_merge_checks = False
use_icsd_statistics = True

NO_workers = 126 + 14  # for cluster
queue_size = 200
queue_size_tf = 100

compare_distributions = True
# NO_random_batches = 20
NO_random_batches = 1000  # make this smaller for the all-spgs run

generation_max_volume = 7000
generation_max_NO_wyckoffs = 100

validation_max_volume = 7000  # None possible
validation_max_NO_wyckoffs = 100  # None possible

verbosity = 2

local = False
if local:
    NO_workers = 8
    verbosity = 1

# spgs = [14, 104] # works well, relatively high val_acc
# spgs = [129, 176] # 93.15%, pretty damn well!
spgs = [2, 15] # pretty much doesn't work at all (so far!), val_acc ~40%, after a full night: ~43%
# after a full night with random volume factors: binary_accuracy: 0.7603 - val_loss: 0.8687 - val_binary_accuracy: 0.4749; still bad
# spgs = [14, 104, 129, 176]  # after 100 epochs: 0.8503 val accuracy
# all spgs (~200): loss: sparse_categorical_accuracy: 0.1248 - val_sparse_categorical_accuracy: 0.0713; it is a beginning!

# like in the Vecsei paper:
start_angle, end_angle, N = 5, 90, 8501
start_angle, end_angle, N = (
    360
    / (2 * np.pi)
    * np.arcsin(1.207930 / 1.5406 * np.sin(2 * np.pi / 360 * start_angle)),
    360
    / (2 * np.pi)
    * np.arcsin(1.207930 / 1.5406 * np.sin(2 * np.pi / 360 * end_angle)),
    8501,
)  # until ICSD has not been re-simulated with Cu-K line
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}")

# load validation set (ICSD):

number_of_values_initial = 9018
simulated_range = np.linspace(0, 90, number_of_values_initial)
start_index = np.argwhere(simulated_range >= start_angle)[0][0]
end_index = np.argwhere(simulated_range <= end_angle)[-1][0]
used_range = simulated_range[start_index : end_index + 1]

path_to_patterns = "../dataset_simulations/patterns/icsd/"
jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    icsd_sim = Simulation(
        "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/ws/uvgnh/Databases/ICSD/cif/",
    )
    icsd_sim.output_dir = path_to_patterns
else:
    icsd_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
    icsd_sim.output_dir = path_to_patterns

icsd_sim.load(load_only=10 if not local else 2)

n_patterns_per_crystal = len(icsd_sim.sim_patterns[0])

icsd_patterns = icsd_sim.sim_patterns
icsd_labels = icsd_sim.sim_labels
icsd_variations = icsd_sim.sim_variations
icsd_crystals = icsd_sim.sim_crystals
icsd_metas = icsd_sim.sim_metas

"""
dist_y = []
for i, label in enumerate(icsd_labels):
    dist_y.append(label[0])
print(np.bincount(dist_y))
"""

# spgs = sorted(np.unique([item[0] for item in icsd_labels]))

for i in reversed(range(0, len(icsd_patterns))):

    if validation_max_NO_wyckoffs is not None:
        _, NO_wyckoffs, _, _ = icsd_sim.get_wyckoff_info(icsd_metas[i][0])

    if (
        np.any(np.isnan(icsd_variations[i][0]))
        or icsd_labels[i][0] not in spgs
        or (
            validation_max_volume is not None
            and icsd_crystals[i].volume > validation_max_volume
        )
        or (
            validation_max_NO_wyckoffs is not None
            and NO_wyckoffs > validation_max_NO_wyckoffs
        )
    ):
        del icsd_patterns[i]
        del icsd_labels[i]
        del icsd_variations[i]
        del icsd_crystals[i]
        del icsd_metas[i]

print(f"{len(icsd_crystals)} crystals remaining in the ICSD validation set.")

if compare_distributions:

    with open(out_base + "spgs.pickle", "wb") as file:
        pickle.dump(spgs, file)

    with open(out_base + "icsd_data.pickle", "wb") as file:
        pickle.dump(
            (
                icsd_crystals,
                icsd_labels,
                [item[:, 0] for item in icsd_variations],
                icsd_metas,
            ),
            file,
        )

val_y = []
for i, label in enumerate(icsd_labels):
    val_y.extend([spgs.index(label[0])] * n_patterns_per_crystal)
val_y = np.array(val_y)

for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y==i)}")

val_x = []
for pattern in icsd_patterns:
    for sub_pattern in pattern:

        f = ip.CubicSpline(
            used_range, sub_pattern[start_index : end_index + 1], bc_type="natural"
        )
        actual_val = f(angle_range)

        val_x.append(actual_val)

assert not np.any(np.isnan(val_x))
assert not np.any(np.isnan(val_y))
assert len(val_x) == len(val_y)

val_x = np.expand_dims(val_x, axis=2)

if not local:
    ray.init(address="auto", include_dashboard=False)
    # ray.init(include_dashboard=True, num_cpus=NO_workers)
else:
    ray.init(include_dashboard=False)

print()
print(ray.cluster_resources())
print()

queue = Queue(maxsize=queue_size)  # store a maximum of `queue_size` batches

if use_icsd_statistics:
    (
        probability_per_element,
        probability_per_spg_per_wyckoff,
    ) = load_wyckoff_statistics()
else:
    probability_per_element, probability_per_spg_per_wyckoff = None, None


@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_with_additional(
    spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements, NO_corn_sizes
):

    patterns, labels, structures, corn_sizes = get_random_xy_patterns(
        spgs=spgs,
        structures_per_spg=structures_per_spg,
        # wavelength=1.5406,  # TODO: Cu-K line
        wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
        N=N,
        NO_corn_sizes=NO_corn_sizes,
        two_theta_range=(start_angle, end_angle),
        max_NO_elements=max_NO_elements,
        do_print=False,
        return_additional=True,
        do_distance_checks=do_distance_checks,
        do_merge_checks=do_merge_checks,
        use_icsd_statistics=use_icsd_statistics,
        probability_per_element=probability_per_element,
        probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
        max_volume=generation_max_volume,
    )

    # Set the label to the right index:
    for i in range(0, len(labels)):
        labels[i] = spgs.index(labels[i])

    patterns = np.array(patterns)
    labels = np.array(labels)

    return patterns, labels, structures, corn_sizes


@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_queue(
    queue,
    spgs,
    structures_per_spg,
    N,
    start_angle,
    end_angle,
    max_NO_elements,
    NO_corn_sizes,
):

    while True:
        try:

            patterns, labels = get_random_xy_patterns(
                spgs=spgs,
                structures_per_spg=structures_per_spg,
                # wavelength=1.5406,  # TODO: Cu-K line
                wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
                N=N,
                NO_corn_sizes=NO_corn_sizes,
                two_theta_range=(start_angle, end_angle),
                max_NO_elements=max_NO_elements,
                do_print=False,
                do_distance_checks=do_distance_checks,
                do_merge_checks=do_merge_checks,
                use_icsd_statistics=use_icsd_statistics,
                probability_per_element=probability_per_element,
                probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
                max_volume=generation_max_volume,
            )

            patterns, labels = shuffle(patterns, labels)

            # Set the label to the right index:
            for i in range(0, len(labels)):
                labels[i] = spgs.index(labels[i])

            patterns = np.array(patterns)
            patterns = np.expand_dims(patterns, axis=2)

            labels = np.array(labels)

            queue.put((patterns, labels))  # blocks if queue is full, which is good

        except Exception as ex:

            print("Error occurred in worker:")
            print(ex)
            print(
                type(ex).__name__,  # TypeError
                __file__,  # /tmp/example.py
                ex.__traceback__.tb_lineno,  # 2
            )


if compare_distributions:
    # pre-store some batches to compare to the rightly / falsely classified icsd samples

    random_comparison_crystals = []
    random_comparison_labels = []
    random_comparison_corn_sizes = []

    object_refs = []
    for i in range(NO_random_batches):
        ref = batch_generator_with_additional.remote(
            spgs, 1, N, start_angle, end_angle, generation_max_NO_wyckoffs, 1
        )
        # ref = batch_generator_with_additional(
        #    spgs, 1, N, start_angle, end_angle, max_NO_elements, 1
        # )
        object_refs.append(ref)

    results = ray.get(object_refs)
    # results = object_refs

    for result in results:
        patterns, labels, crystals, corn_sizes = result
        random_comparison_crystals.extend(crystals)
        random_comparison_labels.extend(labels)
        random_comparison_corn_sizes.extend(corn_sizes)

    with open(out_base + "random_data.pickle", "wb") as file:
        pickle.dump(
            (
                random_comparison_crystals,
                random_comparison_labels,
                random_comparison_corn_sizes,
            ),
            file,
        )

# Start worker tasks
for i in range(0, NO_workers):
    batch_generator_queue.remote(
        queue,
        spgs,
        structures_per_spg,
        N,
        start_angle,
        end_angle,
        generation_max_NO_wyckoffs,
        NO_corn_sizes,
    )

tb_callback = keras.callbacks.TensorBoard(out_base + "tuner_tb")

# log parameters to tensorboard
file_writer = tf.summary.create_file_writer(out_base + "tuner_tb" + "/metrics")
file_writer.set_as_default()

params_txt = (
    f"tag: {tag}  \n  \n"
    f"batches_per_epoch: {batches_per_epoch}  \n"
    f"NO_epochs: {NO_epochs}  \n"
    f"structures_per_spg: {structures_per_spg}  \n"
    f"NO_corn_sizes: {NO_corn_sizes}  \n"
    f"-> batch size: {NO_corn_sizes*structures_per_spg*len(spgs)}  \n  \n"
    f"NO_workers: {NO_workers}  \n"
    f"queue_size: {queue_size}  \n"
    f"queue_size_tf: {queue_size_tf}  \n  \n"
    f"max_NO_elements: {generation_max_NO_wyckoffs}  \n"
    f"start_angle: {start_angle}  \n"
    f"end_angle: {end_angle}  \n"
    f"N: {N}  \n  \n"
    f"do_distance_checks: {str(do_distance_checks)}  \n  \n"
    f"do_merge_checks: {str(do_merge_checks)}  \n  \n"
    f"use_icsd_statistics: {str(use_icsd_statistics)}  \n  \n"
    f"validation_max_volume: {str(validation_max_volume)}  \n  \n"
    f"validation_max_NO_wyckoffs: {str(validation_max_NO_wyckoffs)}"
)
tf.summary.text("Parameters", data=params_txt, step=0)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar("ray queue size", data=queue.size(), step=epoch)


class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):

        """
        patterns, labels = get_random_xy_patterns(
            spgs=spgs,
            structures_per_spg=structures_per_spg,
            #wavelength=1.5406,  # TODO: Cu-K line, when testing on ICSD data, switch to 1.207930 wavelength (scaling)
            wavelength=1.207930, # until ICSD has not been re-simulated with Cu-K line
            N=N,
            two_theta_range=(start_angle, end_angle),
            max_NO_elements=max_NO_elements,
            do_print=False,
        )

        patterns, labels = shuffle(patterns, labels)

        # Set the label to the right index:
        for i in range(0, len(labels)):
            labels[i] = spgs.index(labels[i])

        patterns = np.array(patterns)
        patterns = np.expand_dims(patterns, axis=2)

        labels = np.array(labels)

        return (
            patterns, labels
        )
        """

        return queue.get()


sequence = CustomSequence(batches_per_epoch)

model = build_model_park(None, N, len(spgs))

model.fit(
    x=sequence,
    epochs=NO_epochs,
    batch_size=batches_per_epoch,
    validation_data=(val_x, val_y),
    validation_freq=test_every_X_epochs,
    callbacks=[tb_callback, CustomCallback()],
    verbose=verbosity,
    workers=1,
    max_queue_size=queue_size_tf,
    use_multiprocessing=False,
    steps_per_epoch=batches_per_epoch,
)

model.save(out_base + "final")

if compare_distributions:

    if len(spgs) > 2:

        prediction = model.predict(val_x)
        prediction = np.argmax(prediction, axis=1)

        # print(prediction)
        # print(len(prediction))
        # print(len(val_x))
        # print(len(icsd_crystals))

        # print(rightly_indices)
        # print(falsely_indices)

    elif len(spgs) == 2:

        prob_model = keras.Sequential([model, keras.layers.Activation("sigmoid")])
        prediction = np.array(prob_model.predict(val_x))
        prediction = prediction[:, 0]
        prediction = np.where(prediction > 0.5, 1, 0)

    else:

        raise Exception("Unexpected number of spgs.")

    rightly_indices = np.argwhere(prediction == val_y)[:, 0]
    falsely_indices = np.argwhere(prediction != val_y)[:, 0]

    with open(out_base + "rightly_falsely.pickle", "wb") as file:
        pickle.dump((rightly_indices, falsely_indices), file)

ray.shutdown()

print("Everything finished.")
print("Output dir:")
print(out_base)
