import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
from dataset_simulations.random_simulation_utils import load_dataset_info
import numpy as np
from models import build_model_park
import os
from sklearn.utils import shuffle
from dataset_simulations.simulation import Simulation
import ray
from ray.util.queue import Queue
import pickle
import tensorflow as tf
import sys
from datetime import datetime
import time
import subprocess
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

tag = "4-sps_test_logging"
description = ""

if len(sys.argv) > 1:
    out_base = sys.argv[1] + "/"
else:
    out_base = (
        "classifier_spgs/"
        + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        + "_"
        + tag
        + "/"
    )

os.system("mkdir -p " + out_base)
os.system("mkdir -p " + out_base + "tuner_tb")
os.system("touch " + out_base + tag)

run_analysis_after_run = False  # TODO: Change back again.
analysis_per_spg = True

test_every_X_epochs = 1
# batches_per_epoch = 1500
batches_per_epoch = 1500  # TODO: Change back
# NO_epochs = 200
NO_epochs = 5  # TODO: Change back

# structures_per_spg = 1 # for all spgs
# structures_per_spg = 5
structures_per_spg = 10  # for (2,15) tuple
NO_corn_sizes = 5
# => 4*5*5=100 batch size (for 4 spgs)
do_distance_checks = False
do_merge_checks = False
use_icsd_statistics = True

NO_workers = 127 + 127 + 14  # for cluster
queue_size = 200
queue_size_tf = 100

# NO_random_batches = 20
NO_random_swipes = 1000  # make this smaller for the all-spgs run

generation_max_volume = 7000
generation_max_NO_wyckoffs = 100

validation_max_volume = 7000  # None possible
validation_max_NO_wyckoffs = 100  # None possible

do_symmetry_checks = True
use_NO_wyckoffs_counts = True

verbosity = 2

local = False
if local:
    NO_workers = 8
    verbosity = 1

# spgs = [14, 104] # works well, relatively high val_acc
# spgs = [129, 176] # 93.15%, pretty damn well!
spgs = [
    2,
    15,
]  # pretty much doesn't work at all (so far!), val_acc ~40%, after a full night: ~43%
# after a full night with random volume factors: binary_accuracy: 0.7603 - val_loss: 0.8687 - val_binary_accuracy: 0.4749; still bad
# spgs = [14, 104, 129, 176]  # after 100 epochs: 0.8503 val accuracy
# all spgs (~200): loss: sparse_categorical_accuracy: 0.1248 - val_sparse_categorical_accuracy: 0.0713; it is a beginning!

# as Park:
# start_angle, end_angle, N = 10, 110, 10001

# as Vecsei:
start_angle, end_angle, N = 5, 90, 8501
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}, N: {N}")

(
    probability_per_element,
    probability_per_spg_per_wyckoff,
    NO_wyckoffs_prob_per_spg,
    corrected_labels,
    files_to_use_for_test_set,
    represented_spgs,
) = load_dataset_info()

if not use_icsd_statistics:
    (
        probability_per_element,
        probability_per_spg_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
    ) = (
        None,
        None,
        None,
    )

if not use_NO_wyckoffs_counts:
    NO_wyckoffs_prob_per_spg = None

# Construct validation sets
# Used validation sets:
# - All ICSD entries
# - ICSD entries that match simulation parameters
# - Pre-computed random dataset (the one from the comparison script)
# - Gap between training and val acc that matches simulation parameters

path_to_patterns = "../dataset_simulations/patterns/icsd_vecsei/"
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

icsd_sim.load(
    start=0, stop=files_to_use_for_test_set if not local else 2
)  # to not overflow the memory

n_patterns_per_crystal = len(icsd_sim.sim_patterns[0])

icsd_patterns_all = icsd_sim.sim_patterns
icsd_labels_all = icsd_sim.sim_labels
icsd_variations_all = icsd_sim.sim_variations
icsd_crystals_all = icsd_sim.sim_crystals
icsd_metas_all = icsd_sim.sim_metas

# Mainly to make the volume constraints correct:
conventional_errors_counter = 0
print("Calculating conventional structures...")
for i in reversed(range(0, len(icsd_crystals_all))):

    try:
        current_struc = icsd_crystals_all[i]
        analyzer = SpacegroupAnalyzer(current_struc)
        conv = analyzer.get_conventional_standard_structure()
        icsd_crystals_all[i] = conv

    except Exception as ex:

        print("Error calculating conventional cell of ICSD:")
        print(ex)
        conventional_errors_counter += 1

print(
    f"{conventional_errors_counter} of {len(icsd_crystals_all)} failed to convert to conventional cell."
)

icsd_patterns_match_corrected_labels = icsd_patterns_all.copy()
icsd_labels_match_corrected_labels = (
    corrected_labels if not local else corrected_labels[: len(icsd_labels_all)]
)  # corrected labels from spglib
icsd_crystals_match_corrected_labels = icsd_crystals_all.copy()
icsd_variations_match_corrected_labels = icsd_variations_all.copy()
icsd_metas_match_corrected_labels = icsd_metas_all.copy()

assert len(icsd_labels_match_corrected_labels) == len(icsd_labels_all)

# Only train on spgs that are actually present in the ICSD:
# spgs = sorted(np.unique([item[0] for item in icsd_labels_all]))
# better:
# spgs = represented_spgs

for i in reversed(range(len(represented_spgs))):
    if np.sum(NO_wyckoffs_prob_per_spg[represented_spgs[i]][0:100]) <= 0.01:
        print(
            f"Excluded spg {represented_spgs[i]} from represented_spgs due to low probability in NO_wyckoffs < 100."
        )
        del represented_spgs[i]

for spg in spgs:
    if spg not in represented_spgs:
        raise Exception("Requested space group not represented in prepared statistics.")

for i in reversed(range(0, len(icsd_patterns_all))):

    if np.any(np.isnan(icsd_variations_all[i][0])) or icsd_labels_all[i][0] not in spgs:
        del icsd_patterns_all[i]
        del icsd_labels_all[i]
        del icsd_variations_all[i]
        del icsd_crystals_all[i]
        del icsd_metas_all[i]

for i in reversed(range(0, len(icsd_patterns_match_corrected_labels))):

    if (
        np.any(np.isnan(icsd_variations_match_corrected_labels[i][0]))
        or icsd_labels_match_corrected_labels[i] is None
        or icsd_labels_match_corrected_labels[i] not in spgs
    ):
        del icsd_patterns_match_corrected_labels[i]
        del icsd_labels_match_corrected_labels[i]
        del icsd_variations_match_corrected_labels[i]
        del icsd_crystals_match_corrected_labels[i]
        del icsd_metas_match_corrected_labels[i]

# patterns that fall into the simulation parameter range (volume and NO_wyckoffs)
icsd_patterns_match = icsd_patterns_all.copy()
icsd_labels_match = icsd_labels_all.copy()
icsd_variations_match = icsd_variations_all.copy()
icsd_crystals_match = icsd_crystals_all.copy()
icsd_metas_match = icsd_metas_all.copy()

NO_wyckoffs_cached = {}
for i in reversed(range(0, len(icsd_patterns_match))):

    if validation_max_NO_wyckoffs is not None:
        _, NO_wyckoffs, _, _ = icsd_sim.get_wyckoff_info(icsd_metas_match[i][0])

        if icsd_metas_match[i][0] not in NO_wyckoffs_cached.keys():
            NO_wyckoffs_cached[icsd_metas_match[i][0]] = NO_wyckoffs

    if (
        validation_max_volume is not None
        and icsd_crystals_match[i].volume > validation_max_volume
    ) or (
        validation_max_NO_wyckoffs is not None
        and NO_wyckoffs > validation_max_NO_wyckoffs
    ):
        del icsd_patterns_match[i]
        del icsd_labels_match[i]
        del icsd_variations_match[i]
        del icsd_crystals_match[i]
        del icsd_metas_match[i]

for i in reversed(range(0, len(icsd_patterns_match_corrected_labels))):

    if validation_max_NO_wyckoffs is not None:
        if icsd_metas_match_corrected_labels[i][0] not in NO_wyckoffs_cached.keys():
            _, NO_wyckoffs, _, _ = icsd_sim.get_wyckoff_info(
                icsd_metas_match_corrected_labels[i][0]
            )
        else:
            NO_wyckoffs = NO_wyckoffs_cached[icsd_metas_match_corrected_labels[i][0]]

    if (
        validation_max_volume is not None
        and icsd_crystals_match_corrected_labels[i].volume > validation_max_volume
    ) or (
        validation_max_NO_wyckoffs is not None
        and NO_wyckoffs > validation_max_NO_wyckoffs
    ):
        del icsd_patterns_match_corrected_labels[i]
        del icsd_labels_match_corrected_labels[i]
        del icsd_variations_match_corrected_labels[i]
        del icsd_crystals_match_corrected_labels[i]
        del icsd_metas_match_corrected_labels[i]

with open(out_base + "spgs.pickle", "wb") as file:
    pickle.dump(spgs, file)

with open(out_base + "icsd_data.pickle", "wb") as file:
    pickle.dump(
        (
            icsd_crystals_match,
            icsd_labels_match,
            [item[:, 0] for item in icsd_variations_match],
            icsd_metas_match,
        ),
        file,
    )

val_y_all = []
for i, label in enumerate(icsd_labels_all):
    val_y_all.extend([spgs.index(label[0])] * n_patterns_per_crystal)
val_y_all = np.array(val_y_all)

val_x_all = []
for pattern in icsd_patterns_all:
    for sub_pattern in pattern:
        val_x_all.append(sub_pattern)

val_y_match = []
for i, label in enumerate(icsd_labels_match):
    val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal)
val_y_match = np.array(val_y_match)

print(np.sum(val_y_match == 1))  # TODO: Change back
exit()

val_x_match = []
for pattern in icsd_patterns_match:
    for sub_pattern in pattern:
        val_x_match.append(sub_pattern)

val_y_match_correct_spgs = []
for i, label in enumerate(icsd_labels_match_corrected_labels):
    val_y_match_correct_spgs.extend([spgs.index(label)] * n_patterns_per_crystal)
val_y_match_correct_spgs = np.array(val_y_match_correct_spgs)

val_x_match_correct_spgs = []
for pattern in icsd_patterns_match_corrected_labels:
    for sub_pattern in pattern:
        val_x_match_correct_spgs.append(sub_pattern)

print("Numbers in validation set (that matches sim parameters):")
for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y_match==i)}")

assert not np.any(np.isnan(val_x_all))
assert not np.any(np.isnan(val_y_all))
assert not np.any(np.isnan(val_x_match))
assert not np.any(np.isnan(val_y_match))
assert not np.any(np.isnan(val_x_match_correct_spgs))
assert not np.any(np.isnan(val_y_match_correct_spgs))
assert len(val_x_all) == len(val_y_all)
assert len(val_x_match) == len(val_y_match)
assert len(val_x_match_correct_spgs) == len(val_y_match_correct_spgs)

val_x_all = np.expand_dims(val_x_all, axis=2)
val_x_match = np.expand_dims(val_x_match, axis=2)
val_x_match_correct_spgs = np.expand_dims(val_x_match_correct_spgs, axis=2)

if not local:
    ray.init(address="auto", include_dashboard=False)
    # ray.init(include_dashboard=True, num_cpus=NO_workers)
else:
    ray.init(include_dashboard=False)

print()
print(ray.cluster_resources())
print()

queue = Queue(maxsize=queue_size)  # store a maximum of `queue_size` batches


@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_with_additional(
    spgs, structures_per_spg, N, start_angle, end_angle, max_NO_elements, NO_corn_sizes
):

    patterns, labels, structures, corn_sizes = get_random_xy_patterns(
        spgs=spgs,
        structures_per_spg=structures_per_spg,
        wavelength=1.5406,  # Cu-Ka line
        # wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
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
        NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
        do_symmetry_checks=do_symmetry_checks,
        force_wyckoff_indices=True,
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
                wavelength=1.5406,  # Cu-K line
                # wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
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
                NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
                do_symmetry_checks=do_symmetry_checks,
                force_wyckoff_indices=True,
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


# For the comparison script:
# pre-store some batches to compare to the rightly / falsely classified icsd samples

random_comparison_crystals = []
random_comparison_labels = []
random_comparison_corn_sizes = []

object_refs = []
for i in range(NO_random_swipes):
    ref = batch_generator_with_additional.remote(
        spgs, 1, N, start_angle, end_angle, generation_max_NO_wyckoffs, 1
    )
    # ref = batch_generator_with_additional(
    #    spgs, 1, N, start_angle, end_angle, max_NO_elements, 1
    # )
    object_refs.append(ref)

results = ray.get(object_refs)
# results = object_refs

val_x_random = []
val_y_random = []

for result in results:
    patterns, labels, crystals, corn_sizes = result

    random_comparison_crystals.extend(crystals)
    random_comparison_labels.extend(labels)
    random_comparison_corn_sizes.extend(corn_sizes)

    val_x_random.extend(patterns)
    val_y_random.extend(labels)

val_x_random = np.expand_dims(val_x_random, axis=2)
val_y_random = np.array(val_y_random)

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
file_writer = tf.summary.create_file_writer(out_base + "metrics")

params_txt = (
    f"tag: {tag}  \n"
    f"description: {description}  \n  \n"
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
    f"do_distance_checks: {str(do_distance_checks)}  \n"
    f"do_merge_checks: {str(do_merge_checks)}  \n  \n"
    f"use_icsd_statistics: {str(use_icsd_statistics)}  \n  \n"
    f"validation_max_volume: {str(validation_max_volume)}  \n"
    f"validation_max_NO_wyckoffs: {str(validation_max_NO_wyckoffs)}  \n  \n"
    f"spgs: {str(spgs)}  \n  \n"
    f"do_symmetry_checks: {str(do_symmetry_checks)}  \n  \n"
    f"use_NO_wyckoffs_counts: {str(use_NO_wyckoffs_counts)} \n \n \n"
    f"ray cluster resources: {str(ray.cluster_resources())}"
)

with file_writer.as_default():
    tf.summary.text("Parameters", data=params_txt, step=0)

log_wait_timings = []
test_timings = []

log_ray_queue_size = []

log_all_loss = []
log_match_loss = []
log_match_correct_spgs_loss = []
log_random_loss = []

log_all_accuracy = []
log_match_accuracy = []
log_match_correct_spgs_accuracy = []
log_random_accuracy = []
log_gap_accuracy = []


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % test_every_X_epochs) == 0:

            start = time.time()

            log_ray_queue_size.append((epoch, queue.size()))

            # gather metric names form model
            metric_names = [metric.name for metric in self.model.metrics]

            scores_all = self.model.evaluate(x=val_x_all, y=val_y_all, verbose=0)
            scores_match = self.model.evaluate(x=val_x_match, y=val_y_match, verbose=0)
            scores_match_correct_spgs = self.model.evaluate(
                x=val_x_match_correct_spgs, y=val_y_match_correct_spgs, verbose=0
            )
            scores_random = self.model.evaluate(
                x=val_x_random, y=val_y_random, verbose=0
            )

            assert metric_names[0] == "loss"

            log_all_loss.append((epoch, scores_all[0]))
            log_match_loss.append((epoch, scores_match[0]))
            log_match_correct_spgs_loss.append((epoch, scores_match_correct_spgs[0]))
            log_random_loss.append((epoch, scores_random[0]))

            log_all_accuracy.append((epoch, scores_all[1]))
            log_match_accuracy.append((epoch, scores_match[1]))
            log_match_correct_spgs_accuracy.append(
                (epoch, scores_match_correct_spgs[1])
            )
            log_random_accuracy.append((epoch, scores_random[1]))

            log_gap_accuracy.append((epoch, scores_random[1] - scores_match[1]))

            test_timings.append(time.time() - start)


class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        start = time.time()
        result = queue.get()
        log_wait_timings.append(time.time() - start)
        return result


sequence = CustomSequence(batches_per_epoch)

model = build_model_park(None, N, len(spgs))

model.fit(
    x=sequence,
    epochs=NO_epochs,
    batch_size=batches_per_epoch,
    validation_data=(val_x_match, val_y_match),  # TODO: Remove this later when it fits
    validation_freq=test_every_X_epochs,
    callbacks=[tb_callback, CustomCallback()],
    verbose=verbosity,
    workers=1,
    max_queue_size=queue_size_tf,
    use_multiprocessing=False,
    steps_per_epoch=batches_per_epoch,
)

model.save(out_base + "final")

# Get predictions for val_x_match and write rightly_indices / falsely_indices:

if len(spgs) > 2:

    prediction = model.predict(val_x_match)
    prediction = np.argmax(prediction, axis=1)

elif len(spgs) == 2:

    prob_model = keras.Sequential([model, keras.layers.Activation("sigmoid")])
    prediction = np.array(prob_model.predict(val_x_match))
    prediction = prediction[:, 0]
    prediction = np.where(prediction > 0.5, 1, 0)

else:

    raise Exception("Unexpected number of spgs.")

rightly_indices_match = np.argwhere(prediction == val_y_match)[:, 0]
falsely_indices_match = np.argwhere(prediction != val_y_match)[:, 0]

with open(out_base + "rightly_falsely_icsd.pickle", "wb") as file:
    pickle.dump((rightly_indices_match, falsely_indices_match), file)


# Get predictions for val_x_random and write rightly_indices / falsely_indices:

if len(spgs) > 2:

    prediction = model.predict(val_x_random)
    prediction = np.argmax(prediction, axis=1)

elif len(spgs) == 2:

    prob_model = keras.Sequential([model, keras.layers.Activation("sigmoid")])
    prediction = np.array(prob_model.predict(val_x_random))
    prediction = prediction[:, 0]
    prediction = np.where(prediction > 0.5, 1, 0)

else:

    raise Exception("Unexpected number of spgs.")

rightly_indices_random = np.argwhere(prediction == val_y_random)[:, 0]
falsely_indices_random = np.argwhere(prediction != val_y_random)[:, 0]

with open(out_base + "rightly_falsely_random.pickle", "wb") as file:
    pickle.dump((rightly_indices_random, falsely_indices_random), file)


ray.shutdown()

# log all the stored info to tensorboard:

with file_writer.as_default():
    for i, value in enumerate(log_wait_timings):
        tf.summary.scalar("waiting time", data=value, step=i)

    for i, value in enumerate(test_timings):
        tf.summary.scalar("test time", data=value, step=i)

    for epoch, value in log_ray_queue_size:
        tf.summary.scalar("queue size", data=value, step=epoch)

    for epoch, value in log_all_loss:
        tf.summary.scalar("loss all", data=value, step=epoch)
    for epoch, value in log_match_loss:
        tf.summary.scalar("loss match", data=value, step=epoch)
    for epoch, value in log_match_correct_spgs_loss:
        tf.summary.scalar("loss match_correct_spgs", data=value, step=epoch)
    for epoch, value in log_random_loss:
        tf.summary.scalar("loss random", data=value, step=epoch)

    for epoch, value in log_all_accuracy:
        tf.summary.scalar("accuracy all", data=value, step=epoch)
    for epoch, value in log_match_accuracy:
        tf.summary.scalar("accuracy match", data=value, step=epoch)
    for epoch, value in log_match_correct_spgs_accuracy:
        tf.summary.scalar("accuracy match_correct_spgs", data=value, step=epoch)
    for epoch, value in log_random_accuracy:
        tf.summary.scalar("accuracy random", data=value, step=epoch)

    for epoch, value in log_gap_accuracy:
        tf.summary.scalar("accuracy gap", data=value, step=epoch)

# TODO: Remove again
print("Gap accuracy")
with np.printoptions(threshold=np.inf):
    print(log_gap_accuracy)
print("Accuracy match")
with np.printoptions(threshold=np.inf):
    print(log_match_accuracy)
print("Accuracy random")
with np.printoptions(threshold=np.inf):
    print(log_random_accuracy)

print("Training finished.")
print("Output dir:")
print(out_base)

if run_analysis_after_run:

    print("Starting analysis now...")

    if analysis_per_spg:
        for i, spg in enumerate(spgs):

            if np.sum(val_y_match == i) > 0:
                subprocess.call(
                    f"python compare_random_distribution.py {out_base} {tag} {spg}",
                    shell=True,
                )

    spg_str = " ".join([str(spg) for spg in spgs])
    subprocess.call(
        f"python compare_random_distribution.py {out_base} {tag} {spg_str}",
        shell=True,
    )
