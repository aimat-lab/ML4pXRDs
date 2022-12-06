import tensorflow.keras as keras
from tools.simulation.simulation_core import get_random_xy_patterns
from tools.simulation.simulation_core import mix_patterns_add_background
from tools.generation.random_simulation_utils import load_dataset_info
import numpy as np
from models import (
    build_model_park,
    build_model_park_2_layer_CNN,
    build_model_park_gigantic_size,
    build_model_park_gigantic_size_more_dense,
    build_model_park_gigantic_size_more_dense_bn,
    build_model_park_medium_size,
    build_model_park_huge_size,
    build_model_park_original_spg,
    build_model_park_tiny_size,
    build_model_resnet_i,
)
from utils.distributed_utils import map_to_remote
import os
from sklearn.utils import shuffle
from tools.simulation.simulation import Simulation
import ray
from ray.util.queue import Queue
import pickle
import tensorflow as tf
import sys
from datetime import datetime
import time
import subprocess
from sklearn.metrics import classification_report
from pyxtal.symmetry import Group
import gc
import psutil
from tools.generation.structure_generation import randomize
from tools.simulation.simulation_core import get_xy_patterns
import random
import contextlib
from training.utils.AdamWarmup import AdamWarmup

#######################################################################################################################
##### Configuration of the training script

tag = "name_of_current_run"
description = "Description of current run"

run_analysis_after_training = (
    True  # run analysis script "compare_random_distribution.py" after training
)
analysis_per_spg = False  # run the analysis script separately for each spg (and once for all spgs together)

# If this setting is True, the training uses the crystals from the statistics
# dataset to form the training dataset. No synthetic crystals will be generated
# and no on-the-fly simulation will take place. It is advised to use this
# setting in combination with head_only, since much less computing ressources
# are required.
use_icsd_structures_directly = False

test_every_X_epochs = 1  # compute validation accuracies every X epochs
# Maximum number of samples to test on for each of the constructed validation
# datasets.
max_NO_samples_to_test_on = 10000
# The following setting only applies for training on synthetic crystals. When
# training on ICSD crystals directly, the whole dataset is used for each epoch.
batches_per_epoch = 150 * 6
NO_epochs = 1000

# How many structures to generate per spg. Only applies for training using synthetic crystals.
structures_per_spg = 1
# How many different crystallite sizes to use for each generated crystal. Only applies for training using synthetic crystals.
NO_corn_sizes = 1

# Queue size of the ray queue (used to fetch generated patterns from worker nodes)
queue_size = 100
# Queue size of the tensorflow queue of batches used while training
queue_size_tf = 50

# Used to generate the validation dataset of patterns based on synthetic crystals.
NO_random_validation_samples_per_spg = 100

# Maximum volume and number of atoms in asymmetric unit for synthetic crystals.
generation_max_volume = 7000
generation_max_NO_wyckoffs = 100

# Perform check of spg after generating a crystal using software `spglib`.
do_symmetry_checks = True

# Whether or not to validate also on patterns simulated from the statistics
# dataset. This setting cannot be used together with
# `use_icsd_structures_directly`, since here the statistics dataset is already
# used for training.
use_statistics_dataset_as_validation = False

# The following setting can be used to analyze the difference in accuracy
# between training (using synthetic crystals) and patterns simulated from the
# ICSD (see the discussion in our paper). Three additional validation datasets
# will be generated, which are all based on the ICSD validation dataset. In the
# first one, the ICSD coordinates are replaced with uniformly sampled
# coordinates (using the same approach that we use to generate the synthetic
# crystals). In the second one, the lattice parameters are replaced (lattice
# parameters are sampled using the KDE as described in the paper). In the third
# one, both coordinates and the lattice parameters are replaced / resampled.
generate_randomized_validation_datasets = False
randomization_step = 3  # Only use every n'th sample for the randomization process

# This only applies only to the models that support dropout, especially those
# originating from Park et al. (2020)
use_dropout = False
learning_rate = 0.0001

optimizer = "Adam"
# Use group normalization instead of batch normalization. This setting only
# applies for the ResNet models, where batch normalization was observed to
# result in unstable training.
use_group_norm = True
batchnorm_momentum = 0.0  # only used by ResNet (if use_group_norm is False)

# The denseness factors are sampled from a KDE. If the following setting is
# True, the denseness factor is conditioned on the sum of atomic volumes.
use_conditional_denseness_factor_sampling = True
# If this is False, the lattice parameters are sampled using the heuristic
# implementation of the python library `pyxtal`.
sample_lattice_paras_from_kde = True

# There might be multiple crystallite sizes for each ICSD crystal found in the
# dataset. This setting controls how many of them should be used / loaded for
# training and testing. If the setting is "None", all patterns are used.
load_only_N_patterns_each_test = 1
load_only_N_patterns_each_statistics = 1

preprocess_patterns_sqrt = False  # Apply the sqrt function to the patterns as a preprocessing step (see Zaloga et al. 2020).

# Verbosity setting as passed to fit function of tf
verbosity_tf = 2
generator_verbose = False

# Use more than one GPU on the head node
use_distributed_strategy = True

# Instead of following the distribution of spg labels found in the ICSD, uniformly distribute them
uniformly_distributed = False

# Shuffle the statistics (train) dataset with the test dataset. This option can
# be used to judge the impact that the structure type based split has on the
# test performance.
shuffle_test_match_train_match = False

use_pretrained_model = False  # Make it possible to resume from a previous training run
pretrained_model_path = "/path/to/model"

# This option can be used to run training locally (with restricted computing resources)
# If True, only 8 cores are used.
local = False

# All spg labels to train on; Keep in mind that spgs that do not have more than
# 50 crystals in the statistics dataset will be excluded
spgs = list(range(1, 231))

# Path to the presimulated patterns used for testing
path_to_patterns = "../dataset_simulations/patterns/icsd_vecsei/"  # TODO: Update

# Path to the ICSD directory that contains the "ICSD_data_from_API.csv" file
# and the "cif" directory (which contains all the ICSD cif files)
path_to_icsd_directory_local = os.path.expanduser("~/Dokumente/Big_Files/ICSD/")
path_to_icsd_directory_cluster = os.path.expanduser("~/Databases/ICSD/")

#######################################################################################################################
#######################################################################################################################

# Allow execution from bash (script), where the timestamp is passed as an arg
if len(sys.argv) > 1:
    date_time = sys.argv[1]
    out_base = "classifier_spgs/" + date_time + "/"
else:
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_base = "classifier_spgs/" + date_time + "_" + tag + "/"

print("Processing tag", tag)
print("Training files are in", out_base)

# Allow head-only execution, where only one head computing node (with a GPU)
# without additional computing nodes is used
if len(sys.argv) > 2 and sys.argv[2] == "head_only":
    head_only = True
    print("Running in head-only mode.", flush=True)
else:
    head_only = False

os.system("mkdir -p " + out_base)
os.system("mkdir -p " + out_base + "tuner_tb")
os.system("touch " + out_base + tag)

if not len(sys.argv) > 3:
    if not head_only:
        # By default, we use 2*128 cores on separate compute nodes + 28 cores on the head node
        NO_workers = 1 * 128 + 1 * 128 + 28
    else:
        NO_workers = 30
else:
    NO_workers = int(sys.argv[3])

if local:
    NO_workers = 8
    verbosity_tf = 1  # make it verbose
    generator_verbose = True
    NO_random_validation_samples_per_spg = 5
    randomization_step = 20
    use_distributed_strategy = False

git_revision_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
)

# 2-theta-range and length patterns (same as used by Vecsei et al. 2019)
start_angle, end_angle, N = 5, 90, 8501
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}, N: {N}", flush=True)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start loading dataset info.",
    flush=True,
)

(
    probability_per_spg_per_element,  # To choose the elements
    probability_per_spg_per_element_per_wyckoff,  # TODO: Rename this
    NO_wyckoffs_prob_per_spg,  # TODO: Remove
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    denseness_factors_density_per_spg,
    kde_per_spg,  # TODO: Remove
    all_data_per_spg_tmp,  # TODO: Remove
    denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type,
    per_element,  # TODO: Remove
    represented_spgs,  # spgs that are well-represented (>= 50 crystals in statistics dataset)
    (
        statistics_metas,  # IDs in ICSD database
        statistics_labels,
        statistics_crystals,
        statistics_match_metas,
        statistics_match_labels,
        test_metas,
        test_labels,
        test_crystals,
        corrected_labels,
        test_match_metas,
        test_match_pure_metas,
    ),
) = load_dataset_info()

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done loading dataset info.",
    flush=True,
)

if not use_conditional_denseness_factor_sampling:
    denseness_factors_conditional_sampler_seeds_per_spg = None

if not sample_lattice_paras_from_kde:
    lattice_paras_density_per_lattice_type = None

for i in reversed(range(0, len(spgs))):
    if spgs[i] not in represented_spgs:
        print(f"Excluded spg {spgs[i]} (not enough statistics).")
        del spgs[i]

batch_size = NO_corn_sizes * structures_per_spg * len(spgs)

print("len(spgs): ", len(spgs))
print("batch_size: ", batch_size)

if not uniformly_distributed:
    # Calculate the probability to pick each space group when generating the
    # synthetic crystals
    probability_per_spg = {}
    for i, label in enumerate(statistics_match_labels):
        if label[0] in spgs:
            if label[0] in probability_per_spg.keys():
                probability_per_spg[label[0]] += 1
            else:
                probability_per_spg[label[0]] = 1
    total = np.sum(list(probability_per_spg.values()))
    for key in probability_per_spg.keys():
        probability_per_spg[key] /= total
else:
    probability_per_spg = None

ray.init(
    address="localhost:6379" if not local else None,
    include_dashboard=False,  # Can be used to have a nice overview of the currently used hardware ressources
)

print("Ray cluster resources:")
print(ray.cluster_resources())

jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    path_to_icsd_directory = path_to_icsd_directory_cluster
else:
    path_to_icsd_directory = path_to_icsd_directory_local

icsd_sim_test = Simulation(
    os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
    os.path.join(path_to_icsd_directory, "cif/"),
)
icsd_sim_test.output_dir = path_to_patterns

##### Prepare test datasets

test_metas_flat = [item[0] for item in test_metas]
test_match_metas_flat = [item[0] for item in test_match_metas]
test_match_pure_metas_flat = [item[0] for item in test_match_pure_metas]

metas_to_load_test = []
for i, meta in enumerate(test_metas_flat):
    if test_labels[i][0] in spgs or corrected_labels[i] in spgs:
        metas_to_load_test.append(meta)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start loading patterns for test dataset.",
    flush=True,
)

icsd_sim_test.load(
    load_only_N_patterns_each=load_only_N_patterns_each_test,
    stop=1
    if local
    else None,  # Only load a fraction of all patterns / crystals if running locally
    metas_to_load=metas_to_load_test,  # Only load the patterns with the ICSD ids from the test dataset
)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done loading patterns for test dataset.",
    flush=True,
)

n_patterns_per_crystal_test = len(icsd_sim_test.sim_patterns[0])

icsd_patterns_all = []
icsd_labels_all = []
icsd_variations_all = []
icsd_crystals_all = []
icsd_metas_all = []

# "match" test dataset contains all patterns with less than 100 atoms in the asymmetric unit
# and volume smaller than 7000 A^3
icsd_patterns_match = []
icsd_labels_match = []
icsd_variations_match = []
icsd_crystals_match = []
icsd_metas_match = []

# This validation dataset has "corrected labels", meaning that instead of using
# the spg label as provided by the ICSD, we use the spg label as output by
# `spglib`
icsd_patterns_match_corrected_labels = []
icsd_labels_match_corrected_labels = []
icsd_variations_match_corrected_labels = []
icsd_crystals_match_corrected_labels = []
icsd_metas_match_corrected_labels = []

# This validation dataset only contains `pure` crystals, meaning crystals that
# do not have partial occupancies
icsd_patterns_match_corrected_labels_pure = []
icsd_labels_match_corrected_labels_pure = []
icsd_variations_match_corrected_labels_pure = []
icsd_crystals_match_corrected_labels_pure = []
icsd_metas_match_corrected_labels_pure = []

for i in range(len(icsd_sim_test.sim_crystals)):

    if icsd_sim_test.sim_metas[i][0] in test_metas_flat:
        if icsd_sim_test.sim_labels[i][0] in spgs:
            icsd_patterns_all.append(icsd_sim_test.sim_patterns[i])
            icsd_labels_all.append(icsd_sim_test.sim_labels[i])
            icsd_variations_all.append(icsd_sim_test.sim_variations[i])
            icsd_crystals_all.append(
                test_crystals[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # uses the converted structure (conventional cell)
            icsd_metas_all.append(icsd_sim_test.sim_metas[i])
    else:
        raise Exception("There is a mismatch somewhere.")

    if icsd_sim_test.sim_metas[i][0] in test_match_metas_flat:

        if icsd_sim_test.sim_labels[i][0] in spgs:
            icsd_patterns_match.append(icsd_sim_test.sim_patterns[i])
            icsd_labels_match.append(icsd_sim_test.sim_labels[i])
            icsd_variations_match.append(icsd_sim_test.sim_variations[i])
            icsd_crystals_match.append(
                test_crystals[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # uses the converted structure (conventional cell)
            icsd_metas_match.append(icsd_sim_test.sim_metas[i])

        if (
            corrected_labels[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            in spgs  # Also excludes "None" (if the corrected label could not be determined)
        ):
            icsd_patterns_match_corrected_labels.append(icsd_sim_test.sim_patterns[i])
            icsd_labels_match_corrected_labels.append(
                corrected_labels[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # Use the corrected label
            icsd_variations_match_corrected_labels.append(
                icsd_sim_test.sim_variations[i]
            )
            icsd_crystals_match_corrected_labels.append(
                test_crystals[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # use the converted structure (conventional cell)
            icsd_metas_match_corrected_labels.append(icsd_sim_test.sim_metas[i])

    if icsd_sim_test.sim_metas[i][0] in test_match_pure_metas_flat:
        if (
            corrected_labels[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            in spgs  # Also excludes "None" (if the corrected label could not be determined)
        ):
            icsd_patterns_match_corrected_labels_pure.append(
                icsd_sim_test.sim_patterns[i]
            )
            icsd_labels_match_corrected_labels_pure.append(
                corrected_labels[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # Use the corrected label
            icsd_variations_match_corrected_labels_pure.append(
                icsd_sim_test.sim_variations[i]
            )
            icsd_crystals_match_corrected_labels_pure.append(
                test_crystals[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            )  # use the converted structure (conventional cell)
            icsd_metas_match_corrected_labels_pure.append(icsd_sim_test.sim_metas[i])

# Save the spgs that were used (for the analysis script)
with open(out_base + "spgs.pickle", "wb") as file:
    pickle.dump(spgs, file)

# This (remote) function is later only used to simulate the patterns of the randomized ICSD crystals
# (with replaced coordinates of lattice parameters)
@ray.remote(num_cpus=1, num_gpus=0)
def get_xy_pattern_wrapper(
    crystal,
):
    xs = np.linspace(start_angle, end_angle, N)
    patterns, corn_sizes = get_xy_patterns(
        crystal,
        wavelength=1.5406,
        xs=xs,
        NO_corn_sizes=1,
        two_theta_range=(start_angle, end_angle),
        do_print=False,
        return_corn_sizes=True,
        return_angles_intensities=False,
        return_max_unscaled_intensity_angle=False,
    )
    return patterns[0], corn_sizes[0]


##### Generate match (corrected spgs) validation set with randomized coordinates and reference:

if generate_randomized_validation_datasets:

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start generating randomized dataset (randomized coordinates).",
        flush=True,
    )

    randomized_coords_crystals, reference_crystals, labels = randomize(
        icsd_crystals_match_corrected_labels[::randomization_step],
        randomize_coordinates=True,
        randomize_lattice=False,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )
    # We also get reference crystals, where nothing has been randomized.
    # They differ slightly from the input crystals, because partial occupancies have been ignored.
    # (pyxtal currently doesn't support partial occupancies)

    errors_counter = 0
    for i in reversed(range(len(labels))):
        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[::randomization_step][i]:
                errors_counter += 1

                del labels[i]
                del randomized_coords_crystals[i]
                del reference_crystals[i]

    # Because `pyxtal` uses slightly different parameters for `spglib`, the
    # obtained spg label from `pyxtal` and from our code (when obtaining the
    # corrected labels) differ in rare cases
    print(f"{errors_counter} of {len(labels)} mismatched (different tolerances)")

    randomized_coords_crystals = [
        item for item in randomized_coords_crystals if item is not None
    ]
    reference_crystals = [item for item in reference_crystals if item is not None]
    labels = [item for item in labels if item is not None]

    # Simulate patterns on ray cluster:
    scheduler_fn = lambda crystal: get_xy_pattern_wrapper.remote(crystal)
    results = map_to_remote(
        scheduler_fn=scheduler_fn,
        inputs=randomized_coords_crystals,
        NO_workers=NO_workers,
    )
    randomized_coords_patterns = [result[0] for result in results]
    randomized_coords_corn_sizes = [result[1] for result in results]

    results = map_to_remote(
        scheduler_fn=scheduler_fn,
        inputs=reference_crystals,
        NO_workers=NO_workers,
    )
    reference_patterns = [result[0] for result in results]
    reference_corn_sizes = [result[1] for result in results]

    randomized_coords_labels = []
    for i in range(0, len(labels)):
        randomized_coords_labels.append(spgs.index(labels[i]))

    # For further analysis, save the generated crystals:
    with open(out_base + "randomized_coords_data.pickle", "wb") as file:
        pickle.dump(
            (
                randomized_coords_crystals,
                randomized_coords_labels,
                randomized_coords_corn_sizes,
                reference_crystals,
                reference_corn_sizes,
            ),
            file,
        )

##### Generate match (corrected spgs) validation set with randomized lattice parameters:

if generate_randomized_validation_datasets:

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start generating randomized dataset (randomized lattice).",
        flush=True,
    )

    randomized_lattice_crystals, _, labels = randomize(
        icsd_crystals_match_corrected_labels[::randomization_step],
        randomize_coordinates=False,
        randomize_lattice=True,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[::randomization_step][i]:
                errors_counter += 1

                del labels[i]
                del randomized_lattice_crystals[i]

    print(f"{errors_counter} of {len(labels)} mismatched (different tolerances)")

    randomized_lattice_crystals = [
        item for item in randomized_lattice_crystals if item is not None
    ]
    labels = [item for item in labels if item is not None]

    # Simulate on ray cluster:
    scheduler_fn = lambda crystal: get_xy_pattern_wrapper.remote(crystal)
    results = map_to_remote(
        scheduler_fn=scheduler_fn,
        inputs=randomized_lattice_crystals,
        NO_workers=NO_workers,
    )
    randomized_lattice_patterns = [result[0] for result in results]
    randomized_lattice_corn_sizes = [result[1] for result in results]

    randomized_lattice_labels = []
    for i in range(0, len(labels)):
        randomized_lattice_labels.append(spgs.index(labels[i]))

    with open(out_base + "randomized_lattice_data.pickle", "wb") as file:
        pickle.dump(
            (
                randomized_lattice_crystals,
                randomized_lattice_labels,
                randomized_lattice_corn_sizes,
            ),
            file,
        )

##### Generate match (corrected spgs) validation set with randomized lattice and coords:

if generate_randomized_validation_datasets:

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start generating randomized dataset (randomized coordinates and lattice).",
        flush=True,
    )

    randomized_both_crystals, _, labels = randomize(
        icsd_crystals_match_corrected_labels[::randomization_step],
        randomize_coordinates=True,
        randomize_lattice=True,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[::randomization_step][i]:
                errors_counter += 1

                del labels[i]
                del randomized_both_crystals[i]

    print(f"{errors_counter} of {len(labels)} mismatched (different tolerances)")

    randomized_both_crystals = [
        item for item in randomized_both_crystals if item is not None
    ]
    labels = [item for item in labels if item is not None]

    # Simulate on ray cluster:
    scheduler_fn = lambda crystal: get_xy_pattern_wrapper.remote(crystal)
    results = map_to_remote(
        scheduler_fn=scheduler_fn,
        inputs=randomized_both_crystals,
        NO_workers=NO_workers,
    )
    randomized_both_patterns = [result[0] for result in results]
    randomized_both_corn_sizes = [result[1] for result in results]

    randomized_both_labels = []
    for i in range(0, len(labels)):
        randomized_both_labels.append(spgs.index(labels[i]))

    with open(out_base + "randomized_both_data.pickle", "wb") as file:
        pickle.dump(
            (
                randomized_both_crystals,
                randomized_both_labels,
                randomized_both_corn_sizes,
            ),
            file,
        )

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done generating randomized datasets.",
        flush=True,
    )

#####

val_y_all = []
for i, label in enumerate(icsd_labels_all):
    val_y_all.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
val_y_all = np.array(val_y_all)

val_x_all = []
for pattern in icsd_patterns_all:
    for sub_pattern in pattern:
        val_x_all.append(sub_pattern)

val_y_match_correct_spgs = []
for i, label in enumerate(icsd_labels_match_corrected_labels):
    val_y_match_correct_spgs.extend([spgs.index(label)] * n_patterns_per_crystal_test)
val_y_match_correct_spgs = np.array(val_y_match_correct_spgs)

val_x_match_correct_spgs = []
for pattern in icsd_patterns_match_corrected_labels:
    for sub_pattern in pattern:
        val_x_match_correct_spgs.append(sub_pattern)

val_y_match_correct_spgs_pure = []
for i, label in enumerate(icsd_labels_match_corrected_labels_pure):
    val_y_match_correct_spgs_pure.extend(
        [spgs.index(label)] * n_patterns_per_crystal_test
    )
val_y_match_correct_spgs_pure = np.array(val_y_match_correct_spgs_pure)

val_x_match_correct_spgs_pure = []
for pattern in icsd_patterns_match_corrected_labels_pure:
    for sub_pattern in pattern:
        val_x_match_correct_spgs_pure.append(sub_pattern)

if generate_randomized_validation_datasets:

    val_y_randomized_coords = []
    for i, label in enumerate(randomized_coords_labels):
        val_y_randomized_coords.append(label)
    val_y_randomized_coords = np.array(val_y_randomized_coords)

    val_x_randomized_coords = []
    for pattern in randomized_coords_patterns:
        val_x_randomized_coords.append(pattern)

    val_y_randomized_ref = []
    for i, label in enumerate(randomized_coords_labels):
        val_y_randomized_ref.append(label)
    val_y_randomized_ref = np.array(val_y_randomized_ref)

    val_x_randomized_ref = []
    for pattern in reference_patterns:
        val_x_randomized_ref.append(pattern)

    val_y_randomized_lattice = []
    for i, label in enumerate(randomized_lattice_labels):
        val_y_randomized_lattice.append(label)
    val_y_randomized_lattice = np.array(val_y_randomized_lattice)

    val_x_randomized_lattice = []
    for pattern in randomized_lattice_patterns:
        val_x_randomized_lattice.append(pattern)

    val_y_randomized_both = []
    for i, label in enumerate(randomized_both_labels):
        val_y_randomized_both.append(label)
    val_y_randomized_both = np.array(val_y_randomized_both)

    val_x_randomized_both = []
    for pattern in randomized_both_patterns:
        val_x_randomized_both.append(pattern)

assert not np.any(np.isnan(val_x_all))
assert not np.any(np.isnan(val_y_all))
assert not np.any(np.isnan(val_x_match_correct_spgs))
assert not np.any(np.isnan(val_y_match_correct_spgs))
assert not np.any(np.isnan(val_x_match_correct_spgs_pure))
assert not np.any(np.isnan(val_y_match_correct_spgs_pure))
assert len(val_x_all) == len(val_y_all)
assert len(val_x_match_correct_spgs) == len(val_y_match_correct_spgs)
assert len(val_x_match_correct_spgs_pure) == len(val_y_match_correct_spgs_pure)

if generate_randomized_validation_datasets:
    assert len(val_x_randomized_coords) == len(val_y_randomized_coords)
    assert len(val_x_randomized_ref) == len(val_y_randomized_ref)
    assert len(val_x_randomized_lattice) == len(val_y_randomized_lattice)
    assert len(val_x_randomized_both) == len(val_y_randomized_both)

queue = Queue(
    maxsize=queue_size
)  # store a maximum of `queue_size` batches in the ray queue


def auto_garbage_collect(pct=93.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 93% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 93%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


# This function is used to generate the validation dataset from synthetic crystals
@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_with_additional(
    spgs,
    structures_per_spg,
    N,
    start_angle,
    end_angle,
    max_NO_elements,
    NO_corn_sizes,
):

    patterns, labels, structures, corn_sizes = get_random_xy_patterns(
        spgs=spgs,
        structures_per_spg=structures_per_spg,
        wavelength=1.5406,  # Cu-Ka line
        N=N,
        NO_corn_sizes=NO_corn_sizes,
        two_theta_range=(start_angle, end_angle),
        max_NO_elements=max_NO_elements,
        do_print=False,
        return_additional=True,
        do_distance_checks=do_distance_checks,
        do_merge_checks=do_merge_checks,
        use_icsd_statistics=use_icsd_statistics,
        probability_per_spg_per_element=probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
        max_volume=generation_max_volume,
        NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
        do_symmetry_checks=do_symmetry_checks,
        force_wyckoff_indices=True,
        use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions,
        NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg=denseness_factors_density_per_spg,
        kde_per_spg=kde_per_spg,
        # all_data_per_spg=all_data_per_spg_worker,
        all_data_per_spg=all_data_per_spg,
        use_coordinates_directly=use_coordinates_directly,
        use_lattice_paras_directly=use_lattice_paras_directly,
        denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
        per_element=per_element,
        verbosity=1,  # Show everything here
        probability_per_spg=probability_per_spg,
        add_background_and_noise=add_background_and_noise
        if not mix_impurities
        else False,
        use_vecsei_bg_noise=use_vecsei_bg_noise if not mix_impurities else False,
        caglioti_broadening=caglioti_broadening,
    )

    # Set the label to the right index:
    for i in range(0, len(labels)):
        labels[i] = spgs.index(labels[i])

    patterns = np.array(patterns)
    labels = np.array(labels)

    auto_garbage_collect()

    return patterns, labels, structures, corn_sizes


# This function is used for the on-the-fly generation of training data:
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

    # Pass in the group object to speed up the process (do not construct it from scratch every time)
    group_object_per_spg = {}
    for spg in spgs:
        group_object_per_spg[spg] = Group(spg, dim=3)

    while True:
        try:
            patterns, labels = get_random_xy_patterns(
                spgs=spgs,
                structures_per_spg=structures_per_spg,
                wavelength=1.5406,  # Cu-K line
                N=N,
                NO_corn_sizes=NO_corn_sizes,
                two_theta_range=(start_angle, end_angle),
                max_NO_elements=max_NO_elements,
                do_print=False,
                do_distance_checks=do_distance_checks,
                do_merge_checks=do_merge_checks,
                use_icsd_statistics=use_icsd_statistics,
                probability_per_spg_per_element=probability_per_spg_per_element,
                probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
                max_volume=generation_max_volume,
                NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
                do_symmetry_checks=do_symmetry_checks,
                force_wyckoff_indices=True,
                use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions,
                NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
                denseness_factors_density_per_spg=denseness_factors_density_per_spg,
                kde_per_spg=kde_per_spg,
                all_data_per_spg=all_data_per_spg,
                use_coordinates_directly=use_coordinates_directly,
                use_lattice_paras_directly=use_lattice_paras_directly,
                group_object_per_spg=group_object_per_spg,
                denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                per_element=per_element,
                verbosity=verbosity_generator,
                probability_per_spg=probability_per_spg,
                add_background_and_noise=add_background_and_noise
                if not mix_impurities
                else False,
                use_vecsei_bg_noise=use_vecsei_bg_noise
                if not mix_impurities
                else False,
                caglioti_broadening=caglioti_broadening,
            )

            patterns, labels = shuffle(patterns, labels)

            # Use the index as target:
            for i in range(0, len(labels)):
                labels[i] = spgs.index(labels[i])

            patterns = np.array(patterns)

            if preprocess_patterns_sqrt:
                patterns = np.sqrt(patterns)

            patterns = np.expand_dims(patterns, axis=2)

            labels = np.array(labels)

            auto_garbage_collect()

            queue.put((patterns, labels))  # blocks if queue is full, which is good

        except Exception as ex:

            print("Error occurred in worker:")
            print(ex)
            print(
                type(ex).__name__,
                __file__,
                ex.__traceback__.tb_lineno,
            )


# Generate validation dataset based on synthetic crystals

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start generating validation random structures.",
    flush=True,
)

# These three lists will be pickled to a file to later use them in the analysis script
random_comparison_crystals = []
random_comparison_labels = []
random_comparison_corn_sizes = []

scheduler_fn = lambda input: batch_generator_with_additional.remote(
    spgs,
    1,
    N,
    start_angle,
    end_angle,
    generation_max_NO_wyckoffs,
    1,
)
results = map_to_remote(
    scheduler_fn=scheduler_fn,
    inputs=range(NO_random_validation_samples_per_spg),
    NO_workers=NO_workers,
)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done generating validation random structures.",
    flush=True,
)

print("Sizes of validation sets:")
print(f"all: {len(icsd_labels_all)} * {n_patterns_per_crystal_test}")
print(f"match: {len(icsd_labels_match)} * {n_patterns_per_crystal_test}")
print(
    f"match_correct_spgs: {len(icsd_labels_match_corrected_labels)} * {n_patterns_per_crystal_test}"
)
print(
    f"match_correct_spgs_pure: {len(icsd_labels_match_corrected_labels_pure)} * {n_patterns_per_crystal_test}"
)
print(f"random: {len(results)}")

val_x_random = []
val_y_random = []

for result in results:
    patterns, labels, crystals, corn_sizes = result

    random_comparison_crystals.extend(crystals)
    random_comparison_labels.extend(labels)
    random_comparison_corn_sizes.extend(corn_sizes)

    val_x_random.extend(patterns)
    val_y_random.extend(labels)

val_y_random = np.array(val_y_random)

if preprocess_patterns_sqrt:
    val_x_random = np.sqrt(val_x_random)
    val_x_all = np.sqrt(val_x_all)
    val_x_match_correct_spgs = np.sqrt(val_x_match_correct_spgs)
    val_x_match_correct_spgs_pure = np.sqrt(val_x_match_correct_spgs_pure)
    if generate_randomized_validation_datasets:
        val_x_randomized_coords = np.sqrt(val_x_randomized_coords)
        val_x_randomized_ref = np.sqrt(val_x_randomized_ref)
        val_x_randomized_lattice = np.sqrt(val_x_randomized_lattice)
        val_x_randomized_both = np.sqrt(val_x_randomized_both)

val_x_all = np.expand_dims(val_x_all, axis=2)
val_x_match_correct_spgs = np.expand_dims(val_x_match_correct_spgs, axis=2)
val_x_match_correct_spgs_pure = np.expand_dims(val_x_match_correct_spgs_pure, axis=2)
val_x_random = np.expand_dims(val_x_random, axis=2)

if generate_randomized_validation_datasets:
    val_x_randomized_coords = np.expand_dims(val_x_randomized_coords, axis=2)
    val_x_randomized_ref = np.expand_dims(val_x_randomized_ref, axis=2)
    val_x_randomized_lattice = np.expand_dims(val_x_randomized_lattice, axis=2)
    val_x_randomized_both = np.expand_dims(val_x_randomized_both, axis=2)

# Store them for usage in the analysis script
with open(out_base + "random_data.pickle", "wb") as file:
    pickle.dump(
        (
            random_comparison_crystals,
            random_comparison_labels,
            random_comparison_corn_sizes,
        ),
        file,
    )

#####

# If use_icsd_structures_directly, then create the training dataset
# If use_statistics_dataset_as_validation, then create validation dataset
# (both are based on the statistics dataset, it is just used for different purposes)
if use_icsd_structures_directly or use_statistics_dataset_as_validation:

    if use_icsd_structures_directly and use_statistics_dataset_as_validation:
        raise Exception(
            "Cannot train and validate on statistics dataset at the same time."
        )

    if jobid is not None and jobid != "":
        path_to_icsd_directory = path_to_icsd_directory_cluster
    else:  # local
        path_to_icsd_directory = path_to_icsd_directory_local

    icsd_sim_statistics = Simulation(
        os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
        os.path.join(path_to_icsd_directory, "cif/"),
    )
    icsd_sim_statistics.output_dir = path_to_patterns

    statistics_match_metas_flat = [item[0] for item in statistics_match_metas]

    # Make loading faster if only a few spgs are used for training:
    for i in reversed(range(len(statistics_match_metas_flat))):
        if not statistics_match_labels[i][0] in spgs:
            del statistics_match_labels[i]
            del statistics_match_metas[i]
            del statistics_match_metas_flat[i]

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start loading patterns for statistics / training.",
        flush=True,
    )

    icsd_sim_statistics.load(
        load_only_N_patterns_each=load_only_N_patterns_each_test
        if use_statistics_dataset_as_validation
        else load_only_N_patterns_each_statistics,
        metas_to_load=statistics_match_metas_flat,
        stop=1 if local else None,
    )

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done loading patterns for statistics / training.",
        flush=True,
    )

    statistics_icsd_patterns_match = []
    statistics_icsd_labels_match = []
    statistics_icsd_variations_match = []
    statistics_icsd_crystals_match = []
    statistics_icsd_metas_match = []

    for i in range(len(icsd_sim_statistics.sim_crystals)):
        if (
            icsd_sim_statistics.sim_metas[i][0] in statistics_match_metas_flat
            and icsd_sim_statistics.sim_labels[i][0] in spgs
        ):

            statistics_icsd_patterns_match.append(icsd_sim_statistics.sim_patterns[i])
            statistics_icsd_labels_match.append(icsd_sim_statistics.sim_labels[i])
            statistics_icsd_variations_match.append(
                icsd_sim_statistics.sim_variations[i]
            )
            statistics_icsd_crystals_match.append(
                statistics_crystals[
                    statistics_match_metas_flat.index(
                        icsd_sim_statistics.sim_metas[i][0]
                    )
                ]
            )  # Use the converted structure (conventional cell)
            statistics_icsd_metas_match.append(icsd_sim_statistics.sim_metas[i])

    n_patterns_per_crystal_statistics = len(icsd_sim_statistics.sim_patterns[0])

    #####

    if not shuffle_test_match_train_match:

        val_y_match = []
        for i, label in enumerate(icsd_labels_match):
            val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)

        val_x_match = []
        for pattern in icsd_patterns_match:
            for sub_pattern in pattern:
                val_x_match.append(sub_pattern)

        statistics_y_match = []
        for i, label in enumerate(statistics_icsd_labels_match):
            statistics_y_match.extend(
                [spgs.index(label[0])] * n_patterns_per_crystal_statistics
            )

        statistics_x_match = []
        for pattern in statistics_icsd_patterns_match:
            for sub_pattern in pattern:
                statistics_x_match.append(sub_pattern)

    else:

        # Shuffle the statistics and test (match) dataset to make a comparison of the two possible splits possible:
        # Random vs. structure type based

        val_y_match = []
        val_x_match = []
        statistics_y_match = []
        statistics_x_match = []

        total = (
            len(icsd_labels_match) * n_patterns_per_crystal_test
            + len(statistics_icsd_labels_match) * n_patterns_per_crystal_statistics
        )
        # Probability to put a pattern into the test dataset:
        probability_test = len(icsd_labels_match) * n_patterns_per_crystal_test / total

        for i in range(len(icsd_labels_match)):

            label = icsd_labels_match[i]
            pattern = icsd_patterns_match[i]

            if random.random() < probability_test:
                for sub_pattern in pattern:
                    val_x_match.append(sub_pattern)
                val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
            else:
                for sub_pattern in pattern:
                    statistics_x_match.append(sub_pattern)
                statistics_y_match.extend(
                    [spgs.index(label[0])] * n_patterns_per_crystal_test
                )

        for i in range(len(statistics_icsd_labels_match)):
            label = statistics_icsd_labels_match[i]
            pattern = statistics_icsd_patterns_match[i]

            if random.random() < probability_test:
                for sub_pattern in pattern:
                    val_x_match.append(sub_pattern)
                val_y_match.extend(
                    [spgs.index(label[0])] * n_patterns_per_crystal_statistics
                )
            else:
                for sub_pattern in pattern:
                    statistics_x_match.append(sub_pattern)
                statistics_y_match.extend(
                    [spgs.index(label[0])] * n_patterns_per_crystal_statistics
                )

    statistics_x_match, statistics_y_match = shuffle(
        statistics_x_match, statistics_y_match
    )

    statistics_y_match = np.array(statistics_y_match)

    if preprocess_patterns_sqrt:
        statistics_x_match = np.sqrt(statistics_x_match)

    statistics_x_match = np.expand_dims(statistics_x_match, axis=2)

    print(
        "Size of statistics / training dataset: ", statistics_x_match.shape, flush=True
    )

else:  # if statistics dataset is not used, just construct the match test dataset

    val_y_match = []
    for i, label in enumerate(icsd_labels_match):
        val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)

    val_x_match = []
    for pattern in icsd_patterns_match:
        for sub_pattern in pattern:
            val_x_match.append(sub_pattern)

val_y_match = np.array(val_y_match)
print("Numbers in validation set (that matches sim parameters):")
for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y_match==i)}")

assert not np.any(np.isnan(val_x_match))
assert not np.any(np.isnan(val_y_match))
assert len(val_x_match) == len(val_y_match)

if preprocess_patterns_sqrt:
    val_x_match = np.sqrt(val_x_match)

val_x_match = np.expand_dims(val_x_match, axis=2)

print("Size of match test dataset: ", val_x_match.shape, flush=True)

#####

# Start worker tasks for on-the-fly generation of crystals + simulation
if not use_icsd_structures_directly:
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
    f"git hash: {git_revision_hash}  \n  \n"
    f"batches_per_epoch: {batches_per_epoch}  \n"
    f"NO_epochs: {NO_epochs}  \n"
    f"structures_per_spg: {structures_per_spg}  \n"
    f"NO_corn_sizes: {NO_corn_sizes}  \n"
    f"-> batch size: {batch_size}  \n  \n"
    f"NO_workers: {NO_workers}  \n"
    f"queue_size: {queue_size}  \n"
    f"queue_size_tf: {queue_size_tf}  \n  \n"
    f"max_NO_elements: {generation_max_NO_wyckoffs}  \n"
    f"start_angle: {start_angle}  \n"
    f"end_angle: {end_angle}  \n"
    f"N: {N}  \n  \n"
    f"spgs: {str(spgs)}  \n  \n"
    f"do_symmetry_checks: {str(do_symmetry_checks)}  \n  \n"
    f"use_dropout: {str(use_dropout)} \n \n \n"
    f"learning_rate: {str(learning_rate)} \n \n \n"
    f"use_icsd_structures_directly: {str(use_icsd_structures_directly)} \n \n \n"
    f"load_only_N_patterns_each_test: {str(load_only_N_patterns_each_test)} \n \n \n"
    f"use_conditional_density: {str(use_conditional_denseness_factor_sampling)} \n \n \n"
    f"use_statistics_dataset_as_validation: {str(use_statistics_dataset_as_validation)} \n \n \n"
    f"sample_lattice_paras_from_kde: {str(sample_lattice_paras_from_kde)} \n \n \n"
    f"per_element: {str(per_element)} \n \n"
    f"use_distributed_strategy: {str(use_distributed_strategy)} \n \n"
    f"uniformly_distributed: {str(uniformly_distributed)} \n \n"
    f"shuffle_test_match_train_match: {str(shuffle_test_match_train_match)} \n \n"
    f"ray cluster resources: {str(ray.cluster_resources())}"
)
log_wait_timings = []

# This callback is used to calculate all the test accuracies
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % test_every_X_epochs) == 0:

            start = time.time()

            with file_writer.as_default():

                # Log the current queue size to make debugging easier
                tf.summary.scalar("queue size", data=queue.size(), step=epoch)

                # gather metric names form model
                metric_names = [metric.name for metric in self.model.metrics]

                scores_all = self.model.evaluate(
                    x=val_x_all[0:max_NO_samples_to_test_on],
                    y=val_y_all[0:max_NO_samples_to_test_on],
                    verbose=0,
                )
                scores_match = self.model.evaluate(
                    x=val_x_match[0:max_NO_samples_to_test_on],
                    y=val_y_match[0:max_NO_samples_to_test_on],
                    verbose=0,
                )
                scores_match_correct_spgs = self.model.evaluate(
                    x=val_x_match_correct_spgs[0:max_NO_samples_to_test_on],
                    y=val_y_match_correct_spgs[0:max_NO_samples_to_test_on],
                    verbose=0,
                )
                scores_match_correct_spgs_pure = self.model.evaluate(
                    x=val_x_match_correct_spgs_pure[0:max_NO_samples_to_test_on],
                    y=val_y_match_correct_spgs_pure[0:max_NO_samples_to_test_on],
                    verbose=0,
                )
                scores_random = self.model.evaluate(
                    x=val_x_random[0:max_NO_samples_to_test_on],
                    y=val_y_random[0:max_NO_samples_to_test_on],
                    verbose=0,
                )

                if generate_randomized_validation_datasets:
                    scores_randomized_coords = self.model.evaluate(
                        x=val_x_randomized_coords[0:max_NO_samples_to_test_on],
                        y=val_y_randomized_coords[0:max_NO_samples_to_test_on],
                        verbose=0,
                        batch_size=batch_size,
                    )
                    scores_randomized_ref = self.model.evaluate(
                        x=val_x_randomized_ref[0:max_NO_samples_to_test_on],
                        y=val_y_randomized_ref[0:max_NO_samples_to_test_on],
                        verbose=0,
                        batch_size=batch_size,
                    )
                    scores_randomized_lattice = self.model.evaluate(
                        x=val_x_randomized_lattice[0:max_NO_samples_to_test_on],
                        y=val_y_randomized_lattice[0:max_NO_samples_to_test_on],
                        verbose=0,
                        batch_size=batch_size,
                    )
                    scores_randomized_both = self.model.evaluate(
                        x=val_x_randomized_both[0:max_NO_samples_to_test_on],
                        y=val_y_randomized_both[0:max_NO_samples_to_test_on],
                        verbose=0,
                        batch_size=batch_size,
                    )

                if use_statistics_dataset_as_validation:
                    scores_statistics = self.model.evaluate(
                        x=statistics_x_match[0:max_NO_samples_to_test_on],
                        y=statistics_y_match[0:max_NO_samples_to_test_on],
                        verbose=0,
                        batch_size=batch_size,
                    )

                assert metric_names[0] == "loss"

                tf.summary.scalar("loss all", data=scores_all[0], step=epoch)
                tf.summary.scalar("loss match", data=scores_match[0], step=epoch)
                tf.summary.scalar(
                    "loss match_correct_spgs",
                    data=scores_match_correct_spgs[0],
                    step=epoch,
                )
                tf.summary.scalar(
                    "loss match_correct_spgs_pure",
                    data=scores_match_correct_spgs_pure[0],
                    step=epoch,
                )
                tf.summary.scalar("loss random", data=scores_random[0], step=epoch)

                if generate_randomized_validation_datasets:
                    tf.summary.scalar(
                        "loss randomized coords",
                        data=scores_randomized_coords[0],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "loss randomized ref", data=scores_randomized_ref[0], step=epoch
                    )
                    tf.summary.scalar(
                        "loss randomized lattice",
                        data=scores_randomized_lattice[0],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "loss randomized both",
                        data=scores_randomized_both[0],
                        step=epoch,
                    )

                if use_statistics_dataset_as_validation:
                    tf.summary.scalar(
                        "loss statistics", data=scores_statistics[0], step=epoch
                    )

                tf.summary.scalar("accuracy all", data=scores_all[1], step=epoch)
                tf.summary.scalar(
                    "accuracy match",
                    data=scores_match[1],
                    step=epoch,
                )
                tf.summary.scalar(
                    "accuracy match_correct_spgs",
                    data=scores_match_correct_spgs[1],
                    step=epoch,
                )
                tf.summary.scalar(
                    "accuracy match_correct_spgs_pure",
                    data=scores_match_correct_spgs_pure[1],
                    step=epoch,
                )
                tf.summary.scalar(
                    "accuracy random",
                    data=scores_random[1],
                    step=epoch,
                )

                if generate_randomized_validation_datasets:
                    tf.summary.scalar(
                        "accuracy randomized coords",
                        data=scores_randomized_coords[1],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "accuracy randomized ref",
                        data=scores_randomized_ref[1],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "accuracy randomized lattice",
                        data=scores_randomized_lattice[1],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "accuracy randomized both",
                        data=scores_randomized_both[1],
                        step=epoch,
                    )
                if use_statistics_dataset_as_validation:
                    tf.summary.scalar(
                        "accuracy statistics", data=scores_statistics[1], step=epoch
                    )

                tf.summary.scalar(
                    "top-5 accuracy match", data=scores_match[2], step=epoch
                )

                tf.summary.scalar(
                    "accuracy gap", data=scores_random[1] - scores_match[1], step=epoch
                )

                tf.summary.scalar("test time", data=time.time() - start, step=epoch)

                tf.summary.scalar(
                    "learning_rate", data=self.model.optimizer.lr, step=epoch
                )


# This keras Sequence object is used to fetch the training data from the ray
# queue object and pass it to the training loop
class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches, batch_size, number_of_epochs):
        self.number_of_batches = number_of_batches
        self._batch_size = batch_size
        self._number_of_epochs = number_of_epochs

    def __call__(
        self,
    ):  # This is necessary to be able to wrap this Sequence object in a tf.data.Dataset generator
        """Return next batch using an infinite generator model."""

        for i in range(self.__len__() * self._number_of_epochs):
            yield self.__getitem__(i)

            if (i + 1) % self.__len__() == 0:
                self.on_epoch_end()

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        start = time.time()
        result = queue.get()
        log_wait_timings.append(time.time() - start)
        auto_garbage_collect()
        return result


print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start training.",
    flush=True,
)

if use_distributed_strategy:
    strategy = tf.distribute.MirroredStrategy()

with (strategy.scope() if use_distributed_strategy else contextlib.nullcontext()):

    sequence = CustomSequence(batches_per_epoch, batch_size, NO_epochs)

    if use_distributed_strategy:
        dataset = tf.data.Dataset.from_generator(
            sequence,
            output_types=(tf.float64, tf.int64),
            output_shapes=(
                tf.TensorShape([None, None, None]),
                tf.TensorShape(
                    [
                        None,
                    ]
                ),
            ),
        )

    model_name = "model_resnet_50+additional_dense"

    if not use_pretrained_model:

        # model = build_model_park_2_layer_CNN(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # 7-label-version
        # model = build_model_park(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_park_medium_size(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_park_tiny_size(None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate)

        # Resnet-50 + additional dense layer
        model = build_model_resnet_i(
            None,
            N,
            len(spgs),
            lr=learning_rate,
            optimizer=optimizer,  # TODO: Remove this option
            batchnorm_momentum=batchnorm_momentum,
            i=50,
            disable_batchnorm=False,
            use_group_norm=use_group_norm,
            add_additional_dense_layer=True,  # Add one more dense layer
        )

    else:

        model = keras.models.load_model(
            pretrained_model_path, custom_objects={"AdamWarmup": AdamWarmup}
        )
        model.optimizer.learning_rate.assign(learning_rate)

    params_txt += "\n \n" + f"model_name: {model_name}"
    with file_writer.as_default():
        tf.summary.text("Parameters", data=params_txt, step=0)

    if not use_icsd_structures_directly:
        model.fit(
            x=dataset if use_distributed_strategy else sequence,
            epochs=NO_epochs,
            callbacks=[tb_callback, CustomCallback()],
            verbose=verbosity_tf,
            workers=1,
            max_queue_size=queue_size_tf,
            use_multiprocessing=False,  # Set this to False, since we use `ray` for distributed computing
            steps_per_epoch=batches_per_epoch,
        )
    else:
        model.fit(
            x=statistics_x_match,
            y=statistics_y_match,
            epochs=NO_epochs,
            batch_size=batch_size,
            callbacks=[tb_callback, CustomCallback()],
            verbose=verbosity_tf,
            workers=1,
            max_queue_size=queue_size_tf,
            use_multiprocessing=False,
        )

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Finished training.",
    flush=True,
)

model.save(out_base + "final")

#####
# We now evaluate the model on the match validation dataset and random
# validation dataset and store the correctly / falsely classified list indices
# in a pickle file. This way the analysis script
# `compare_random_distribution.py` can later use them.

# Get predictions for val_x_match and write rightly_indices / falsely_indices:
prediction_match = model.predict(val_x_match, batch_size=batch_size)
prediction_match = np.argmax(prediction_match, axis=1)

rightly_indices_match = np.argwhere(prediction_match == val_y_match)[:, 0]
falsely_indices_match = np.argwhere(prediction_match != val_y_match)[:, 0]

with open(out_base + "rightly_falsely_icsd.pickle", "wb") as file:
    pickle.dump((rightly_indices_match, falsely_indices_match), file)

# Get predictions for val_x_random and write rightly_indices / falsely_indices:
prediction_random = model.predict(val_x_random, batch_size=batch_size)
prediction_random = np.argmax(prediction_random, axis=1)

rightly_indices_random = np.argwhere(prediction_random == val_y_random)[:, 0]
falsely_indices_random = np.argwhere(prediction_random != val_y_random)[:, 0]

with open(out_base + "rightly_falsely_random.pickle", "wb") as file:
    pickle.dump((rightly_indices_random, falsely_indices_random), file)

# Get predictions for val_x_randomized_coords and write rightly_indices / falsely_indices:
if generate_randomized_validation_datasets:

    prediction_randomized_coords = model.predict(
        val_x_randomized_coords, batch_size=batch_size
    )
    prediction_randomized_coords = np.argmax(prediction_randomized_coords, axis=1)

    rightly_indices_randomized_coords = np.argwhere(
        prediction_randomized_coords == val_y_randomized_coords
    )[:, 0]
    falsely_indices_randomized_coords = np.argwhere(
        prediction_randomized_coords != val_y_randomized_coords
    )[:, 0]

    with open(out_base + "rightly_falsely_randomized_coords.pickle", "wb") as file:
        pickle.dump(
            (rightly_indices_randomized_coords, falsely_indices_randomized_coords), file
        )

    # Get predictions for val_x_randomized_ref and write rightly_indices / falsely_indices:
    prediction_randomized_ref = model.predict(
        val_x_randomized_ref, batch_size=batch_size
    )
    prediction_randomized_ref = np.argmax(prediction_randomized_ref, axis=1)

    rightly_indices_randomized_ref = np.argwhere(
        prediction_randomized_ref == val_y_randomized_ref
    )[:, 0]
    falsely_indices_randomized_ref = np.argwhere(
        prediction_randomized_ref != val_y_randomized_ref
    )[:, 0]

    with open(out_base + "rightly_falsely_randomized_ref.pickle", "wb") as file:
        pickle.dump(
            (rightly_indices_randomized_ref, falsely_indices_randomized_ref), file
        )

#####

ray.shutdown()

# The waiting times can later be used for debugging. If the waiting times are
# too big, more cores for the generation and simulation should be added.
with file_writer.as_default():
    for i, value in enumerate(log_wait_timings):
        tf.summary.scalar("waiting time", data=value, step=i)

#####
# Generate classification report on match validation dataset and save them to a pickle file:

report = classification_report(
    [spgs[i] for i in val_y_match],
    [spgs[i] for i in prediction_match],
    output_dict=True,
)
print("Classification report on match validation dataset:")
print(report)
with open(out_base + "classification_report_match.pickle", "wb") as file:
    pickle.dump(report, file)

# Do the same for the random validation dataset:
report = classification_report(
    [spgs[i] for i in val_y_random],
    [spgs[i] for i in prediction_random],
    output_dict=True,
)
print("Classification report on random dataset:")
print(report)
with open(out_base + "classification_report_random.pickle", "wb") as file:
    pickle.dump(report, file)

if (
    run_analysis_after_training
):  # Automatically trigger the analysis script after training

    print("Starting analysis now...", flush=True)

    if analysis_per_spg:
        for i, spg in enumerate(spgs):
            if np.sum(val_y_match == i) > 0:
                subprocess.call(
                    f"python compare_random_distribution.py {out_base} {date_time}_{tag} {spg}",
                    shell=True,
                )

    # Run analysis on all spgs
    spg_str = " ".join([str(spg) for spg in spgs])
    subprocess.call(
        f"python compare_random_distribution.py {out_base} {date_time}_{tag} {spg_str}",
        shell=True,
    )
