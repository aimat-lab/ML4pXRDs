import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
from dataset_simulations.random_simulation_utils import load_dataset_info
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
    build_model_resnet_50_old,
)

# from utils.transformer import build_model_transformer
from utils.transformer_vit import build_model_transformer_vit
from utils.distributed_utils import map_to_remote
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
from sklearn.metrics import classification_report
from pyxtal.symmetry import Group
import gc
import psutil
from sklearn.preprocessing import StandardScaler
from dataset_simulations.core.structure_generation import randomize
from dataset_simulations.core.quick_simulation import get_xy_patterns
import random
import contextlib
from train_dataset.utils.AdamWarmup import AdamWarmup
from train_dataset.utils.test_unet.rruff_helpers import *
from train_dataset.generate_background_noise_utils import generate_samples_gp
from train_dataset.utils.background_functions_vecsei import (
    generate_background_noise_vecsei,
)

tag = "all-spgs-random-direct-training-vecsei"
description = ""

if len(sys.argv) > 1:
    date_time = sys.argv[1]  # get it from the bash script
    out_base = "classifier_spgs/" + date_time + "/"
else:
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_base = "classifier_spgs/" + date_time + "_" + tag + "/"

if len(sys.argv) > 2 and sys.argv[2] == "head_only":
    head_only = True
    print("Running in head-only mode.", flush=True)
else:
    head_only = False

os.system("mkdir -p " + out_base)
os.system("mkdir -p " + out_base + "tuner_tb")
os.system("touch " + out_base + tag)

run_analysis_after_run = True
analysis_per_spg = False

test_every_X_epochs = 1
batches_per_epoch = 150 * 6  # doesn't count for direct training
NO_epochs = 1000

# For ViT:
# structures_per_spg = 1
# NO_corn_sizes = 1

structures_per_spg = 1  # for all spgs

# structures_per_spg = 5
# structures_per_spg = 10  # for (2,15) tuple
# structures_per_spg = 10  # for (2,15) tuple
# NO_corn_sizes = 5
NO_corn_sizes = 1
# structures_per_spg = 1  # 30-spg
# NO_corn_sizes = 3 # 30-spg

do_distance_checks = False
do_merge_checks = False
use_icsd_statistics = True

if not len(sys.argv) > 3:
    if not head_only:
        # NO_workers = 127 + 127 + 8  # for int-nano cluster
        NO_workers = 1 * 128 + 1 * 128 + 28  # for int-nano cluster
    else:
        NO_workers = 30  # for int-nano cluster
else:
    NO_workers = int(sys.argv[3])

# NO_workers = 14
# NO_workers = 40 * 5 + 5  # for bwuni

queue_size = 100  # if use_retention_of_patterns==True, then this is not used
queue_size_tf = 50

# NO_random_batches = 20
# NO_random_swipes = 1000  # make this smaller for the all-spgs run
# NO_random_swipes = 300 # 30-spg

NO_random_samples_per_spg = 100

generation_max_volume = 7000
generation_max_NO_wyckoffs = 100

validation_max_volume = 7000  # None possible
validation_max_NO_wyckoffs = 100  # None possible

do_symmetry_checks = True

use_NO_wyckoffs_counts = True
use_element_repetitions = True  # Overwrites use_NO_wyckoffs_counts
use_kde_per_spg = False  # Overwrites use_element_repetitions and use_NO_wyckoffs_counts
use_all_data_per_spg = False  # Overwrites all the previous ones
use_coordinates_directly = False
use_lattice_paras_directly = False
use_icsd_structures_directly = True  # This overwrites most of the previous settings and doesn't generate any crystals randomly (except for validation)!

use_statistics_dataset_as_validation = False
generate_randomized_validation_datasets = False
randomization_step = 3  # Only use every n'th sample for the randomization process

use_dropout = False

learning_rate = 0.0001

momentum = 0.9  # only used with SGD
optimizer = "Adam"  # not used for ViT
use_group_norm = True
use_reduce_lr_on_plateau = False
batchnorm_momentum = 0.0  # only used by ResNet and gigantic_more_dense_bn currently
estimate_bn_averages_using_random = False  # instead of the moving averages
calculate_random_accuracy_using_training_true = False
calculate_match_accuracy_using_training_true = False
max_NO_samples_to_test_on = 10000  # this should be plenty; this is only during the run.
log_bn_averages = False

use_denseness_factors_density = True
use_conditional_density = True
sample_lattice_paras_from_kde = True

load_only_N_patterns_each_test = 1  # None possible
load_only_N_patterns_each_train = 2  # None possible

scale_patterns = False
scale_patterns_sqrt = False

use_retention_of_patterns = False
retention_rate = 0.7

verbosity_tf = 2
verbosity_generator = 2

use_distributed_strategy = True

uniformly_distributed = False

shuffle_test_match_train_match = False

add_background_and_noise = True
use_vecsei_bg_noise = True
use_rruff_validation_dataset = True

use_pretrained_model = False  # Make it possible to resume from a previous training run
pretrained_model_path = "/home/ws/uvgnh/MSc/HEOs_MSc/train_dataset/classifier_spgs/07-06-2022_09-43-41/final"

local = False
if local:
    NO_workers = 8
    verbosity_tf = 1
    verbosity_generator = 1
    NO_random_samples_per_spg = 5
    randomization_step = 20
    use_distributed_strategy = False

git_revision_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
)

# spgs = [14, 104] # works well, relatively high val_acc
# spgs = [129, 176] # 93.15%, pretty damn well!
# spgs = [
#    2,
#    15,
# ]  # pretty much doesn't work at all (so far!), val_acc ~40%, after a full night: ~43%
# after a full night with random volume factors: binary_accuracy: 0.7603 - val_loss: 0.8687 - val_binary_accuracy: 0.4749; still bad
# spgs = [14, 104, 129, 176]  # after 100 epochs: 0.8503 val accuracy
# all spgs (~200): loss: sparse_categorical_accuracy: 0.1248 - val_sparse_categorical_accuracy: 0.0713; it is a beginning!

# spgs = list(range(201, 231))

# spgs = list(range(10, 21))
# spgs = list(range(150, 231))
# spgs = list(range(100, 231))

spgs = list(range(1, 231))
# spgs = [2, 15]

if len(spgs) == 2:
    NO_random_samples_per_spg = 500

# as Park:
# start_angle, end_angle, N = 10, 110, 10001

# as Vecsei:
start_angle, end_angle, N = 5, 90, 8501
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}, N: {N}", flush=True)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start loading dataset info.",
    flush=True,
)

(
    probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff,
    NO_wyckoffs_prob_per_spg,
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    denseness_factors_density_per_spg,
    kde_per_spg,
    all_data_per_spg_tmp,
    denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type,
    per_element,
    represented_spgs,
    (
        statistics_metas,
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

if scale_patterns and use_icsd_structures_directly:
    raise Exception(
        "Invalid options: Scaling patterns not compatible with direct icsd training."
    )

if not use_kde_per_spg:
    kde_per_spg = None

if not use_icsd_statistics:
    (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
    ) = (
        None,
        None,
        None,
    )

if not use_NO_wyckoffs_counts:
    NO_wyckoffs_prob_per_spg = None

if not use_element_repetitions:
    NO_unique_elements_prob_per_spg = None
    NO_repetitions_prob_per_spg_per_element = None

if not use_denseness_factors_density:
    denseness_factors_density_per_spg = None

if not use_conditional_density:
    denseness_factors_conditional_sampler_seeds_per_spg = None

if not sample_lattice_paras_from_kde:
    lattice_paras_density_per_lattice_type = None

if not use_all_data_per_spg:
    all_data_per_spg = None
else:
    all_data_per_spg = {}
    for spg in spgs:
        all_data_per_spg[spg] = all_data_per_spg_tmp[spg]

if estimate_bn_averages_using_random and batchnorm_momentum != 0.0:
    raise Exception(
        "Estimating the bn averages using the random distribution is only supported if batchnorm_momentum = 0.0."
    )

for i in reversed(range(0, len(spgs))):
    if spgs[i] not in represented_spgs:
        print(f"Excluded spg {spgs[i]} (not enough statistics).")
        del spgs[i]

batch_size = NO_corn_sizes * structures_per_spg * len(spgs)

print("len(spgs): ", len(spgs))
print("batch_size: ", batch_size)

if not uniformly_distributed:
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
    include_dashboard=False,
)

print()
print(ray.cluster_resources())
print()

# Construct validation sets
# Used validation sets:
# - All ICSD entries
# - ICSD entries that match simulation parameters
# - Pre-computed random dataset (the one from the comparison script)
# - Gap between training and val acc that matches simulation parameters

path_to_patterns = "../dataset_simulations/patterns/icsd_vecsei/"
jobid = os.getenv("SLURM_JOB_ID")
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

##### Prepare test datasets #####

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
    stop=1 if local else None,
    metas_to_load=metas_to_load_test,
)  # to not overflow the memory

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done loading patterns for test dataset.",
    flush=True,
)

if add_background_and_noise:
    for i, pattern in enumerate(icsd_sim_test.sim_patterns):
        for j in range(pattern.shape[0]):
            if not use_vecsei_bg_noise:
                pattern[j, :] = generate_samples_gp(
                    1,
                    (start_angle, end_angle),
                    n_angles_output=8501,
                    icsd_patterns=[pattern[j, :]],
                    original_range=True,
                )[0][0]
            else:
                pattern[j, :] += generate_background_noise_vecsei(angle_range)
                pattern[j, :] -= np.min(pattern[j, :])
                pattern[j, :] /= np.max(pattern[j, :])

n_patterns_per_crystal_test = len(icsd_sim_test.sim_patterns[0])

icsd_patterns_all = []
icsd_labels_all = []
icsd_variations_all = []
icsd_crystals_all = []
icsd_metas_all = []

icsd_patterns_match = []
icsd_labels_match = []
icsd_variations_match = []
icsd_crystals_match = []
icsd_metas_match = []

icsd_patterns_match_corrected_labels = []
icsd_labels_match_corrected_labels = []
icsd_variations_match_corrected_labels = []
icsd_crystals_match_corrected_labels = []
icsd_metas_match_corrected_labels = []

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
            )  # use the converted structure (conventional cell)
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
            )  # use the converted structure (conventional cell)
            icsd_metas_match.append(icsd_sim_test.sim_metas[i])

        if (
            corrected_labels[test_metas_flat.index(icsd_sim_test.sim_metas[i][0])]
            in spgs  # also excludes "None"
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
            in spgs  # also excludes "None"
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
        add_background_and_noise=add_background_and_noise,
        use_vecsei_bg_noise=use_vecsei_bg_noise,
    )
    return patterns[0], corn_sizes[0]


####### Generate match (corrected spgs) validation set with randomized coordinates and reference:

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

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[::randomization_step][i]:
                errors_counter += 1

                del labels[i]
                del randomized_coords_crystals[i]
                del reference_crystals[i]

    print(f"{errors_counter} of {len(labels)} mismatched (different tolerances)")

    randomized_coords_crystals = [
        item for item in randomized_coords_crystals if item is not None
    ]
    reference_crystals = [item for item in reference_crystals if item is not None]
    labels = [item for item in labels if item is not None]

    # Simulate on ray cluster:
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

##############

####### Generate match (corrected spgs) validation set with randomized lattice:

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

##############

####### Generate match (corrected spgs) validation set with randomized lattice and coords:

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

##############

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start constructing numpy arrays.",
    flush=True,
)

val_y_all = []
for i, label in enumerate(icsd_labels_all):
    val_y_all.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
val_y_all = np.array(val_y_all)

val_x_all = []
for pattern in icsd_patterns_all:
    for sub_pattern in pattern:
        val_x_all.append(sub_pattern)

# Moved creation of val_x_match and val_y_match down for shuffling option

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
        # val_y_randomized.append(spgs.index(label))
        val_y_randomized_coords.append(label)
    val_y_randomized_coords = np.array(val_y_randomized_coords)

    val_x_randomized_coords = []
    for pattern in randomized_coords_patterns:
        val_x_randomized_coords.append(pattern)

    val_y_randomized_ref = []
    for i, label in enumerate(randomized_coords_labels):
        # val_y_randomized_ref.append(spgs.index(label))
        val_y_randomized_ref.append(label)
    val_y_randomized_ref = np.array(val_y_randomized_ref)

    val_x_randomized_ref = []
    for pattern in reference_patterns:
        val_x_randomized_ref.append(pattern)

    val_y_randomized_lattice = []
    for i, label in enumerate(randomized_lattice_labels):
        # val_y_randomized.append(spgs.index(label))
        val_y_randomized_lattice.append(label)
    val_y_randomized_lattice = np.array(val_y_randomized_lattice)

    val_x_randomized_lattice = []
    for pattern in randomized_lattice_patterns:
        val_x_randomized_lattice.append(pattern)

    val_y_randomized_both = []
    for i, label in enumerate(randomized_both_labels):
        # val_y_randomized.append(spgs.index(label))
        val_y_randomized_both.append(label)
    val_y_randomized_both = np.array(val_y_randomized_both)

    val_x_randomized_both = []
    for pattern in randomized_both_patterns:
        val_x_randomized_both.append(pattern)

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done constructing numpy arrays.",
    flush=True,
)


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
    if not use_retention_of_patterns
    else int(batches_per_epoch * (1 - retention_rate))
)  # store a maximum of `queue_size` batches

# all_data_per_spg_handle = ray.put(all_data_per_spg)


def auto_garbage_collect(pct=93.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


@ray.remote(num_cpus=1, num_gpus=0)
def batch_generator_with_additional(
    spgs,
    structures_per_spg,
    N,
    start_angle,
    end_angle,
    max_NO_elements,
    NO_corn_sizes,
    # all_data_per_spg_handle,
):

    # all_data_per_spg_worker = ray.get(all_data_per_spg_handle)

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
        add_background_and_noise=add_background_and_noise,
        use_vecsei_bg_noise=use_vecsei_bg_noise,
    )

    # Set the label to the right index:
    for i in range(0, len(labels)):
        labels[i] = spgs.index(labels[i])

    patterns = np.array(patterns)
    labels = np.array(labels)

    auto_garbage_collect()

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
    sc=None
    # all_data_per_spg_handle,
):

    # all_data_per_spg_worker = ray.get(all_data_per_spg_handle)

    group_object_per_spg = {}
    for spg in spgs:
        group_object_per_spg[spg] = Group(spg, dim=3)

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
                group_object_per_spg=group_object_per_spg,
                denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                per_element=per_element,
                verbosity=verbosity_generator,
                probability_per_spg=probability_per_spg,
                add_background_and_noise=add_background_and_noise,
                use_vecsei_bg_noise=use_vecsei_bg_noise,
            )

            patterns, labels = shuffle(patterns, labels)

            # Set the label to the right index:
            for i in range(0, len(labels)):
                labels[i] = spgs.index(labels[i])

            patterns = np.array(patterns)

            if scale_patterns_sqrt:
                patterns = np.sqrt(patterns)

            if sc is not None:
                patterns = sc.transform(patterns)

            patterns = np.expand_dims(patterns, axis=2)

            labels = np.array(labels)

            auto_garbage_collect()

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

print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start generating validation random structures.",
    flush=True,
)

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
    inputs=range(NO_random_samples_per_spg),
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

if scale_patterns_sqrt:
    val_x_random = np.sqrt(val_x_random)
    val_x_all = np.sqrt(val_x_all)
    val_x_match_correct_spgs = np.sqrt(val_x_match_correct_spgs)
    val_x_match_correct_spgs_pure = np.sqrt(val_x_match_correct_spgs_pure)
    if generate_randomized_validation_datasets:
        val_x_randomized_coords = np.sqrt(val_x_randomized_coords)
        val_x_randomized_ref = np.sqrt(val_x_randomized_ref)
        val_x_randomized_lattice = np.sqrt(val_x_randomized_lattice)
        val_x_randomized_both = np.sqrt(val_x_randomized_both)

if scale_patterns:
    sc = StandardScaler(with_mean=False)
    val_x_random = sc.fit_transform(val_x_random)

    with open(out_base + "sc.pickle", "wb") as file:
        pickle.dump(sc, file)

    val_x_all = sc.transform(val_x_all)
    val_x_match_correct_spgs = sc.transform(val_x_match_correct_spgs)
    val_x_match_correct_spgs_pure = sc.transform(val_x_match_correct_spgs_pure)

    if generate_randomized_validation_datasets:
        val_x_randomized_coords = sc.transform(val_x_randomized_coords)
        val_x_randomized_ref = sc.transform(val_x_randomized_ref)
        val_x_randomized_lattice = sc.transform(val_x_randomized_lattice)
        val_x_randomized_both = sc.transform(val_x_randomized_both)

val_x_all = np.expand_dims(val_x_all, axis=2)
val_x_match_correct_spgs = np.expand_dims(val_x_match_correct_spgs, axis=2)
val_x_match_correct_spgs_pure = np.expand_dims(val_x_match_correct_spgs_pure, axis=2)
val_x_random = np.expand_dims(val_x_random, axis=2)

if generate_randomized_validation_datasets:
    val_x_randomized_coords = np.expand_dims(val_x_randomized_coords, axis=2)
    val_x_randomized_ref = np.expand_dims(val_x_randomized_ref, axis=2)
    val_x_randomized_lattice = np.expand_dims(val_x_randomized_lattice, axis=2)
    val_x_randomized_both = np.expand_dims(val_x_randomized_both, axis=2)

with open(out_base + "random_data.pickle", "wb") as file:
    pickle.dump(
        (
            random_comparison_crystals,
            random_comparison_labels,
            random_comparison_corn_sizes,
        ),
        file,
    )

#########################################################
# Load the rruff test dataset:
if use_rruff_validation_dataset:

    xs_rruff, ys_rruff, dif_files, raw_files = get_rruff_patterns(
        only_refitted_patterns=False,
        only_selected_patterns=True,
        start_angle=5,
        end_angle=90,
        reduced_resolution=False,
        only_if_dif_exists=True,  # skips patterns where no dif is file
    )

    val_x_rruff = []
    val_y_rruff = []
    for i, pattern in enumerate(ys_rruff):
        if len(pattern) == 8501:

            try:
                data, wavelength, spg_number = dif_parser(dif_files[i])
            except Exception as ex:
                continue

            if spg_number in spgs:
                val_x_rruff.append(pattern)
                val_y_rruff.append(spgs.index(spg_number))

    val_y_rruff = np.array(val_y_rruff)

    if scale_patterns_sqrt:
        val_x_rruff = np.sqrt(val_x_rruff)

    if scale_patterns:
        val_x_rruff = sc.transform(val_x_rruff)

    val_x_rruff = np.expand_dims(val_x_rruff, axis=2)

    print("Size of rruff validation dataset:", val_x_rruff.shape, val_y_rruff.shape)

#########################################################
# Prepare the training directly from ICSD OR statistics validation dataset

if use_icsd_structures_directly or use_statistics_dataset_as_validation:

    if use_icsd_structures_directly and use_statistics_dataset_as_validation:
        raise Exception(
            "Cannot train and validate on statistics dataset at the same time."
        )

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
        else load_only_N_patterns_each_train,
        metas_to_load=statistics_match_metas_flat,
        stop=1 if local else None,
    )  # to not overflow the memory if local

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Done loading patterns for statistics / training.",
        flush=True,
    )

    if add_background_and_noise:
        for i, pattern in enumerate(icsd_sim_statistics.sim_patterns):
            for j in range(pattern.shape[0]):
                if not use_vecsei_bg_noise:
                    pattern[j, :] = generate_samples_gp(
                        1,
                        (start_angle, end_angle),
                        n_angles_output=8501,
                        icsd_patterns=[pattern[j, :]],
                        original_range=True,
                    )[0][0]
                else:
                    pattern[j, :] += generate_background_noise_vecsei(angle_range)
                    pattern[j, :] -= np.min(pattern[j, :])
                    pattern[j, :] /= np.max(pattern[j, :])

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
            )  # use the converted structure (conventional cell)
            statistics_icsd_metas_match.append(icsd_sim_statistics.sim_metas[i])

    n_patterns_per_crystal_statistics = len(icsd_sim_statistics.sim_patterns[0])

    ##########

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

        val_y_match = []
        val_x_match = []
        statistics_y_match = []
        statistics_x_match = []

        total = (
            len(icsd_labels_match) * n_patterns_per_crystal_test
            + len(statistics_icsd_labels_match) * n_patterns_per_crystal_statistics
        )
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

    ##########

    statistics_x_match, statistics_y_match = shuffle(
        statistics_x_match, statistics_y_match
    )

    statistics_y_match = np.array(statistics_y_match)

    if scale_patterns_sqrt:
        statistics_x_match = np.sqrt(statistics_x_match)

    if scale_patterns:
        statistics_x_match = sc.transform(statistics_x_match)

    statistics_x_match = np.expand_dims(statistics_x_match, axis=2)

    print(
        "Size of statistics / training dataset: ", statistics_x_match.shape, flush=True
    )

else:

    val_y_match = []
    for i, label in enumerate(icsd_labels_match):
        val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)

    val_x_match = []
    for pattern in icsd_patterns_match:
        for sub_pattern in pattern:
            val_x_match.append(sub_pattern)

##########

val_y_match = np.array(val_y_match)
print("Numbers in validation set (that matches sim parameters):")
for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y_match==i)}")

assert not np.any(np.isnan(val_x_match))
assert not np.any(np.isnan(val_y_match))
assert len(val_x_match) == len(val_y_match)

if scale_patterns_sqrt:
    val_x_match = np.sqrt(val_x_match)

if scale_patterns:
    val_x_match = sc.transform(val_x_match)

val_x_match = np.expand_dims(val_x_match, axis=2)

print("Size of test dataset: ", val_x_match.shape, flush=True)

#########################################################

if not use_icsd_structures_directly:
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
            sc=sc if scale_patterns else None
            # all_data_per_spg_handle,
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
    f"do_distance_checks: {str(do_distance_checks)}  \n"
    f"do_merge_checks: {str(do_merge_checks)}  \n  \n"
    f"use_icsd_statistics: {str(use_icsd_statistics)}  \n  \n"
    f"validation_max_volume: {str(validation_max_volume)}  \n"
    f"validation_max_NO_wyckoffs: {str(validation_max_NO_wyckoffs)}  \n  \n"
    f"spgs: {str(spgs)}  \n  \n"
    f"do_symmetry_checks: {str(do_symmetry_checks)}  \n  \n"
    f"use_NO_wyckoffs_counts: {str(use_NO_wyckoffs_counts)} \n \n \n"
    f"use_element_repetitions: {str(use_element_repetitions)} \n \n \n"
    f"use_dropout: {str(use_dropout)} \n \n \n"
    f"learning_rate: {str(learning_rate)} \n \n \n"
    f"use_denseness_factors_density: {str(use_denseness_factors_density)} \n \n \n"
    f"use_kde_per_spg: {str(use_kde_per_spg)} \n \n \n"
    f"use_all_data_per_spg: {str(use_all_data_per_spg)} \n \n \n"
    f"use_coordinates_directly: {str(use_coordinates_directly)} \n \n \n"
    f"use_lattice_paras_directly: {str(use_lattice_paras_directly)} \n \n \n"
    f"use_icsd_structures_directly: {str(use_icsd_structures_directly)} \n \n \n"
    f"load_only_N_patterns_each_test: {str(load_only_N_patterns_each_test)} \n \n \n"
    f"scale_patterns: {str(scale_patterns)} \n \n \n"
    f"use_conditional_density: {str(use_conditional_density)} \n \n \n"
    f"use_statistics_dataset_as_validation: {str(use_statistics_dataset_as_validation)} \n \n \n"
    f"sample_lattice_paras_from_kde: {str(sample_lattice_paras_from_kde)} \n \n \n"
    f"per_element: {str(per_element)} \n \n"
    f"use_distributed_strategy: {str(use_distributed_strategy)} \n \n"
    f"uniformly_distributed: {str(uniformly_distributed)} \n \n"
    f"shuffle_test_match_train_match: {str(shuffle_test_match_train_match)} \n \n"
    f"add_background_and_noise: {str(add_background_and_noise)} \n \n"
    f"use_vecsei_bg_noise: {str(use_vecsei_bg_noise)} \n \n"
    f"ray cluster resources: {str(ray.cluster_resources())}"
)

log_wait_timings = []


def change_bn_momentum(model, bn_momentum):
    """This is quite hacky. But what needs to be done, needs to be done."""

    for layer in model.layers:
        if "batch_normalization" in layer._name:
            layer.momentum = bn_momentum
        if hasattr(layer, "sub_layers"):
            for sub_layer in layer.sub_layers:
                if "batch_normalization" in sub_layer._name:
                    sub_layer.momentum = bn_momentum


def calculate_accuracy_training_true(
    model,
    x_data,
    y_data,
    batch_size,
    n_batches=None,
    change_bn_momentum_to=0.0,
):

    if change_bn_momentum_to != 0.0:
        change_bn_momentum(model, 0.0)  # First, set it to zero

    total_correct = 0

    if n_batches is None:
        n_batches = int(x_data.shape[0] / batch_size)

    for i in range(0, n_batches):  # only use actually full batches here for testing

        if (
            i == 1 and change_bn_momentum_to != 0.0
        ):  # after the first batch has been processed
            change_bn_momentum(model, change_bn_momentum_to)

        prediction = model(
            x_data[i * batch_size : (i + 1) * batch_size, :, :],
            training=True,
        )  # run this in training mode

        prediction = np.argmax(prediction, axis=1)

        rightly_indices = np.argwhere(
            prediction == y_data[i * batch_size : (i + 1) * batch_size]
        )[:, 0]

        total_correct += len(rightly_indices)

    accuracy = total_correct / (n_batches * batch_size)

    return accuracy


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % test_every_X_epochs) == 0:

            start = time.time()

            with file_writer.as_default():

                tf.summary.scalar("queue size", data=queue.size(), step=epoch)

                # gather metric names form model
                metric_names = [metric.name for metric in self.model.metrics]

                if estimate_bn_averages_using_random:
                    # We need ~ 1000 samples for a good average
                    n_batches = int(np.ceil(870 / batch_size))
                    accuracy = calculate_accuracy_training_true(
                        self.model,
                        val_x_random,
                        val_y_random,  # pre-estimate the bn averages before evaluation
                        batch_size=batch_size,
                        n_batches=n_batches,
                        change_bn_momentum_to=1 - (1 / n_batches),
                    )

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
                if use_rruff_validation_dataset:
                    scores_rruff = self.model.evaluate(
                        x=val_x_rruff[0:max_NO_samples_to_test_on],
                        y=val_y_rruff[0:max_NO_samples_to_test_on],
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

                if (
                    calculate_random_accuracy_using_training_true
                ):  # for debugging of batchnormalization
                    accuracy_random_training_true = calculate_accuracy_training_true(
                        self.model, val_x_random, val_y_random, batch_size=batch_size
                    )
                    tf.summary.scalar(
                        "accuracy random training=True",
                        data=accuracy_random_training_true,
                        step=epoch,
                    )

                if (
                    calculate_match_accuracy_using_training_true
                ):  # for debugging of batchnormalization
                    accuracy_match_training_true = calculate_accuracy_training_true(
                        self.model, val_x_match, val_y_match, batch_size=batch_size
                    )
                    tf.summary.scalar(
                        "accuracy match training=True",
                        data=accuracy_match_training_true,
                        step=epoch,
                    )

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
                if use_rruff_validation_dataset:
                    tf.summary.scalar(
                        "loss rruff",
                        data=scores_rruff[0],
                        step=epoch,
                    )
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
                if use_rruff_validation_dataset:
                    tf.summary.scalar(
                        "accuracy rruff",
                        data=scores_rruff[1],
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

                if log_bn_averages:
                    mean = np.average(self.model.layers[2].moving_mean.numpy())
                    variance = np.average(self.model.layers[2].moving_variance.numpy())
                    tf.summary.scalar("bn_0_average_means", data=mean, step=epoch)
                    tf.summary.scalar(
                        "bn_0_average_variances", data=variance, step=epoch
                    )
                    tf.summary.scalar(
                        "bn_0_mean_0",
                        data=self.model.layers[2].moving_mean.numpy()[0],
                        step=epoch,
                    )
                    tf.summary.scalar(
                        "bn_0_variance_0",
                        data=self.model.layers[2].moving_variance.numpy()[0],
                        step=epoch,
                    )


class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches, batch_size, number_of_epochs):
        self.number_of_batches = number_of_batches
        self._batch_size = batch_size
        self._number_of_epochs = number_of_epochs

        self._current_index = 0

        if use_retention_of_patterns:
            self.patterns = None
            self.labels = None
            self.indices = None

    def __call__(self):
        """Return next batch using an infinite generator model."""

        for i in range(self.__len__() * self._number_of_epochs):
            yield self.__getitem__(i)

            if (i + 1) % self.__len__() == 0:
                self.on_epoch_end()

        # self._current_index = (self._current_index + 1) % self.number_of_batches
        # result = self[self._current_index]
        # return result

    def __len__(self):
        return self.number_of_batches

    def on_epoch_end(self):

        if use_retention_of_patterns:

            self.replace_patterns()
            random.shuffle(self.indices)

    def __getitem__(self, idx):

        if not use_retention_of_patterns:

            start = time.time()
            result = queue.get()
            log_wait_timings.append(time.time() - start)
            auto_garbage_collect()
            return result

        else:

            log_wait_timings.append(0)

            indices = self.indices[
                idx
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs) : (idx + 1)
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs)
            ]

            return self.patterns[indices, :, :], self.labels[indices]

    def pre_compute(self):

        # Pre-compute a whole epoch

        self.patterns = np.empty(
            shape=(
                self.number_of_batches * structures_per_spg * NO_corn_sizes * len(spgs),
                N,
                1,
            )
        )
        self.labels = np.empty(
            shape=(
                self.number_of_batches * structures_per_spg * NO_corn_sizes * len(spgs)
            )
        )

        for i in range(0, self.number_of_batches):

            (patterns, labels) = queue.get()
            self.patterns[
                i
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs) : (i + 1)
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs),
                :,
                :,
            ] = patterns
            self.labels[
                i
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs) : (i + 1)
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs)
            ] = labels

        self.indices = list(range(0, self.patterns.shape[0]))
        random.shuffle(self.indices)

    def replace_patterns(self):

        for i in range(0, int(self.number_of_batches * (1 - retention_rate))):

            (patterns, labels) = queue.get()

            indices_to_replace = self.indices[
                i
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs) : (i + 1)
                * structures_per_spg
                * NO_corn_sizes
                * len(spgs)
            ]

            self.patterns[indices_to_replace, :, :] = patterns
            self.labels[indices_to_replace] = labels


print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Start training.",
    flush=True,
)

if use_distributed_strategy:
    strategy = tf.distribute.MirroredStrategy()

with (strategy.scope() if use_distributed_strategy else contextlib.nullcontext()):

    if use_reduce_lr_on_plateau:
        lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", verbose=1, factor=0.2, patience=25, cooldown=20
        )

    sequence = CustomSequence(batches_per_epoch, batch_size, NO_epochs)

    if use_retention_of_patterns:
        sequence.pre_compute()

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

    model_name = "model_gigantic_additional_dense"

    if not use_pretrained_model:

        # model = build_model_park_2_layer_CNN(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_park(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_park_medium_size(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # Resnet-10
        # model = build_model_resnet_i(
        #    None,
        #    N,
        #    len(spgs),
        #    lr=learning_rate,
        #    momentum=momentum,
        #    optimizer=optimizer,
        #    batchnorm_momentum=batchnorm_momentum,
        #    i=10,
        #    disable_batchnorm=False,
        # )

        # Resnet-50 + additional dense layer
        model = build_model_resnet_i(
            None,
            N,
            len(spgs),
            lr=learning_rate,
            momentum=momentum,
            optimizer=optimizer,
            batchnorm_momentum=batchnorm_momentum,
            i=50,
            disable_batchnorm=False,
            use_group_norm=use_group_norm,
            add_additional_dense_layer=True,  # one more dense layer
        )

        # model = build_model_park_tiny_size(None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate)
        # model = build_model_resnet_50(None, N, len(spgs), False, lr=learning_rate)

        # model = build_model_park_huge_size(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_transformer(None, N, len(spgs), lr=learning_rate, epochs=NO_epochs, steps_per_epoch=batches_per_epoch)

        # model = build_model_transformer_vit(
        #    None,
        #    N,
        #    len(spgs),
        #    lr=learning_rate,
        #    epochs=NO_epochs,
        #    steps_per_epoch=batches_per_epoch,
        # )

        # model = build_model_park_gigantic_size(
        #    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
        # )

        # model = build_model_park_gigantic_size_more_dense(
        #   None,
        #   N,
        #   len(spgs),
        #   use_dropout=use_dropout,
        #   lr=learning_rate,
        #   momentum=momentum,
        #   optimizer=optimizer,
        # )

        # model = build_model_park_gigantic_size_more_dense_bn(
        #    None,
        #    N,
        #    len(spgs),
        #    use_dropout=use_dropout,
        #    lr=learning_rate,
        #    momentum=momentum,
        #    optimizer=optimizer,
        #    bn_momentum=batchnorm_momentum,
        # )

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
            callbacks=[tb_callback, CustomCallback()]
            if not use_reduce_lr_on_plateau
            else [tb_callback, CustomCallback(), lr_callback],
            verbose=verbosity_tf,
            workers=1,
            max_queue_size=queue_size_tf,
            use_multiprocessing=False,
            steps_per_epoch=batches_per_epoch,
        )
    else:
        model.fit(
            x=statistics_x_match,
            y=statistics_y_match,
            epochs=NO_epochs,
            batch_size=batch_size,
            callbacks=[tb_callback, CustomCallback()]
            if not use_reduce_lr_on_plateau
            else [tb_callback, CustomCallback(), lr_callback],
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

if use_rruff_validation_dataset:
    # Get predictions for val_x_rruff:
    prediction_rruff = model.predict(val_x_rruff, batch_size=batch_size)
    prediction_rruff = np.argmax(prediction_rruff, axis=1)

# Get predictions for val_x_randomized and write rightly_indices / falsely_indices:
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

##########

ray.shutdown()

with file_writer.as_default():
    for i, value in enumerate(log_wait_timings):
        tf.summary.scalar("waiting time", data=value, step=i)

print("Training finished.")
print("Output dir:")
print(out_base)

report = classification_report(
    [spgs[i] for i in val_y_match],
    [spgs[i] for i in prediction_match],
    output_dict=True,
)
print("Classification report on match validation dataset:")
print(report)
with open(out_base + "classification_report_match.pickle", "wb") as file:
    pickle.dump(report, file)

report = classification_report(
    [spgs[i] for i in val_y_random],
    [spgs[i] for i in prediction_random],
    output_dict=True,
)
print("Classification report on random dataset:")
print(report)
with open(out_base + "classification_report_random.pickle", "wb") as file:
    pickle.dump(report, file)

if use_rruff_validation_dataset:
    report = classification_report(
        [spgs[i] for i in val_y_rruff],
        [spgs[i] for i in prediction_rruff],
        output_dict=True,
    )
    print("Classification report on rruff dataset:")
    print(report)
    with open(out_base + "classification_report_rruff.pickle", "wb") as file:
        pickle.dump(report, file)

if run_analysis_after_run:

    print("Starting analysis now...", flush=True)

    if analysis_per_spg:
        for i, spg in enumerate(spgs):

            if np.sum(val_y_match == i) > 0:
                subprocess.call(
                    f"python compare_random_distribution.py {out_base} {date_time}_{tag} {spg}",
                    shell=True,
                )

    spg_str = " ".join([str(spg) for spg in spgs])
    subprocess.call(
        f"python compare_random_distribution.py {out_base} {date_time}_{tag} {spg_str}",
        shell=True,
    )
