import tensorflow.keras as keras
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
from dataset_simulations.random_simulation_utils import load_dataset_info
import numpy as np
from models import (
    build_model_park,
    build_model_park_medium_size,
    build_model_park_huge_size,
    build_model_park_original_spg,
    build_model_park_tiny_size,
    build_model_resnet_10,
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
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.metrics import classification_report
from pyxtal.symmetry import Group
import gc
import psutil
from sklearn.preprocessing import StandardScaler
from dataset_simulations.core.structure_generation import randomize
from dataset_simulations.core.quick_simulation import get_xy_patterns
import random

tag = "all-spgs-direct-original-new-split"
description = ""

if len(sys.argv) > 1:
    date_time = sys.argv[1]  # get it from the bash script
    out_base = "classifier_spgs/" + date_time + "/"
else:
    date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    out_base = "classifier_spgs/" + date_time + "_" + tag + "/"

os.system("mkdir -p " + out_base)
os.system("mkdir -p " + out_base + "tuner_tb")
os.system("touch " + out_base + tag)

run_analysis_after_run = True
analysis_per_spg = False

test_every_X_epochs = 1
batches_per_epoch = 150
NO_epochs = 600  # roughly equivalent to 1400 epochs used for training on random data

structures_per_spg = 2  # for all spgs
# structures_per_spg = 5
# structures_per_spg = 10  # for (2,15) tuple
# structures_per_spg = 10  # for (2,15) tuple
# NO_corn_sizes = 5
NO_corn_sizes = 3
# structures_per_spg = 1  # 30-spg
# NO_corn_sizes = 3 # 30-spg

do_distance_checks = False
do_merge_checks = False
use_icsd_statistics = True

NO_workers = 127 + 127 + 8  # for int-nano cluster
# NO_workers = 14
# NO_workers = 40 * 5 + 5  # for bwuni

queue_size = 120  # if use_retention_of_patterns==True, then this is not used
queue_size_tf = 60

# NO_random_batches = 20
# NO_random_swipes = 1000  # make this smaller for the all-spgs run
# NO_random_swipes = 300 # 30-spg

NO_random_samples_per_spg = 200

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

use_dropout = True

learning_rate = 0.0003

# momentum = 0.7
# optimizer = "SGD"
use_reduce_lr_on_plateau = False

use_denseness_factors_density = True
use_conditional_density = True

sample_lattice_paras_from_kde = True

load_only_N_patterns_each_test = 1  # None possible
load_only_N_patterns_each_train = 2  # None possible

scale_patterns = False

use_retention_of_patterns = False
retention_rate = 0.7

verbosity = 2

local = False
if local:
    NO_workers = 8
    verbosity = 1
    NO_random_samples_per_spg = 20

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

if len(spgs) == 2:
    NO_random_samples_per_spg = 500

# as Park:
# start_angle, end_angle, N = 10, 110, 10001

# as Vecsei:
start_angle, end_angle, N = 5, 90, 8501
angle_range = np.linspace(start_angle, end_angle, N)
print(f"Start-angle: {start_angle}, end-angle: {end_angle}, N: {N}", flush=True)

(
    probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff,
    NO_wyckoffs_prob_per_spg,
    corrected_labels,
    statistics_metas,
    test_metas,
    represented_spgs,
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    denseness_factors_density_per_spg,
    kde_per_spg,
    all_data_per_spg_tmp,
    denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type,
) = load_dataset_info()

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
else:
    for i in reversed(range(0, len(spgs))):
        if denseness_factors_density_per_spg[spgs[i]] is None:
            print(
                f"Excluded spg {spgs[i]} due to missing denseness_factor density (not enough statistics)."
            )
            del spgs[i]

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

for i in reversed(range(len(represented_spgs))):
    if np.sum(NO_wyckoffs_prob_per_spg[represented_spgs[i]][0:100]) <= 0.01:
        print(
            f"Excluded spg {represented_spgs[i]} from represented_spgs due to low probability in NO_wyckoffs < 100."
        )
        del represented_spgs[i]

for spg in spgs:
    if spg not in represented_spgs:
        raise Exception("Requested space group not represented in prepared statistics.")

batch_size = NO_corn_sizes * structures_per_spg * len(spgs)

print("len(spgs): ", len(spgs))
print("batch_size: ", batch_size)

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

icsd_sim_test.load(
    load_only_N_patterns_each=load_only_N_patterns_each_test,
    stop=6 if local else None,
    metas_to_load=[item[0] for item in test_metas],
)  # to not overflow the memory

n_patterns_per_crystal_test = len(icsd_sim_test.sim_patterns[0])

icsd_patterns_all = icsd_sim_test.sim_patterns
icsd_labels_all = icsd_sim_test.sim_labels
icsd_variations_all = icsd_sim_test.sim_variations
icsd_crystals_all = icsd_sim_test.sim_crystals
icsd_metas_all = icsd_sim_test.sim_metas

# Mainly to make the volume constraints correct:
conventional_errors_counter = 0
conventional_counter = 0
print(
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Calculating conventional structures...",
    flush=True,
)

for i in reversed(range(0, len(icsd_crystals_all))):
    # Only needed if the sample will actually be used later!
    if icsd_labels_all[i][0] in spgs or corrected_labels[icsd_metas_all[i][0]] in spgs:
        conventional_counter += 1
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
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: {conventional_errors_counter} of {conventional_counter} failed to convert to conventional cell.",
    flush=True,
)

icsd_patterns_match_corrected_labels = icsd_patterns_all.copy()
icsd_crystals_match_corrected_labels = icsd_crystals_all.copy()
icsd_variations_match_corrected_labels = icsd_variations_all.copy()
icsd_metas_match_corrected_labels = icsd_metas_all.copy()
icsd_labels_match_corrected_labels = [
    corrected_labels[meta[0]] for meta in icsd_metas_match_corrected_labels
]  # corrected labels from spglib

assert len(icsd_labels_match_corrected_labels) == len(icsd_labels_all)

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
is_pure_counter = 0

for i in reversed(range(0, len(icsd_patterns_match))):

    if validation_max_NO_wyckoffs is not None:
        is_pure, NO_wyckoffs, _, _, _, _, _, _ = icsd_sim_test.get_wyckoff_info(
            icsd_metas_match[i][0]
        )

        if icsd_metas_match[i][0] not in NO_wyckoffs_cached.keys():
            NO_wyckoffs_cached[icsd_metas_match[i][0]] = is_pure, NO_wyckoffs

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
    else:
        if is_pure:
            is_pure_counter += 1

print(f"is_pure: {is_pure_counter} of {len(icsd_patterns_match)}")

icsd_patterns_match_corrected_labels_pure = icsd_patterns_match_corrected_labels.copy()
icsd_labels_match_corrected_labels_pure = icsd_labels_match_corrected_labels.copy()
icsd_variations_match_corrected_labels_pure = (
    icsd_variations_match_corrected_labels.copy()
)
icsd_crystals_match_corrected_labels_pure = icsd_crystals_match_corrected_labels.copy()
icsd_metas_match_corrected_labels_pure = icsd_metas_match_corrected_labels.copy()

for i in reversed(range(0, len(icsd_patterns_match_corrected_labels))):

    if validation_max_NO_wyckoffs is not None:
        if icsd_metas_match_corrected_labels[i][0] not in NO_wyckoffs_cached.keys():
            is_pure, NO_wyckoffs, _, _, _, _, _, _ = icsd_sim_test.get_wyckoff_info(
                icsd_metas_match_corrected_labels[i][0]
            )
        else:
            is_pure, NO_wyckoffs = NO_wyckoffs_cached[
                icsd_metas_match_corrected_labels[i][0]
            ]

    if is_pure != icsd_crystals_match_corrected_labels[i].is_ordered:
        print("########## Warning: is_pure != is_ordered")

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

    if (
        (
            validation_max_volume is not None
            and icsd_crystals_match_corrected_labels_pure[i].volume
            > validation_max_volume
        )
        or (
            validation_max_NO_wyckoffs is not None
            and NO_wyckoffs > validation_max_NO_wyckoffs
        )
        or not is_pure
    ):

        del icsd_patterns_match_corrected_labels_pure[i]
        del icsd_labels_match_corrected_labels_pure[i]
        del icsd_variations_match_corrected_labels_pure[i]
        del icsd_crystals_match_corrected_labels_pure[i]
        del icsd_metas_match_corrected_labels_pure[i]

icsd_patterns_match_inorganic = icsd_patterns_match.copy()
icsd_labels_match_inorganic = icsd_labels_match.copy()
icsd_variations_match_inorganic = icsd_variations_match.copy()
icsd_crystals_match_inorganic = icsd_crystals_match.copy()
icsd_metas_match_inorganic = icsd_metas_match.copy()

exp_inorganic, exp_metalorganic, theoretical = icsd_sim_test.get_content_types()

for i in reversed(range(0, len(icsd_patterns_match_inorganic))):

    if icsd_metas_match_inorganic[i][0] not in exp_inorganic:

        del icsd_patterns_match_inorganic[i]
        del icsd_labels_match_inorganic[i]
        del icsd_variations_match_inorganic[i]
        del icsd_crystals_match_inorganic[i]
        del icsd_metas_match_inorganic[i]

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
    )
    return patterns[0], corn_sizes[0]


####### Generate match (corrected spgs) validation set with randomized coordinates and reference:

if generate_randomized_validation_datasets:

    randomized_coords_crystals, reference_crystals, labels = randomize(
        icsd_crystals_match_corrected_labels,
        randomize_coordinates=True,
        randomize_lattice=False,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[i]:
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

    randomized_lattice_crystals, _, labels = randomize(
        icsd_crystals_match_corrected_labels,
        randomize_coordinates=False,
        randomize_lattice=True,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[i]:
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

##############

####### Generate match (corrected spgs) validation set with randomized lattice and coords:

if generate_randomized_validation_datasets:

    randomized_both_crystals, _, labels = randomize(
        icsd_crystals_match_corrected_labels,
        randomize_coordinates=True,
        randomize_lattice=True,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )

    errors_counter = 0
    for i in reversed(range(len(labels))):

        label = labels[i]

        if label is not None:
            if label != icsd_labels_match_corrected_labels[i]:
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

##############

val_y_all = []
for i, label in enumerate(icsd_labels_all):
    val_y_all.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
val_y_all = np.array(val_y_all)

val_x_all = []
for pattern in icsd_patterns_all:
    for sub_pattern in pattern:
        val_x_all.append(sub_pattern)

val_y_match = []
for i, label in enumerate(icsd_labels_match):
    val_y_match.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
val_y_match = np.array(val_y_match)

val_x_match = []
for pattern in icsd_patterns_match:
    for sub_pattern in pattern:
        val_x_match.append(sub_pattern)

val_y_match_inorganic = []
for i, label in enumerate(icsd_labels_match_inorganic):
    val_y_match_inorganic.extend([spgs.index(label[0])] * n_patterns_per_crystal_test)
val_y_match_inorganic = np.array(val_y_match_inorganic)

val_x_match_inorganic = []
for pattern in icsd_patterns_match_inorganic:
    for sub_pattern in pattern:
        val_x_match_inorganic.append(sub_pattern)

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


print("Numbers in validation set (that matches sim parameters):")
for i in range(0, len(spgs)):
    print(f"Spg {spgs[i]} : {np.sum(val_y_match==i)}")

assert not np.any(np.isnan(val_x_all))
assert not np.any(np.isnan(val_y_all))
assert not np.any(np.isnan(val_x_match))
assert not np.any(np.isnan(val_y_match))
assert not np.any(np.isnan(val_x_match_correct_spgs))
assert not np.any(np.isnan(val_y_match_correct_spgs))
assert not np.any(np.isnan(val_x_match_correct_spgs_pure))
assert not np.any(np.isnan(val_y_match_correct_spgs_pure))
assert len(val_x_all) == len(val_y_all)
assert len(val_x_match) == len(val_y_match)
assert len(val_x_match_inorganic) == len(val_y_match_inorganic)
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
            )

            patterns, labels = shuffle(patterns, labels)

            # Set the label to the right index:
            for i in range(0, len(labels)):
                labels[i] = spgs.index(labels[i])

            patterns = np.array(patterns)

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
    f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Finished generating validation random structures.",
    flush=True,
)

print("Sizes of validation sets:")
print(f"all: {len(icsd_labels_all)} * {n_patterns_per_crystal_test}")
print(f"match: {len(icsd_labels_match)} * {n_patterns_per_crystal_test}")
print(
    f"match_inorganic: {len(icsd_labels_match_inorganic)} * {n_patterns_per_crystal_test}"
)
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

if scale_patterns:
    sc = StandardScaler(with_mean=False)
    val_x_random = sc.fit_transform(val_x_random)

    with open(out_base + "sc.pickle", "wb") as file:
        pickle.dump(sc, file)

    val_x_all = sc.transform(val_x_all)
    val_x_match = sc.transform(val_x_match)
    val_x_match_inorganic = sc.transform(val_x_match_inorganic)
    val_x_match_correct_spgs = sc.transform(val_x_match_correct_spgs)
    val_x_match_correct_spgs_pure = sc.transform(val_x_match_correct_spgs_pure)

    if generate_randomized_validation_datasets:
        val_x_randomized_coords = sc.transform(val_x_randomized_coords)
        val_x_randomized_ref = sc.transform(val_x_randomized_ref)
        val_x_randomized_lattice = sc.transform(val_x_randomized_lattice)
        val_x_randomized_both = sc.transform(val_x_randomized_both)

val_x_all = np.expand_dims(val_x_all, axis=2)
val_x_match = np.expand_dims(val_x_match, axis=2)
val_x_match_inorganic = np.expand_dims(val_x_match_inorganic, axis=2)
val_x_match_correct_spgs = np.expand_dims(val_x_match_correct_spgs, axis=2)
val_x_match_correct_spgs_pure = np.expand_dims(val_x_match_correct_spgs_pure, axis=2)
val_x_random = np.expand_dims(val_x_random, axis=2)

if generate_randomized_validation_datasets:
    val_x_randomized_coords = np.expand_dims(val_x_randomized_coords, axis=2)
    val_x_randomized_ref = np.expand_dims(val_x_randomized_ref, axis=2)
    val_x_randomized_lattice = np.expand_dims(val_x_randomized_lattice, axis=2)
    val_x_randomized_both = np.expand_dims(val_x_randomized_both, axis=2)

"""
for j in range(0, 100):

    plt.figure()
    for i in range(j * 4, (j + 1) * 4):
        plt.plot(
            angle_range,
            val_x_random[i, :, 0],
            label="Random "
            + str(val_y_random[i])
            + " "
            + str(random_comparison_crystals[i].volume),
        )
    plt.legend()

    plt.figure()
    for i in range(j * 4, (j + 1) * 4):
        plt.plot(
            angle_range,
            val_x_match[i * 5, :, 0],
            label="ICSD "
            + str(val_y_match[i])
            + " "
            + str(icsd_crystals_match[i].volume)
            + " "
            + str(icsd_metas_match[i][0]),
        )
    plt.legend()
    plt.show()
"""

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

    icsd_sim_statistics.load(
        load_only_N_patterns_each=load_only_N_patterns_each_test
        if use_statistics_dataset_as_validation
        else load_only_N_patterns_each_train,
        metas_to_load=[item[0] for item in statistics_metas],
        stop=6 if local else None,
    )  # to not overflow the memory if local

    statistics_icsd_patterns_match = icsd_sim_statistics.sim_patterns
    statistics_icsd_labels_match = icsd_sim_statistics.sim_labels
    statistics_icsd_variations_match = icsd_sim_statistics.sim_variations
    statistics_icsd_crystals_match = icsd_sim_statistics.sim_crystals
    statistics_icsd_metas_match = icsd_sim_statistics.sim_metas

    if (
        True
    ):  # Check for overlaps (in structure prototypes) between the test and statistics dataset
        overlap_counter = 0

        statistics_icsd_metas_match_unpacked = [
            item[0] for item in statistics_icsd_metas_match
        ]
        icsd_metas_match_unpacked = [item[0] for item in icsd_metas_match]

        prototypes_match = [
            icsd_sim_statistics.icsd_structure_types[
                icsd_sim_statistics.icsd_ids.index(meta)
            ]
            for meta in icsd_metas_match_unpacked
        ]

        for meta in statistics_icsd_metas_match_unpacked:

            prototype_statistics = icsd_sim_statistics.icsd_structure_types[
                icsd_sim_statistics.icsd_ids.index(meta)
            ]

            if meta in icsd_metas_match_unpacked or (
                isinstance(prototype_statistics, str)
                and prototype_statistics in prototypes_match
            ):
                overlap_counter += 1

        print(
            f"{overlap_counter} of {len(statistics_icsd_metas_match_unpacked)} prototypes overlapped."
        )

    # Mainly to make the volume constraints correct:
    conventional_errors_counter = 0
    conventional_counter = 0
    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: Calculating conventional structures for statistics dataset...",
        flush=True,
    )
    for i in reversed(range(0, len(statistics_icsd_crystals_match))):

        if statistics_icsd_labels_match[i][0] in spgs:  # speedup

            conventional_counter += 1
            try:
                current_struc = statistics_icsd_crystals_match[i]
                analyzer = SpacegroupAnalyzer(current_struc)
                conv = analyzer.get_conventional_standard_structure()
                statistics_icsd_crystals_match[i] = conv

            except Exception as ex:

                print("Error calculating conventional cell of ICSD (statistics):")
                print(ex)
                conventional_errors_counter += 1

    print(
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}: {conventional_errors_counter} of {conventional_counter} failed to convert to conventional cell (statistics).",
        flush=True,
    )

    for i in reversed(range(0, len(statistics_icsd_patterns_match))):

        if (
            np.any(np.isnan(statistics_icsd_variations_match[i][0]))
            or statistics_icsd_labels_match[i][0] not in spgs
        ):
            del statistics_icsd_patterns_match[i]
            del statistics_icsd_labels_match[i]
            del statistics_icsd_variations_match[i]
            del statistics_icsd_crystals_match[i]
            del statistics_icsd_metas_match[i]

    for i in reversed(range(0, len(statistics_icsd_patterns_match))):
        if validation_max_NO_wyckoffs is not None:
            (
                is_pure,
                NO_wyckoffs,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = icsd_sim_statistics.get_wyckoff_info(statistics_icsd_metas_match[i][0])

        if (
            validation_max_volume is not None
            and statistics_icsd_crystals_match[i].volume > validation_max_volume
        ) or (
            validation_max_NO_wyckoffs is not None
            and NO_wyckoffs > validation_max_NO_wyckoffs
        ):
            del statistics_icsd_patterns_match[i]
            del statistics_icsd_labels_match[i]
            del statistics_icsd_variations_match[i]
            del statistics_icsd_crystals_match[i]
            del statistics_icsd_metas_match[i]

    n_patterns_per_crystal_statistics = len(icsd_sim_statistics.sim_patterns[0])

    statistics_y_match = []
    for i, label in enumerate(statistics_icsd_labels_match):
        statistics_y_match.extend(
            [spgs.index(label[0])] * n_patterns_per_crystal_statistics
        )

    statistics_x_match = []
    for pattern in statistics_icsd_patterns_match:
        for sub_pattern in pattern:
            statistics_x_match.append(sub_pattern)

    statistics_x_match, statistics_y_match = shuffle(
        statistics_x_match, statistics_y_match
    )

    statistics_y_match = np.array(statistics_y_match)
    statistics_x_match = np.expand_dims(statistics_x_match, axis=2)

    print(
        "Size of statistics / training dataset: ", statistics_x_match.shape, flush=True
    )

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
    f"ray cluster resources: {str(ray.cluster_resources())}"
)

with file_writer.as_default():
    tf.summary.text("Parameters", data=params_txt, step=0)

log_wait_timings = []


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if ((epoch + 1) % test_every_X_epochs) == 0:

            start = time.time()

            with file_writer.as_default():

                tf.summary.scalar("queue size", data=queue.size(), step=epoch)

                # gather metric names form model
                metric_names = [metric.name for metric in self.model.metrics]

                # print("########## Metric names:")
                # print(metric_names)

                scores_all = self.model.evaluate(x=val_x_all, y=val_y_all, verbose=0)
                scores_match = self.model.evaluate(
                    x=val_x_match, y=val_y_match, verbose=0
                )
                scores_match_inorganic = self.model.evaluate(
                    x=val_x_match_inorganic, y=val_y_match_inorganic, verbose=0
                )
                scores_match_correct_spgs = self.model.evaluate(
                    x=val_x_match_correct_spgs, y=val_y_match_correct_spgs, verbose=0
                )
                scores_match_correct_spgs_pure = self.model.evaluate(
                    x=val_x_match_correct_spgs_pure,
                    y=val_y_match_correct_spgs_pure,
                    verbose=0,
                )
                scores_random = self.model.evaluate(
                    x=val_x_random, y=val_y_random, verbose=0
                )

                if generate_randomized_validation_datasets:
                    scores_randomized_coords = self.model.evaluate(
                        x=val_x_randomized_coords, y=val_y_randomized_coords, verbose=0
                    )
                    scores_randomized_ref = self.model.evaluate(
                        x=val_x_randomized_ref, y=val_y_randomized_ref, verbose=0
                    )
                    scores_randomized_lattice = self.model.evaluate(
                        x=val_x_randomized_lattice,
                        y=val_y_randomized_lattice,
                        verbose=0,
                    )
                    scores_randomized_both = self.model.evaluate(
                        x=val_x_randomized_both, y=val_y_randomized_both, verbose=0
                    )
                if use_statistics_dataset_as_validation:
                    scores_statistics = self.model.evaluate(
                        x=statistics_x_match, y=statistics_y_match, verbose=0
                    )

                assert metric_names[0] == "loss"

                tf.summary.scalar("loss all", data=scores_all[0], step=epoch)
                tf.summary.scalar("loss match", data=scores_match[0], step=epoch)
                tf.summary.scalar(
                    "loss match_inorganic", data=scores_match_inorganic[0], step=epoch
                )
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
                tf.summary.scalar("accuracy match", data=scores_match[1], step=epoch)
                tf.summary.scalar(
                    "accuracy match_inorganic",
                    data=scores_match_inorganic[1],
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
                tf.summary.scalar("accuracy random", data=scores_random[1], step=epoch)
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
                tf.summary.scalar(
                    "accuracy gap inorganic",
                    data=scores_random[1] - scores_match_inorganic[1],
                    step=epoch,
                )

                tf.summary.scalar("test time", data=time.time() - start, step=epoch)


class CustomSequence(keras.utils.Sequence):
    def __init__(self, number_of_batches):
        self.number_of_batches = number_of_batches

        if use_retention_of_patterns:
            self.patterns = None
            self.labels = None
            self.indices = None

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


sequence = CustomSequence(batches_per_epoch)

if use_retention_of_patterns:
    sequence.pre_compute()

# model = build_model_park(None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate)
# model = build_model_resnet_10(None, N, len(spgs), lr=learning_rate, momentum=momentum, optimizer=optimizer)
# model = build_model_park_tiny_size(None, N, len(spgs), use_dropout=use_dropout)
# model = build_model_resnet_50(None, N, len(spgs), False, lr=learning_rate)
# model = build_model_park_huge_size(None, N, len(spgs), use_dropout=use_dropout)

# model = build_model_transformer(None, N, len(spgs), lr=learning_rate, epochs=NO_epochs, steps_per_epoch=batches_per_epoch)

# model = build_model_transformer_vit(
#    None,
#    N,
#    len(spgs),
#    lr=learning_rate,
#    epochs=NO_epochs,
#    steps_per_epoch=batches_per_epoch,
# )

model = build_model_park_original_spg(
    None, N, len(spgs), use_dropout=use_dropout, lr=learning_rate
)

if use_reduce_lr_on_plateau:
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", verbose=1, factor=0.5
    )

if not use_icsd_structures_directly:
    model.fit(
        x=sequence,
        epochs=NO_epochs,
        # TODO: Removed the batch_size parameter here, any impact?
        callbacks=[tb_callback, CustomCallback()]
        if not use_reduce_lr_on_plateau
        else [tb_callback, CustomCallback(), lr_callback],
        verbose=verbosity,
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
        verbose=verbosity,
        workers=1,
        max_queue_size=queue_size_tf,
        use_multiprocessing=False,
    )

model.save(out_base + "final")

# Get predictions for val_x_match and write rightly_indices / falsely_indices:
prediction_match = model.predict(val_x_match)
prediction_match = np.argmax(prediction_match, axis=1)

rightly_indices_match = np.argwhere(prediction_match == val_y_match)[:, 0]
falsely_indices_match = np.argwhere(prediction_match != val_y_match)[:, 0]

with open(out_base + "rightly_falsely_icsd.pickle", "wb") as file:
    pickle.dump((rightly_indices_match, falsely_indices_match), file)


# Get predictions for val_x_random and write rightly_indices / falsely_indices:
prediction_random = model.predict(val_x_random)
prediction_random = np.argmax(prediction_random, axis=1)

rightly_indices_random = np.argwhere(prediction_random == val_y_random)[:, 0]
falsely_indices_random = np.argwhere(prediction_random != val_y_random)[:, 0]

with open(out_base + "rightly_falsely_random.pickle", "wb") as file:
    pickle.dump((rightly_indices_random, falsely_indices_random), file)

# Get predictions for val_x_randomized and write rightly_indices / falsely_indices:
if generate_randomized_validation_datasets:
    prediction_randomized_coords = model.predict(val_x_randomized_coords)
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
    prediction_randomized_ref = model.predict(val_x_randomized_ref)
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
with open(out_base + "classification_report_match.pickle", "wb") as file:
    pickle.dump(report, file)

report = classification_report(
    [spgs[i] for i in val_y_random],
    [spgs[i] for i in prediction_random],
    output_dict=True,
)
with open(out_base + "classification_report_random.pickle", "wb") as file:
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
