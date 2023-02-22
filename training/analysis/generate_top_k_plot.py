"""
This script can be used to generate a top-k plot (see our paper) of the specified model.
Please change `model_path` to point to the directory of your run.
"""

from ml4pxrd_tools.manage_dataset import load_dataset_info
import tensorflow.keras as keras
from training.utils.AdamWarmup import AdamWarmup
from training.utils.distributed_utils import map_to_remote
import os
from ml4pxrd_tools.simulation.icsd_simulator import ICSDSimulator
import pickle
import ray
from ml4pxrd_tools.simulation.simulation_smeared import get_synthetic_smeared_patterns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ml4pxrd_tools.matplotlib_defaults

figure_double_width_pub = ml4pxrd_tools.matplotlib_defaults.pub_width

model_path = "/home/ws/uvgnh/MSc/ML4pXRDs/training/classifier_spgs/07-01-2023_18-31-34/final"  # ResNet-101 sqrt-scaling
preprocess_patterns_sqrt = (
    True  # Whether or not the model uses the sqrt-preprocessing of the patterns
)

(
    (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg,
        denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type,
        per_element,
        represented_spgs,
        probability_per_spg,
    ),
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

with open(os.path.dirname(model_path) + "/spgs.pickle", "rb") as file:
    spgs = pickle.load(file)

model = keras.models.load_model(model_path, custom_objects={"AdamWarmup": AdamWarmup})

path_to_icsd_directory_local = os.path.expanduser("~/Dokumente/Big_Files/ICSD/")
path_to_icsd_directory_cluster = os.path.expanduser("~/Databases/ICSD/")

##### load match dataset

jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    path_to_icsd_directory = path_to_icsd_directory_cluster
else:
    path_to_icsd_directory = path_to_icsd_directory_local

icsd_sim_test = ICSDSimulator(
    os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
    os.path.join(path_to_icsd_directory, "cif/"),
)

##### Load ICSD test data:

test_match_metas_flat = [item[0] for item in test_match_metas]
test_metas_flat = [item[0] for item in test_metas]
metas_to_load_test = []
for i, meta in enumerate(test_metas_flat):
    if test_labels[i][0] in spgs:
        metas_to_load_test.append(meta)

icsd_sim_test.load(
    load_only_N_patterns_each=1,
    metas_to_load=metas_to_load_test,  # Only load the patterns with the ICSD ids from the test dataset
)

icsd_patterns_match = []
icsd_labels_match = []

for i in range(len(icsd_sim_test.sim_crystals)):

    if icsd_sim_test.sim_metas[i][0] in test_match_metas_flat:

        if icsd_sim_test.sim_labels[i][0] in spgs:
            icsd_patterns_match.append(icsd_sim_test.sim_patterns[i])
            icsd_labels_match.append(icsd_sim_test.sim_labels[i])

##### Construct x and y numpy arrays for ICSD test dataset

val_y_match = []
for i, label in enumerate(icsd_labels_match):
    val_y_match.extend([spgs.index(label[0])] * 1)

val_x_match = []
for pattern in icsd_patterns_match:
    for sub_pattern in pattern:
        val_x_match.append(sub_pattern)

val_y_match = np.array(val_y_match)
if preprocess_patterns_sqrt:
    val_x_match = np.sqrt(val_x_match)
val_x_match = np.expand_dims(val_x_match, axis=2)

##### Need to generate a small dataset with random patterns to apply the model to:


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

    patterns, labels, structures, corn_sizes = get_synthetic_smeared_patterns(
        spgs=spgs,
        N_structures_per_spg=structures_per_spg,
        wavelength=1.5406,
        N=N,
        NO_corn_sizes=NO_corn_sizes,
        two_theta_range=(start_angle, end_angle),
        max_NO_atoms_asymmetric_unit=max_NO_elements,
        return_structures_and_corn_sizes=True,
        probability_per_spg_per_element=probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
        max_volume=7000,
        do_symmetry_checks=True,
        NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg=denseness_factors_density_per_spg,
        denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
        per_element=per_element,
        is_verbose=False,
        probability_per_spg=probability_per_spg,
    )

    # Set the label to the right index:
    for i in range(0, len(labels)):
        labels[i] = spgs.index(labels[i])

    patterns = np.array(patterns)
    labels = np.array(labels)

    return patterns, labels, structures, corn_sizes


scheduler_fn = lambda input: batch_generator_with_additional.remote(
    spgs,
    1,
    8501,
    5,
    90,
    100,
    1,
)
results = map_to_remote(
    scheduler_fn=scheduler_fn,
    inputs=range(100),
    NO_workers=16,
)

val_x_random = []
val_y_random = []

for result in results:
    patterns, labels, crystals, corn_sizes = result

    val_x_random.extend(patterns)
    val_y_random.extend(labels)

val_y_random = np.array(val_y_random)
if preprocess_patterns_sqrt:
    val_x_random = np.sqrt(val_x_random)
val_x_random = np.expand_dims(val_x_random, axis=2)

# Calculate top-k accuracy


def get_top_k_accuracies(true_labels, predictions, ks):

    top_k_accuracies = []

    for k in ks:
        m = tf.keras.metrics.TopKCategoricalAccuracy(k=k)
        m.update_state(tf.one_hot(true_labels, depth=len(spgs)), predictions)
        top_k_accuracies.append(m.result().numpy())

    return top_k_accuracies


prediction_match = model.predict(val_x_match, batch_size=145)

accs = get_top_k_accuracies(val_y_match, prediction_match, range(1, 21))

prediction_match = np.argmax(prediction_match, axis=1)

print("Match")
print(accs)

plt.figure(
    figsize=(
        figure_double_width_pub * 0.95 * 0.5,
        figure_double_width_pub * 0.7 * 0.5,
    )
)
plt.plot(list(range(1, 21)), accs, label="ICSD")
plt.xlabel("k")
plt.ylabel("Accuracy")

#####

prediction_random = model.predict(val_x_random, batch_size=145)

accs = get_top_k_accuracies(val_y_random, prediction_random, range(1, 21))

prediction_random = np.argmax(prediction_random, axis=1)

print("Random")
print(accs)

plt.plot(list(range(1, 21)), accs, label="Synthetic (training)")

plt.legend()

plt.tight_layout()
plt.savefig(
    "top_k.pdf",
    bbox_inches="tight",
)

"""
# Where do the wrongly classified patterns land?

prediction_match = np.array(prediction_match)
val_y_match = np.array(val_y_match)

wrongs_indices_match = np.argwhere(prediction_match != val_y_match)
wrongly_predicted_as_match = prediction_match[wrongs_indices_match][:, 0]
wrongly_predicted_as_match = [spgs[index] for index in wrongly_predicted_as_match]

prediction_random = np.array(prediction_random)
val_y_random = np.array(val_y_random)

wrongs_indices_random = np.argwhere(prediction_random != val_y_random)
wrongly_predicted_as_random = prediction_random[wrongs_indices_random][:, 0]
wrongly_predicted_as_random = [spgs[index] for index in wrongly_predicted_as_random]

plt.figure()
plt.hist(
    wrongly_predicted_as_match, alpha=0.5, bins=np.arange(0, 231) + 0.5, label="ICSD"
)
plt.hist(
    wrongly_predicted_as_random,
    alpha=0.5,
    bins=np.arange(0, 231) + 0.5,
    label="Synthetic",
)
plt.xlabel("Space group")
plt.ylabel("Count")
plt.legend()
plt.savefig("wrongly_predicted_as.pdf")
"""
