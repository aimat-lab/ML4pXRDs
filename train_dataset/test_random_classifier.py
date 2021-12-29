import sys

sys.path.append("../dataset_simulations")
sys.path.append("./")
sys.path.append("../")

import tensorflow.keras as keras
import os
from dataset_simulations.random_simulation import Simulation
import numpy as np
import pickle
import gc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# classifier_model_name = "random_25-11-2021_12:09:51_test"
classifier_model_name = "random_27-11-2021_09:12:22_test"
classifier_model = keras.models.load_model(
    "classifier/" + classifier_model_name + "/final"
)
is_conv_model = True

number_of_values_initial = 9018
simulated_range = np.linspace(0, 90, number_of_values_initial)
# only use a restricted range of the simulated patterns
start_x = 10
end_x = 90  # different from above
step = 1
start_index = np.argwhere(simulated_range >= start_x)[0][0]
end_index = np.argwhere(simulated_range <= end_x)[-1][0]
used_range = simulated_range[start_index : end_index + 1 : step]
number_of_values = len(used_range)

scale_features = True

# Patterns to test the model on:
path_to_patterns = "../dataset_simulations/patterns/icsd/"

jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    sim = Simulation(
        "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/kit/iti/la2559/Databases/ICSD/cif/",
    )
    sim.output_dir = path_to_patterns
else:
    sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
    sim.output_dir = path_to_patterns

sim.load()

n_patterns_per_crystal = len(sim.sim_patterns[0])

patterns = sim.sim_patterns
labels = sim.sim_labels
variations = sim.sim_variations

"""
########## Plotting the histogram of spgs in the ICSD

spgs = [sim.get_space_group_number(id) for id in sim.icsd_ids]

for i in reversed(range(0, len(spgs))):
    if spgs[i] is None:
        del spgs[i]

print(f"Number of ICSD entries with spg number: {len(spgs)}")

plt.figure()
plt.hist(spgs, bins=np.arange(1, 231) + 0.5)
plt.xlabel("International space group number")
plt.savefig("distribution_spgs.png")
# plt.show()

########## Plotting the histogram of spgs in the simulation data

spgs = [label[0] for label in labels]
# spgs_compare = [sim.get_space_group_number(meta[0]) for meta in sim.sim_metas]

plt.figure()
plt.hist(spgs, bins=np.arange(1, 231) + 0.5)
plt.xlabel("International space group number")
plt.savefig("distribution_spgs.png")
# plt.show()

########## Plotting the histogram number of elements in icsd

lengths = []
for i, id in enumerate(sim.icsd_sumformulas):
    lengths.append(len(id.split(" ")))

plt.figure()
plt.hist(lengths, bins=np.arange(0, np.max(lengths)) + 0.5)
plt.xlabel("Number of elements")
plt.savefig("distribution_NO_elements.png")
# plt.show()
"""

# the space groups to test for:
ys_unique = [14, 104]

# counter_14 = 0
# counter_104 = 0

__wyckoff_strs = []

for i in reversed(range(0, len(patterns))):

    # index = sim.icsd_ids.index(sim.sim_metas[i][0])
    # NO_elements = len(sim.icsd_sumformulas[index].split(" "))

    is_pure, NO_wyckoffs, wyckoff_str = sim.get_wyckoff_info(sim.sim_metas[i][0])
    # print(NO_wyckoffs)

    if (
        np.any(np.isnan(variations[i][0]))
        or labels[i][0] not in ys_unique
        # or NO_wyckoffs > 5
        # or not is_pure
    ):
        del patterns[i]
        del labels[i]
        del variations[i]
    #    elif labels[i][0] == 14:
    #        if counter_14 < 10:
    #            plt.figure()
    #            plt.plot(patterns[i][0])
    #            plt.savefig(f"14_{i}.pdf")
    #            counter_14 += 1
    #    elif labels[i][0] == 104:
    #        if counter_104 < 10:
    #            plt.figure()
    #            plt.plot(patterns[i][0])
    #            plt.savefig(f"104_{i}.pdf")
    #            counter_104 += 1
    else:
        __wyckoff_strs.append(wyckoff_str)
# exit()

__wyckoff_strs = list(reversed(__wyckoff_strs))

wyckoff_strs = []
for wyckoff_str in __wyckoff_strs:
    wyckoff_strs.extend([wyckoff_str] * 5)

counter = [0, 0]

y = []
corn_sizes = []

for i, label in enumerate(labels):
    y.extend([ys_unique.index(label[0])] * n_patterns_per_crystal)
    counter[ys_unique.index(label[0])] += 1
    corn_sizes.extend([item[0] for item in sim.sim_variations[i]])

y = np.array(y)

x_unscaled = []
for pattern in patterns:
    for sub_pattern in pattern:
        x_unscaled.append(sub_pattern[start_index : end_index + 1 : step])

assert not np.any(np.isnan(x_unscaled))
assert not np.any(np.isnan(y))
assert len(x_unscaled) == len(y)

print("##### Loaded {} training points with {} classes".format(len(x_unscaled), 2))

if scale_features:

    with open("classifier/" + classifier_model_name + "/scaler", "rb") as file:
        sc = pickle.load(file)

    x = sc.transform(x_unscaled)

    del x_unscaled[:]
    del x_unscaled
    gc.collect()

if is_conv_model:
    x = np.expand_dims(x, axis=2)

x = x.astype(np.float32)
y = y.astype(np.float32)

score, acc = classifier_model.evaluate(
    x, y, batch_size=x.shape[0]
)  # score is the value of the loss function

print("Test score:", score)
print("Test accuracy:", acc)

print()

print("Distribution of the test set (14, 104):")
print(counter)

print()

prob_model = keras.Sequential([classifier_model, keras.layers.Activation("sigmoid")])
# prob_model = classifier_model
predicted_y = np.array(prob_model.predict(x, batch_size=x.shape[0]))
# predicted_y = np.array(prob_model(x))

predicted_y = predicted_y[:, 0]
predicted_y = np.where(predicted_y > 0.5, 1.0, 0.0)

predicted_y = predicted_y.astype(np.float32)

print()
print("Predicted as spg 14:")
print(np.sum(predicted_y == 0))

print("Predicted as spg 104:")
print(np.sum(predicted_y == 1))

print()
print("Correctly predicted as spg 14:")
print(np.sum((predicted_y == 0) & (predicted_y == y)))
print("Correctly predicted as spg 104:")
print(np.sum((predicted_y == 1) & (predicted_y == y)))

print()
print(
    f"Correctly classified: {np.sum(predicted_y == y)} ({np.sum(predicted_y == y) / len(y)} %)"
)

# falsely classified:
falsely_indices = np.argwhere(predicted_y != y)[:, 0]
rightly_indices = np.argwhere(predicted_y == y)[:, 0]

print(len(wyckoff_strs))
print(len(predicted_y))
print(len(y))

""" Analyze dependence on corn sizes
wrong_cornsizes = []
for i in falsely_indices:
    print(wyckoff_strs[i])
    print()

    wrong_cornsizes.append(corn_sizes[i])

plt.figure()
plt.hist(wrong_cornsizes)
plt.show()
"""

print()
print("Classification report:")
print(classification_report(y, predicted_y))
