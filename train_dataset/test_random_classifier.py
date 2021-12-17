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
########## Plotting the histogram of spgs in icsd

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

for i in reversed(range(0, len(patterns))):

    # index = sim.icsd_ids.index(sim.sim_metas[i][0])
    # NO_elements = len(sim.icsd_sumformulas[index].split(" "))

    is_pure, NO_wyckoffs = sim.get_wyckoff_info(sim.sim_metas[i][0])
    # print(NO_wyckoffs)

    if (
        np.any(np.isnan(variations[i][0]))
        or labels[i][0] not in ys_unique
        or NO_wyckoffs > 5
        or not is_pure
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

# exit()

counter = [0, 0]

y = []
for label in labels:
    y.extend([ys_unique.index(label[0])] * n_patterns_per_crystal)
    counter[ys_unique.index(label[0])] += 1
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

score, acc = classifier_model.evaluate(
    x, y, batch_size=128
)  # score is the value of the loss function

print("Test score:", score)
print("Test accuracy:", acc)

print()

print("Distribution of the test set (14, 104):")
print(counter)

print()

prob_model = keras.Sequential([classifier_model, keras.layers.Activation("sigmoid")])
predicted_y = np.array(prob_model.predict(x))
predicted_y = predicted_y[:, 0]

predicted_y = np.where(predicted_y > 0.5, 1, 0)

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

print()
print("Classification report:")
print(classification_report(y, predicted_y))
