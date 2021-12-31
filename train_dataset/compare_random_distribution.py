import sys

sys.path.append("../dataset_simulations")
sys.path.append("./")
sys.path.append("../")

import os
from dataset_simulations.random_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from pyxtal.database.element import Element
import pickle

jobid = os.getenv("SLURM_JOB_ID")

# read ICSD data:
if jobid is not None and jobid != "":
    icsd_sim = Simulation(
        "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/kit/iti/la2559/Databases/ICSD/cif/",
    )
else:
    icsd_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
icsd_sim.output_dir = "../dataset_simulations/patterns/icsd/"
icsd_sim.load(load_patterns_angles_intensities=False)

# read random data:
if jobid is not None and jobid != "":
    random_sim = Simulation(
        "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/kit/iti/la2559/Databases/ICSD/cif/",
    )
else:
    random_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )
random_sim.output_dir = "../dataset_simulations/patterns/icsd/"
random_sim.load(load_patterns_angles_intensities=False)

n_patterns_per_crystal = 5

with open("falsely_rightly.pickle", "rb") as file:
    falsely_indices, rightly_indices = pickle.load(file)

icsd_variations = icsd_sim.sim_variations
icsd_crystals = icsd_sim.sim_crystals
icsd_labels = icsd_sim.sim_labels

random_variations = random_sim.sim_variations
random_crystals = random_sim.sim_crystals
random_labels = random_sim.sim_labels

"""
########## Plotting the histogram of spgs in the ICSD

spgs = [icsd_sim.get_space_group_number(id) for id in icsd_sim.icsd_ids]

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
# spgs_compare = [icsd_sim.get_space_group_number(meta[0]) for meta in icsd_sim.sim_metas]

plt.figure()
plt.hist(spgs, bins=np.arange(1, 231) + 0.5)
plt.xlabel("International space group number")
plt.savefig("distribution_spgs.png")
# plt.show()

########## Plotting the histogram of number of elements in icsd

lengths = []
for i, id in enumerate(icsd_sim.icsd_sumformulas):
    lengths.append(len(id.split(" ")))

plt.figure()
plt.hist(lengths, bins=np.arange(0, np.max(lengths)) + 0.5)
plt.xlabel("Number of elements")
plt.savefig("distribution_NO_elements.png")
# plt.show()
"""

# the space groups to test for:
ys_unique = [14, 104]


# preprocess icsd-data:

icsd_NO_wyckoffs = []
icsd_elements = []

for i in reversed(range(0, len(icsd_variations))):
    is_pure, NO_wyckoffs, wyckoff_str, elements = icsd_sim.get_wyckoff_info(
        icsd_sim.sim_metas[i][0]
    )

    if np.any(np.isnan(icsd_variations[i][0])) or icsd_labels[i][0] not in ys_unique:
        del icsd_labels[i]
        del icsd_variations[i]
        del icsd_crystals[i]
    else:
        icsd_NO_wyckoffs.append(NO_wyckoffs)
        icsd_elements.append(elements)

icsd_NO_wyckoffs = list(reversed(icsd_NO_wyckoffs))
icsd_elements = list(reversed(icsd_elements))

icsd_corn_sizes = []
for i, label in enumerate(icsd_labels):
    icsd_corn_sizes.extend([item[0] for item in icsd_sim.sim_variations[i]])


# preprocess random data:

random_NO_wyckoffs = []
random_elements = []


def get_wyckoff_info(crystal):
    print()


for i in reversed(range(0, len(random_variations))):

    is_pure, NO_wyckoffs, wyckoff_str, elements = get_wyckoff_info(random_crystals[i])

    if (
        np.any(np.isnan(random_variations[i][0]))
        or random_labels[i][0] not in ys_unique
    ):
        del random_labels[i]
        del random_variations[i]
        del random_crystals[i]
    else:
        random_NO_wyckoffs.append(NO_wyckoffs)
        random_elements.append(elements)

random_NO_wyckoffs = list(reversed(random_NO_wyckoffs))
random_elements = list(reversed(random_elements))

random_corn_sizes = []
for i, label in enumerate(random_labels):
    random_corn_sizes.extend([item[0] for item in random_sim.sim_variations[i]])

# calculate histograms:


def get_denseness_factor(structure):

    try:

        actual_volume = structure.volume

        calculated_volume = 0
        for atom in structure:
            specie = str(atom.specie.element)
            r = (Element(specie).covalent_radius + Element(specie).vdw_radius) / 2
            calculated_volume += 4 / 3 * np.pi * r ** 3

        return actual_volume / calculated_volume

    except:
        return None


falsely_volumes = []
falsely_denseness_factors = []

rightly_volumes = []
rightly_denseness_factors = []

for i in falsely_indices:

    structure = crystals[int(i / 5)]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    falsely_volumes.append(volume)

    if denseness_factor is not None:
        falsely_denseness_factors.append(denseness_factor)

for i in rightly_indices:

    structure = crystals[int(i / 5)]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    rightly_volumes.append(volume)

    if denseness_factor is not None:
        rightly_denseness_factors.append(denseness_factor)

# plot volumes:
bins_volumes = np.linspace(
    min(np.min(rightly_volumes), np.min(falsely_volumes)),
    max(np.max(rightly_volumes), np.max(falsely_volumes)),
    30,
)
plt.hist(
    [rightly_volumes, falsely_volumes],
    bins_volumes,
    label=["rightly", "falsely"],
    alpha=0.5,
)
plt.legend(loc="upper right")
plt.show()

# plot denseness factors:
bins_denseness_factors = np.linspace(
    min(np.min(rightly_denseness_factors), np.min(falsely_denseness_factors)),
    max(np.max(rightly_denseness_factors), np.max(falsely_denseness_factors)),
    30,
)
plt.hist(
    [rightly_denseness_factors, falsely_denseness_factors],
    bins_denseness_factors,
    label=["rightly", "falsely"],
    alpha=0.5,
)
plt.legend(loc="upper right")
plt.show()

# TODO:
# Create separate script for this, remove stuff from each
# Make a function out of this! Also with crystals as arguments (and tag), so it can be reused for random simulation, too!

# also add corn sizes, here!
# get lattice parameters => hist (all in the same histogram)
# get number of wyckoff sites => hist
# get number of elements => hist
# get number of repetitions of element (are these then different wyckoff sites?)
# => where in the cif file is written what kind of wyckoff site we are dealing with?

# get occupancies => hist all of them

# VOLUMES OF 10**3???
