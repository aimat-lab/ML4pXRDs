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
from pyxtal import pyxtal
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group

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
icsd_sim.load(load_patterns_angles_intensities=False, load_only=14)

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
random_sim.output_dir = "../dataset_simulations/patterns/random_crystals_only/"
random_sim.load(load_patterns_angles_intensities=False, load_only=1)

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
icsd_NO_elements = []
icsd_occupancies = []

for i in reversed(range(0, len(icsd_variations))):
    is_pure, NO_wyckoffs, elements, occupancies = icsd_sim.get_wyckoff_info(
        icsd_sim.sim_metas[i][0]
    )

    if np.any(np.isnan(icsd_variations[i][0])) or icsd_labels[i][0] not in ys_unique:
        del icsd_labels[i]
        del icsd_variations[i]
        del icsd_crystals[i]
    else:
        icsd_NO_wyckoffs.append(NO_wyckoffs)
        icsd_elements.append(elements)
        icsd_NO_elements.append(len(np.unique(elements)))
        icsd_occupancies.append(occupancies)

icsd_NO_wyckoffs = list(reversed(icsd_NO_wyckoffs))
icsd_elements = list(reversed(icsd_elements))
icsd_occupancies = list(reversed(icsd_occupancies))

icsd_corn_sizes = []
for i, label in enumerate(icsd_labels):
    icsd_corn_sizes.append([item[0] for item in icsd_sim.sim_variations[i]])


# preprocess random data:

random_NO_wyckoffs = []
random_elements = []


def get_wyckoff_info(crystal):

    # info = crystal.get_space_group_info()

    # cif_writer = CifWriter(crystal)
    # cif_writer.write_file("cif_test.cif")

    struc = pyxtal()
    struc.from_seed(crystal)

    # pymatgen_s = struc.to_pymatgen()
    # pymatgen_p = pymatgen_s.get_primitive_structure()

    # TODO: What is wrong here? Why is this not the primitive structure???

    # test = struc.get_alternatives()
    # group = Group(139, dim=3)

    elements = []

    for site in struc.atom_sites:
        specie_str = str(site.specie)
        elements.append(specie_str)

    return len(struc.atom_sites), elements


random_NO_elements = []

for i in reversed(range(0, len(random_variations))):

    print(f"{i} of {len(random_variations)}")

    success = True
    try:
        NO_wyckoffs, elements = get_wyckoff_info(random_crystals[i])
    except Exception as ex:
        print(ex)
        success = False

    if (
        not success
        or np.any(np.isnan(random_variations[i][0]))
        or random_labels[i][0] not in ys_unique
    ):
        del random_labels[i]
        del random_variations[i]
        del random_crystals[i]
    else:
        random_NO_wyckoffs.append(NO_wyckoffs)
        random_elements.append(elements)
        random_NO_elements.append(len(np.unique(elements)))

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
falsely_lattice_paras = []
falsely_corn_sizes = []
falsely_NO_wyckoffs = []
falsely_NO_elements = []

rightly_volumes = []
rightly_denseness_factors = []
rightly_lattice_paras = []
rightly_corn_sizes = []
rightly_NO_wyckoffs = []
rightly_NO_elements = []

random_volumes = []
random_denseness_factors = []
random_lattice_paras = []

for i in falsely_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    falsely_volumes.append(volume)
    falsely_corn_sizes.extend(icsd_corn_sizes[index])
    falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    falsely_NO_elements.append(icsd_NO_wyckoffs[index])

    if denseness_factor is not None:
        falsely_denseness_factors.append(denseness_factor)

for i in rightly_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    rightly_volumes.append(volume)
    rightly_corn_sizes.extend(icsd_corn_sizes[index])
    rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])

    if denseness_factor is not None:
        rightly_denseness_factors.append(denseness_factor)

for i, structure in random_crystals:

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    random_volumes.append(volume)

    if denseness_factor is not None:
        random_denseness_factors.append(denseness_factor)

# plot volumes:
bins_volumes = np.linspace(
    min(np.min(rightly_volumes), np.min(falsely_volumes), np.min(random_volumes)),
    max(np.max(rightly_volumes), np.max(falsely_volumes), np.max(random_volumes)),
    30,
)
plt.figure()
plt.hist(
    [rightly_volumes, falsely_volumes, random_volumes],
    bins_volumes,
    label=["rightly", "falsely", "random"],
    alpha=0.5,
)
plt.legend(loc="upper right")
plt.show()
plt.savefig("comparison_volumes.png")

# plot denseness factors:
bins_denseness_factors = np.linspace(
    min(
        np.min(rightly_denseness_factors),
        np.min(falsely_denseness_factors),
        np.min(random_denseness_factors),
    ),
    max(
        np.max(rightly_denseness_factors),
        np.max(falsely_denseness_factors),
        np.max(random_denseness_factors),
    ),
    30,
)
plt.figure()
plt.hist(
    [rightly_denseness_factors, falsely_denseness_factors, random_denseness_factors],
    bins_denseness_factors,
    label=["rightly", "falsely", "random"],
    alpha=0.5,
)
plt.legend(loc="upper right")
plt.show()
plt.savefig("comparison_denseness_factors.png")

# plot corn sizes:
bins_corn_sizes = np.linspace(
    min(
        np.min(rightly_corn_sizes),
        np.min(falsely_corn_sizes),
        np.min(random_corn_sizes),
    ),
    max(
        np.max(rightly_corn_sizes),
        np.max(falsely_corn_sizes),
        np.max(random_corn_sizes),
    ),
    30,
)
plt.figure()
plt.hist(
    [rightly_corn_sizes, falsely_corn_sizes, random_corn_sizes],
    bins_corn_sizes,
    label=["rightly", "falsely", "random"],
    alpha=0.5,
)
plt.legend(loc="upper right")
plt.show()
plt.savefig("comparison_corn_sizes.png")

# plot NO_wyckoffs:
bins_NO_wyckoffs = (
    np.arange(
        min(
            np.min(rightly_NO_wyckoffs),
            np.min(falsely_NO_wyckoffs),
            np.min(random_NO_wyckoffs),
        ),
        max(
            np.max(rightly_NO_wyckoffs),
            np.max(falsely_NO_wyckoffs),
            np.max(random_NO_wyckoffs),
        )
        + 1,
    )
    + 0.5
)
plt.figure()
plt.hist(
    [rightly_NO_wyckoffs, falsely_NO_wyckoffs, random_NO_wyckoffs],
    bins=bins_NO_wyckoffs,
)
plt.xlabel("Number of set wyckoff sites")
plt.savefig("NO_wyckoffs.png")
plt.show()

# plot NO_elements:
bins_NO_elements = (
    np.arange(
        min(
            np.min(rightly_NO_elements),
            np.min(falsely_NO_elements),
            np.min(random_NO_elements),
        ),
        max(
            np.max(rightly_NO_elements),
            np.max(falsely_NO_elements),
            np.max(random_NO_elements),
        )
        + 1,
    )
    + 0.5
)
plt.figure()
plt.hist(
    [rightly_NO_elements, falsely_NO_elements, random_NO_elements],
    bins=bins_NO_elements,
)
plt.xlabel("Number of unique elements on wyckoff sites")
plt.savefig("NO_elements.png")
plt.show()

# TODO:
# get number of unique elements (unique!) on wyckoff sites # TODO: make sure that for random AND unique it is actually unique!
# Use the second column, remove +, -, 1+, 1-, etc.

# get lattice parameters => hist (all in the same histogram), get them above in a list, too!
# get number of repetitions of element (on different wyckoff sites)
# get occupancies => hist all of them (only for the icsd data)

# TODO: Why VOLUMES OF 10**3???

# => where in the cif file is written what kind of wyckoff site we are dealing with?
# The general wyckoff site is always written in the cif file!
# This is because the special ones are only special cases of the general wyckoff position!
# Only the general wyckoff position is needed to generate all the coordinates.
