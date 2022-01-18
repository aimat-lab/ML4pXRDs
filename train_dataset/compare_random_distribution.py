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
import re

in_base = "classifier_spgs/18-01-2022_10-32-34_just_test_it/"

with open(in_base + "spgs.pickle", "rb") as file:
    spgs = pickle.load(file)

with open(in_base + "icsd_data.pickle", "rb") as file:
    icsd_crystals, icsd_labels, icsd_variations, icsd_metas = pickle.load(file)

with open(in_base + "random_data.pickle", "rb") as file:
    (
        random_crystals,
        random_labels,
        random_variations,
    ) = pickle.load(file)
random_variations_flat = []
for item in random_variations:
    random_variations_flat.extend(item)

with open(in_base + "rightly_falsely.pickle", "rb") as file:
    rightly_indices, falsely_indices = pickle.load(file)

# Get infos from icsd crystals:

icsd_NO_wyckoffs = []
icsd_elements = []
icsd_NO_elements = []
icsd_occupancies = []
icsd_element_repetitions = []

# Just for the icsd meta-data (ids):
jobid = os.getenv("SLURM_JOB_ID")
if jobid is not None and jobid != "":
    sim = Simulation(
        "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/ws/uvgnh/Databases/ICSD/cif/",
    )
else:
    sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

for i in range(0, len(icsd_variations)):

    is_pure, NO_wyckoffs, elements, occupancies = sim.get_wyckoff_info(icsd_metas[i][0])

    elements_unique = np.unique(elements)

    icsd_NO_wyckoffs.append(NO_wyckoffs)
    icsd_elements.append(elements)
    icsd_NO_elements.append(len(elements_unique))
    icsd_occupancies.append(occupancies)

    reps = []
    for el in elements_unique:
        reps.append(np.sum(np.array(elements) == el))
    icsd_element_repetitions.append(reps)


# preprocess random data:


def get_wyckoff_info(crystal):
    # returns: Number of set wyckoffs, elements

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


random_NO_wyckoffs = []
random_elements = []
random_NO_elements = []
random_element_repetitions = []

for i in range(0, len(random_variations)):

    print(f"Processing random: {i} of {len(random_variations)}")

    success = True
    try:
        NO_wyckoffs, elements = get_wyckoff_info(random_crystals[i])
    except Exception as ex:
        print(ex)
        success = False

    if success:

        elements_unique = np.unique(elements)

        random_NO_wyckoffs.append(NO_wyckoffs)
        random_elements.append(elements)
        random_NO_elements.append(len(elements_unique))

        reps = []
        for el in elements_unique:
            reps.append(np.sum(np.array(elements) == el))
        random_element_repetitions.extend(reps)

############## Calculate histograms:


def get_denseness_factor(structure):

    try:

        actual_volume = structure.volume

        calculated_volume = 0
        for atom in structure:
            specie = atom.species_string
            specie = re.sub(r"\d*\+?$", "", specie)
            specie = re.sub(r"\d*\-?$", "", specie)

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
falsely_occupancies = []
falsely_element_repetitions = []

rightly_volumes = []
rightly_denseness_factors = []
rightly_lattice_paras = []
rightly_corn_sizes = []
rightly_NO_wyckoffs = []
rightly_NO_elements = []
rightly_occupancies = []
rightly_element_repetitions = []

random_volumes = []
random_denseness_factors = []
random_lattice_paras = []

for i in falsely_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    falsely_volumes.append(volume)
    falsely_corn_sizes.extend(icsd_variations[index])
    falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    falsely_NO_elements.append(icsd_NO_wyckoffs[index])
    falsely_occupancies.extend(icsd_occupancies[index])
    falsely_element_repetitions.extend(icsd_element_repetitions[index])

    falsely_lattice_paras.append(structure.lattice.a)
    falsely_lattice_paras.append(structure.lattice.b)
    falsely_lattice_paras.append(structure.lattice.c)

    if denseness_factor is not None:
        falsely_denseness_factors.append(denseness_factor)

for i in rightly_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    rightly_volumes.append(volume)
    rightly_corn_sizes.extend(icsd_variations[index])
    rightly_NO_elements.append(icsd_NO_elements[index])
    rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    rightly_occupancies.extend(icsd_occupancies[index])
    rightly_element_repetitions.extend(icsd_element_repetitions[index])

    rightly_lattice_paras.append(structure.lattice.a)
    rightly_lattice_paras.append(structure.lattice.b)
    rightly_lattice_paras.append(structure.lattice.c)

    if denseness_factor is not None:
        rightly_denseness_factors.append(denseness_factor)

for i, structure in enumerate(random_crystals):

    volume = structure.volume
    denseness_factor = get_denseness_factor(structure)

    random_volumes.append(volume)

    random_lattice_paras.append(structure.lattice.a)
    random_lattice_paras.append(structure.lattice.b)
    random_lattice_paras.append(structure.lattice.c)

    if denseness_factor is not None:
        random_denseness_factors.append(denseness_factor)

# plot volumes:
bins_volumes = np.linspace(
    min(np.min(rightly_volumes), np.min(falsely_volumes), np.min(random_volumes)),
    max(np.max(rightly_volumes), np.max(falsely_volumes), np.max(random_volumes)),
    200,
)
plt.figure()
plt.hist(
    [rightly_volumes, falsely_volumes, random_volumes],
    bins_volumes,
    label=["rightly", "falsely", "random"],
)
plt.legend(loc="upper right")
plt.xlabel("volume")
plt.ylabel("count")
plt.savefig("comparison_volumes.png", dpi=400)
plt.show()

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
)
plt.legend(loc="upper right")
plt.xlabel("denseness factor")
plt.ylabel("count")
plt.savefig("comparison_denseness_factors.png", dpi=400)
plt.show()

# plot corn sizes:
bins_corn_sizes = np.linspace(
    min(
        np.min(rightly_corn_sizes),
        np.min(falsely_corn_sizes),
        np.min(random_variations_flat),
    ),
    max(
        np.max(rightly_corn_sizes),
        np.max(falsely_corn_sizes),
        np.max(random_variations_flat),
    ),
    30,
)
plt.figure()
plt.hist(
    [rightly_corn_sizes, falsely_corn_sizes, random_variations_flat],
    bins_corn_sizes,
    label=["rightly", "falsely", "random"],
)
plt.legend(loc="upper right")
plt.xlabel("corn size")
plt.ylabel("count")
plt.savefig("comparison_corn_sizes.png", dpi=400)
plt.show()

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
    label=["rightly", "falsely", "random"],
)
plt.legend(loc="upper right")
plt.xlabel("Number of set wyckoff sites")
plt.ylabel("count")
plt.savefig("NO_wyckoffs.png", dpi=400)
plt.show()

# plot NO_elements (unique number of elements on wyckoff sites):
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
    label=["rightly", "falsely", "random"],
)
plt.legend()
plt.xlabel("Number of unique elements on wyckoff sites")
plt.ylabel("count")
plt.savefig("NO_elements.png", dpi=400)
plt.show()

# plot lattice_paras:
bins_lattice_paras = np.linspace(
    min(
        np.min(rightly_lattice_paras),
        np.min(falsely_lattice_paras),
        np.min(random_lattice_paras),
    ),
    max(
        np.max(rightly_lattice_paras),
        np.max(falsely_lattice_paras),
        np.max(random_lattice_paras),
    ),
    30,
)
plt.figure()
plt.hist(
    [rightly_lattice_paras, falsely_lattice_paras, random_lattice_paras],
    bins_lattice_paras,
    label=["rightly", "falsely", "random"],
)
plt.legend(loc="upper right")
plt.xlabel("lattice para")
plt.ylabel("count")
plt.savefig("comparison_lattice_paras.png", dpi=400)
plt.show()

# plot occupancies:
bins_occupancies = np.linspace(
    min(
        np.min(rightly_occupancies),
        np.min(falsely_occupancies),
    ),
    max(
        np.max(rightly_occupancies),
        np.max(falsely_occupancies),
    ),
    30,
)
plt.figure()
plt.hist(
    [rightly_occupancies, falsely_occupancies],
    bins_occupancies,
    label=["rightly", "falsely"],
)
plt.legend(loc="upper right")
plt.xlabel("occupancy")
plt.ylabel("count")
plt.savefig("comparison_occupancies.png", dpi=400)
plt.show()

# plot number of element repetitions:
bins_element_repetitions = (
    np.arange(
        min(
            np.min(rightly_element_repetitions),
            np.min(falsely_element_repetitions),
            np.min(random_element_repetitions),
        ),
        max(
            np.max(rightly_element_repetitions),
            np.max(falsely_element_repetitions),
            np.max(random_element_repetitions),
        )
        + 1,
    )
    + 0.5
)
plt.figure()
plt.hist(
    [
        rightly_element_repetitions,
        falsely_element_repetitions,
        random_element_repetitions,
    ],
    bins=bins_element_repetitions,
    label=["rightly", "falsely", "random"],
)
plt.legend()
plt.xlabel("Number of element repetitions on wyckoff sites")
plt.ylabel("count")
plt.savefig("NO_element_repetitions.png", dpi=400)
plt.show()


# Info about wyckoff positions in cif file format:
# => where in the cif file is written what kind of wyckoff site we are dealing with?
# The general wyckoff site is always written in the cif file!
# This is because the special ones are only special cases of the general wyckoff position!
# Only the general wyckoff position is needed to generate all the coordinates.
