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
import random

# in_base = "classifier_spgs/runs_from_cluster/spgs-2-15-batch-size-100/"
# in_base = "classifier_spgs/runs_from_cluster/4-spg-1000-epochs/"
# in_base = "classifier_spgs/runs_from_cluster/4-spgs-new_generation/"
in_base = "classifier_spgs/runs_from_cluster/2-spgs-new_generation_max_volume/"
# tag = "2-15-batch-size-100"
# tag = "4-spg-1000-epochs"
# tag = "4-spgs-new_generation"
tag = "2-spgs-new_generation_max_volume"

out_base = "comparison_plots/" + tag + "/"
os.system("mkdir -p " + out_base)

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

with open(in_base + "rightly_falsely.pickle", "rb") as file:
    rightly_indices, falsely_indices = pickle.load(file)

# random_crystals = random_crystals[0:100]
# random_labels = random_labels[0:100]
# random_variations = random_variations[0:100]

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
    icsd_occupancies.append(np.mean(occupancies))

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


def get_denseness_factor_ran(structure):

    try:

        actual_volume = structure.volume

        calculated_volume = 0
        for atom in structure:
            specie = atom.species_string
            specie = re.sub(r"\d*\+?$", "", specie)
            specie = re.sub(r"\d*\-?$", "", specie)

            r = random.uniform(
                Element(specie).covalent_radius, Element(specie).vdw_radius
            )
            calculated_volume += 4 / 3 * np.pi * r ** 3

        return actual_volume / calculated_volume

    except:
        return None


def get_denseness_factors(structure):

    denseness_factors = []
    for i in range(0, 10):
        denseness_factor = get_denseness_factor_ran(structure)

        if denseness_factor is not None:
            denseness_factors.append(denseness_factor)

    return denseness_factors


falsely_volumes = []
falsely_denseness_factors = []
falsely_lattice_paras = []
falsely_corn_sizes = []
falsely_NO_wyckoffs = []
falsely_NO_elements = []
falsely_occupancies = []
falsely_element_repetitions = []
falsely_NO_atoms = []

rightly_volumes = []
rightly_denseness_factors = []
rightly_lattice_paras = []
rightly_corn_sizes = []
rightly_NO_wyckoffs = []
rightly_NO_elements = []
rightly_occupancies = []
rightly_element_repetitions = []
rightly_NO_atoms = []

random_volumes = []
random_denseness_factors = []
random_lattice_paras = []
random_NO_atoms = []

for i in falsely_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume

    denseness_factors = get_denseness_factors(structure)

    falsely_NO_atoms.append(len(structure.frac_coords))

    falsely_volumes.append(volume)
    falsely_corn_sizes.extend(icsd_variations[index])
    falsely_NO_elements.append(icsd_NO_elements[index])
    falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    falsely_occupancies.append(icsd_occupancies[index])
    falsely_element_repetitions.extend(icsd_element_repetitions[index])

    falsely_lattice_paras.append(structure.lattice.a)
    falsely_lattice_paras.append(structure.lattice.b)
    falsely_lattice_paras.append(structure.lattice.c)

    falsely_denseness_factors.extend(denseness_factors)

for i in rightly_indices:

    index = int(i / 5)

    structure = icsd_crystals[index]

    volume = structure.volume

    denseness_factors = get_denseness_factors(structure)

    rightly_NO_atoms.append(len(structure.frac_coords))

    rightly_volumes.append(volume)
    rightly_corn_sizes.extend(icsd_variations[index])
    rightly_NO_elements.append(icsd_NO_elements[index])
    rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
    rightly_occupancies.append(icsd_occupancies[index])
    rightly_element_repetitions.extend(icsd_element_repetitions[index])

    rightly_lattice_paras.append(structure.lattice.a)
    rightly_lattice_paras.append(structure.lattice.b)
    rightly_lattice_paras.append(structure.lattice.c)

    rightly_denseness_factors.extend(denseness_factors)

for i, structure in enumerate(random_crystals):

    volume = structure.volume

    denseness_factors = get_denseness_factors(structure)

    random_NO_atoms.append(len(structure.frac_coords))

    random_volumes.append(volume)

    random_lattice_paras.append(structure.lattice.a)
    random_lattice_paras.append(structure.lattice.b)
    random_lattice_paras.append(structure.lattice.c)

    random_denseness_factors.extend(denseness_factors)

################# hist plotting ################

bin_spacing_continuous = 60


def create_histogram(
    tag,
    data,
    labels,
    xlabel,
    is_int=False,
    only_proportions=False,
    min_is_zero=True,
    fixed_min=None,
    fixed_max=None,
):
    # Data: rightly, falsely, random or only rightly, falsely

    # determine range on x axis:
    min = 10 ** 9
    max = 0

    for item in data:
        new_min = np.min(item)
        new_max = np.max(item)

        if new_min < min:
            min = new_min

        if new_max > max:
            max = new_max

    if fixed_max is not None:
        max = fixed_max

    if fixed_min is not None:
        min = fixed_min

    if min_is_zero:
        min = 0

    if not is_int:
        bins = np.linspace(
            min,
            max,
            bin_spacing_continuous,
        )
    else:

        bins = (
            np.arange(min, max + 2, 1 if (max - min) < 60 else int((max - min) / 60))
            - 0.5
        )

    bin_width = bins[1] - bins[0]

    hists = []
    for i, item in enumerate(data):
        if i == 2:  # random
            hist, edges = np.histogram(item, bins, density=True)
        else:
            hist, edges = np.histogram(
                item,
                bins,
            )

        hists.append(hist)

    # to handle rightly and falsely:
    total_hist = hists[0] + hists[1]

    hists[0] = hists[0] / total_hist
    hists[1] = hists[1] / total_hist

    hists[0] = np.nan_to_num(hists[0])
    hists[1] = np.nan_to_num(hists[1])

    if not only_proportions:
        total_hist = total_hist / (np.sum(total_hist) * bin_width)
        hists[0] = hists[0] * total_hist
        hists[1] = hists[1] * total_hist

    # Figure size
    plt.figure()
    ax1 = plt.gca()

    ax1.set_xlabel(xlabel)

    if not only_proportions:
        ax1.set_ylabel("probability density")
    else:
        ax1.set_ylabel("proportion for each bin")

    if only_proportions and len(data) > 2:
        ax2 = ax1.twinx()
        ax2.set_ylabel("probability density")
        ax2.tick_params(axis="y", labelcolor="b")
        ax2.yaxis.label.set_color("b")

    # falsely
    h1 = ax1.bar(
        bins[:-1],
        hists[1],
        bottom=0,
        color="r",
        label=labels[1],
        width=bin_width,
        align="edge",
    )
    # rightly
    h2 = ax1.bar(
        bins[:-1],
        hists[0],
        bottom=hists[1],
        color="g",
        label=labels[0],
        width=bin_width,
        align="edge",
    )

    if len(data) > 2:
        # random
        (h3,) = (ax1 if not only_proportions else ax2).step(
            bins[:-1], hists[2], color="b", label=labels[2], where="post"
        )
        ax1.legend(loc="best", handles=[h1, h2, h3])

    else:

        ax1.legend(loc="best", handles=[h1, h2])

    ax1.set_xlim(left=0, right=None)
    ax1.set_ylim(bottom=0, top=None)

    if len(data) > 2 and only_proportions:
        ax2.set_ylim(bottom=0, top=None)

    plt.tight_layout()
    plt.savefig(
        f"{out_base}{tag}{'_prop' if only_proportions else ''}.png", bbox_inches="tight"
    )
    # plt.show()


for flag in [True, False]:
    create_histogram(
        "volumes",
        [rightly_volumes, falsely_volumes, random_volumes],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        r"volume / $Å^3$",
        is_int=False,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "denseness_factors",
        [
            rightly_denseness_factors,
            falsely_denseness_factors,
            random_denseness_factors,
        ],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "denseness factor",
        is_int=False,
        only_proportions=flag,
        min_is_zero=True,
        # fixed_max=10,
    )

for flag in [True, False]:
    create_histogram(
        "corn_sizes",
        [rightly_corn_sizes, falsely_corn_sizes, random_variations],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "corn size",
        is_int=False,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "NO_wyckoffs",
        [rightly_NO_wyckoffs, falsely_NO_wyckoffs, random_NO_wyckoffs],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "Number of set wyckoff sites",
        is_int=True,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "NO_elements",
        [rightly_NO_elements, falsely_NO_elements, random_NO_elements],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "Number of unique elements on wyckoff sites",
        is_int=True,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "lattice_paras",
        [rightly_lattice_paras, falsely_lattice_paras, random_lattice_paras],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        r"lattice parameter / $Å$",
        is_int=False,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "occupancies",
        [rightly_occupancies, falsely_occupancies],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
        ],
        "occupancy",
        is_int=False,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "element_repetitions",
        [
            rightly_element_repetitions,
            falsely_element_repetitions,
            random_element_repetitions,
        ],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "Number of element repetitions on wyckoff sites",
        is_int=True,
        only_proportions=flag,
        min_is_zero=True,
    )

for flag in [True, False]:
    create_histogram(
        "NO_atoms",
        [
            rightly_NO_atoms,
            falsely_NO_atoms,
            random_NO_elements,
        ],
        [
            "ICSD correctly classified",
            "ICSD incorrectly classified",
            "randomly generated structures",
        ],
        "Number of atoms in the unit cell",
        is_int=True,
        only_proportions=flag,
        min_is_zero=True,
    )

# Info about wyckoff positions in cif file format:
# => where in the cif file is written what kind of wyckoff site we are dealing with?
# The general wyckoff site is always written in the cif file!
# This is because the special ones are only special cases of the general wyckoff position!
# Only the general wyckoff position is needed to generate all the coordinates.
