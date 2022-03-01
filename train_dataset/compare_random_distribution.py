import sys
import os
from dataset_simulations.random_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from pyxtal.database.element import Element
import pickle
from pyxtal import pyxtal
import re
import random
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view

if __name__ == "__main__":

    if len(sys.argv) > 2:
        # Probably running directly from the training script, so take arguments

        in_base = sys.argv[1]
        tag = sys.argv[2]

        spgs = [int(spg) for spg in sys.argv[3:]]

        if len(spgs) == 0:
            spgs = None  # all spgs
        else:
            tag += "/" + "_".join([str(spg) for spg in spgs])

    else:
        in_base = "classifier_spgs/runs_from_cluster/2-spgs-new_generation_max_volume/"
        tag = "2-spgs-new_generation_max_volume"

        spgs = None  # analyse all space groups; alternative: list of spgs

    show_sample_structures = False
    samples_to_show = 3
    counter_shown_icsd_samples = 0
    counter_shown_random_samples = 0

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

    # limit the range:
    if False:
        random_crystals = random_crystals[0:200]
        random_labels = random_labels[0:200]
        random_variations = random_variations[0:200]
        icsd_crystals = icsd_crystals[0:200]
        icsd_labels = icsd_labels[0:200]
        icsd_variations = icsd_variations[0:200]
        icsd_metas = icsd_metas[0:200]

    print("Calculating conventional structures...")
    for i in reversed(range(0, len(icsd_crystals))):

        try:
            # percentage = i / (len(icsd_crystals) + len(random_crystals)) * 100
            # print(f"{int(percentage)}%")

            current_struc = icsd_crystals[i]

            if show_sample_structures and counter_shown_icsd_samples < samples_to_show:
                counter_shown_icsd_samples += 1
                ase_struc = AseAtomsAdaptor.get_atoms(current_struc)
                view(ase_struc)
                input()

            analyzer = SpacegroupAnalyzer(current_struc)
            conv = analyzer.get_conventional_standard_structure()
            icsd_crystals[i] = conv

        except Exception as ex:
            print("Error calculating conventional cell of ICSD:")
            print(ex)

    # this is actually not really needed, but just in case...
    for i in reversed(range(0, len(random_crystals))):

        try:
            # percentage = (
            #    (i + len(icsd_crystals)) / (len(icsd_crystals) + len(random_crystals)) * 100
            # )
            # print(f"{int(percentage)}%")

            current_struc = random_crystals[i]

            if (
                show_sample_structures
                and counter_shown_random_samples < samples_to_show
            ):

                counter_shown_random_samples += 1

                ase_struc = AseAtomsAdaptor.get_atoms(current_struc)

                view(ase_struc)

                input()

            analyzer = SpacegroupAnalyzer(current_struc)
            conv = analyzer.get_conventional_standard_structure()
            random_crystals[i] = conv

        except Exception as ex:
            print("Error calculating conventional cell of random:")
            print("(doesn't matter)")
            print(ex)

    # Get infos from icsd crystals:

    icsd_NO_wyckoffs = []
    icsd_elements = []
    icsd_NO_elements = []
    icsd_occupancies = []
    icsd_occupancies_weights = []
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

        is_pure, NO_wyckoffs, elements, occupancies = sim.get_wyckoff_info(
            icsd_metas[i][0]
        )

        elements_unique = np.unique(elements)

        icsd_NO_wyckoffs.append(NO_wyckoffs)
        icsd_elements.append(elements)
        icsd_NO_elements.append(len(elements_unique))
        icsd_occupancies.append(occupancies)
        icsd_occupancies_weights.append([1 / len(occupancies)] * len(occupancies))

        reps = []
        for el in elements_unique:
            reps.append(np.sum(np.array(elements) == el))
        icsd_element_repetitions.append(reps)

    # preprocess random data:

    def get_wyckoff_info(crystal):
        # returns: Number of set wyckoffs, elements

        struc = pyxtal()
        struc.from_seed(crystal)

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

        if success and (spgs is None or random_labels[i][0] in spgs):

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

                splitted_sites = [
                    item.strip() for item in atom.species_string.split(",")
                ]

                for splitted_site in splitted_sites:

                    splitted = splitted_site.split(":")

                    specie = re.sub(r"\d*[,.]?\d*\+?$", "", splitted[0])
                    specie = re.sub(r"\d*[,.]?\d*\-?$", "", specie)

                    if (
                        "-" in specie
                        or "+" in specie
                        or ":" in specie
                        or "," in specie
                        or "." in specie
                    ):
                        raise Exception(
                            "Something went wrong in get_denseness_factor_ran function."
                        )

                    if len(splitted) > 1:

                        occupancy = float(splitted[1])

                    else:
                        # occupancy = atom.species.element_composition.to_reduced_dict[
                        #    specie
                        # ]

                        # if occupancy != 1.0:
                        #    print("Occupancy not 1.0.")

                        occupancy = 1.0

                    r = random.uniform(
                        Element(specie).covalent_radius, Element(specie).vdw_radius
                    )
                    calculated_volume += 4 / 3 * np.pi * r**3 * occupancy

            return actual_volume / calculated_volume

        except Exception as ex:

            print("Not able to get denseness factor:")
            print(ex)

            # For D and Am exceptions are OK

            return None

    def get_denseness_factors(structure):

        denseness_factors = []
        for i in range(0, 10):
            denseness_factor = get_denseness_factor_ran(structure)

            if denseness_factor is not None:
                denseness_factors.append(denseness_factor)

        return denseness_factors

    falsely_volumes = []
    falsely_angles = []
    falsely_denseness_factors = []
    falsely_lattice_paras = []
    falsely_corn_sizes = []
    falsely_NO_wyckoffs = []
    falsely_NO_elements = []
    falsely_occupancies = []
    falsely_occupancies_weights = []
    falsely_element_repetitions = []
    falsely_NO_atoms = []

    rightly_volumes = []
    rightly_angles = []
    rightly_denseness_factors = []
    rightly_lattice_paras = []
    rightly_corn_sizes = []
    rightly_NO_wyckoffs = []
    rightly_NO_elements = []
    rightly_occupancies = []
    rightly_occupancies_weights = []
    rightly_element_repetitions = []
    rightly_NO_atoms = []

    random_volumes = []
    random_angles = []
    random_denseness_factors = []
    random_lattice_paras = []
    random_NO_atoms = []

    for i in falsely_indices:

        index = int(i / 5)

        if spgs is None or icsd_labels[index][0] in spgs:

            structure = icsd_crystals[index]

            volume = structure.volume

            denseness_factors = get_denseness_factors(structure)

            falsely_NO_atoms.append(len(structure.frac_coords))

            falsely_volumes.append(volume)
            falsely_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            falsely_corn_sizes.extend(icsd_variations[index])
            falsely_NO_elements.append(icsd_NO_elements[index])
            falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            falsely_occupancies.extend(icsd_occupancies[index])
            falsely_occupancies_weights.extend(icsd_occupancies_weights[index])
            falsely_element_repetitions.extend(icsd_element_repetitions[index])

            falsely_lattice_paras.append(structure.lattice.a)
            falsely_lattice_paras.append(structure.lattice.b)
            falsely_lattice_paras.append(structure.lattice.c)

            falsely_denseness_factors.extend(denseness_factors)

    for i in rightly_indices:

        index = int(i / 5)

        if spgs is None or icsd_labels[index][0] in spgs:

            structure = icsd_crystals[index]

            volume = structure.volume

            denseness_factors = get_denseness_factors(structure)

            rightly_NO_atoms.append(len(structure.frac_coords))

            rightly_volumes.append(volume)
            rightly_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            rightly_corn_sizes.extend(icsd_variations[index])
            rightly_NO_elements.append(icsd_NO_elements[index])
            rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            rightly_occupancies.extend(icsd_occupancies[index])
            rightly_occupancies_weights.extend(icsd_occupancies_weights[index])
            rightly_element_repetitions.extend(icsd_element_repetitions[index])

            rightly_lattice_paras.append(structure.lattice.a)
            rightly_lattice_paras.append(structure.lattice.b)
            rightly_lattice_paras.append(structure.lattice.c)

            rightly_denseness_factors.extend(denseness_factors)

    for i, structure in enumerate(random_crystals):

        if spgs is None or random_labels[i][0] in spgs:

            volume = structure.volume

            denseness_factors = get_denseness_factors(structure)

            random_NO_atoms.append(len(structure.frac_coords))

            random_volumes.append(volume)
            random_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )

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
        weights=None,
    ):
        # Data: rightly, falsely, random or only rightly, falsely

        # determine range on x axis:
        min = 10**9
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
                np.arange(
                    min, max + 2, 1 if (max - min) < 60 else int((max - min) / 60)
                )
                - 0.5
            )

        bin_width = bins[1] - bins[0]

        hists = []
        for i, item in enumerate(data):
            if i == 2:  # random
                hist, edges = np.histogram(item, bins, density=True)
            else:
                hist, edges = np.histogram(
                    item, bins, weights=weights  # for occupancies
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
            f"{out_base}{tag}{'_prop' if only_proportions else ''}.png",
            bbox_inches="tight",
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
            "angles",
            [rightly_angles, falsely_angles, random_angles],
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "randomly generated structures",
            ],
            r"angle / °",
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
            weights=[rightly_occupancies_weights, falsely_occupancies_weights],
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
                random_NO_atoms,
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
