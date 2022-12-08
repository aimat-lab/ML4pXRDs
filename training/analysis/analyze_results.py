"""This script can be used to analyze the results of a training run.
Since this script was built during the incremental work on this project,
it contains the analysis of a lot of different features, of which 
only a few are relevant. Feel free to use your own analysis script 
for a more lightweight analysis of the results.
"""

import sys
import os
from sympy import comp
from dataset_simulations.random_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view
from ase.io import write
from training.analysis.analyse_magpie import get_magpie_features
from training.analysis.denseness_factor import get_denseness_factor
from pyxtal.symmetry import Group
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from ml4pxrd_tools.simulation.simulation_core import get_xy_patterns
from pymatgen.core.periodic_table import Species
from training.analysis.entropy import get_chemical_ordering
from training.analysis.entropy import get_structural_complexity
import ml4pxrd_tools.matplotlib_defaults as matplotlib_defaults

figure_double_width_pub = matplotlib_defaults.pub_width
figure_double_width = 10

fix_important_ranges = True
zoom = True  # Add zoomed-in subplots inside the main plot

if __name__ == "__main__":

    if False:
        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        plt.plot([0.0001, 0.0002, 0.0003], [3, 2, 1])
        plt.xlabel("hello world")
        plt.ylabel("hello world")
        plt.tight_layout()
        plt.savefig("test.pdf")
        exit()

    if len(sys.argv) > 2:

        # Probably running directly from the training script, so take arguments
        in_base = sys.argv[1]

        if in_base[-1] != "/":
            in_base += "/"

        tag = sys.argv[2]

        spgs_to_analyze = [int(spg) for spg in sys.argv[3:]]

        if len(spgs_to_analyze) == 0:
            spgs_to_analyze = None  # all spgs
        elif len(spgs_to_analyze) == 1:
            tag += "/" + str(spgs_to_analyze[0])
        else:
            tag += "/all"

    else:

        # in_base = "classifier_spgs/runs_from_cluster/initial_tests/10-03-2022_14-34-51/"
        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/initial_tests/17-03-2022_10-11-11/"
        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/02-05-2022_11-22-37/"

        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/05-06-2022_12-32-24/randomized_coords/"
        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/07-06-2022_09-43-41/"
        in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/19-06-2022_10-15-26/"

        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/09-04-2022_22-56-44/"
        # tag = "magpie_10-03-2022_14-34-51"
        # tag = "volumes_densenesses_4-spg"
        # tag = "look_at_structures"
        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/initial_tests/20-03-2022_02-06-52/"

        # tag = "4-spg-2D-scatters"
        # tag = "volumes_densenesses_2-spg_test/15"

        # tag = "runs_from_cluster/continued_tests/09-04-2022_22-56-44_spgs-50-230_huge_size"
        # tag = "07-06-2022_09-43-41"
        tag = "19-06-2022_10-15-26"

        # spgs_to_analyze = [14, 104, 176, 129]
        spgs_to_analyze = None
        # spgs_to_analyze = [15]
        # spgs_to_analyze = None  # analyse all space groups; alternative: list of spgs

        # spgs_to_analyze = None

    print(f"Analysing {spgs_to_analyze if spgs_to_analyze is not None else 'all'} spgs")

    calculate_conventionals = False

    compute_magpie_features = False

    analyse_complexity_ordering = False

    show_sample_structures = False
    samples_to_show_icsd = 50
    counter_shown_icsd_rightly = 0
    counter_shown_icsd_falsely = 0
    counter_shown_random_rightly = 0
    counter_shown_random_falsely = 0

    show_sample_xrds = False
    xrds_to_show = 2000
    # xrds_to_show = 10**9  # show them all
    show_individual = False
    counter_xrds_icsd_rightly = 0
    counter_xrds_icsd_falsely = 0
    counter_xrds_random_rightly = 0
    counter_xrds_random_falsely = 0

    xrds_icsd_rightly_average = None
    xrds_icsd_falsely_average = None
    xrds_random_rightly_average = None
    xrds_random_falsely_average = None

    out_base = "comparison_plots/" + tag + "/"
    os.system("mkdir -p " + out_base)

    if show_sample_structures:
        os.system("mkdir -p " + out_base + "icsd_rightly_structures")
        os.system("mkdir -p " + out_base + "icsd_falsely_structures")
        os.system("mkdir -p " + out_base + "random_rightly_structures")
        os.system("mkdir -p " + out_base + "random_falsely_structures")

    if show_sample_xrds:
        os.system("mkdir -p " + out_base + "icsd_rightly_xrds")
        os.system("mkdir -p " + out_base + "icsd_falsely_xrds")
        os.system("mkdir -p " + out_base + "random_rightly_xrds")
        os.system("mkdir -p " + out_base + "random_falsely_xrds")

    with open(in_base + "spgs.pickle", "rb") as file:
        spgs = pickle.load(file)

    with open(in_base + "icsd_data.pickle", "rb") as file:
        data = pickle.load(file)
        icsd_crystals, icsd_labels, icsd_variations, icsd_metas = (
            data[0],
            data[1],
            data[2],
            data[3],
        )

    n_patterns_per_crystal = len(icsd_variations[0])

    with open(in_base + "random_data.pickle", "rb") as file:
        data = pickle.load(file)
        (random_crystals, random_labels, random_variations,) = (
            data[0],
            data[1],
            data[2],
        )
    random_labels = [spgs[index] for index in random_labels]

    print(
        f"{len(random_crystals)} random crystals, {len(icsd_crystals)} icsd crystals loaded"
    )

    with open(in_base + "rightly_falsely_icsd.pickle", "rb") as file:
        rightly_indices_icsd, falsely_indices_icsd = pickle.load(file)

    with open(in_base + "rightly_falsely_random.pickle", "rb") as file:
        rightly_indices_random, falsely_indices_random = pickle.load(file)

    # limit the range:
    if False:  # TODO: Change back
        to_process = 300
        random_crystals = random_crystals[0:to_process]
        random_labels = random_labels[0:to_process]
        random_variations = random_variations[0:to_process]
        icsd_crystals = icsd_crystals[0:to_process]
        icsd_labels = icsd_labels[0:to_process]
        icsd_variations = icsd_variations[0:to_process]
        icsd_metas = icsd_metas[0:to_process]

    if calculate_conventionals:
        print("Calculating conventional structures...")
        for i in reversed(range(0, len(icsd_crystals))):

            try:
                # percentage = i / (len(icsd_crystals) + len(random_crystals)) * 100
                # print(f"{int(percentage)}%")

                current_struc = icsd_crystals[i]

                analyzer = SpacegroupAnalyzer(current_struc)
                conv = analyzer.get_conventional_standard_structure()
                icsd_crystals[i] = conv

            except Exception as ex:
                print("Error calculating conventional cell of ICSD:")
                print(ex)

        # This is actually not really needed, but just in case...
        for i in reversed(range(0, len(random_crystals))):

            try:
                # percentage = (
                #    (i + len(icsd_crystals)) / (len(icsd_crystals) + len(random_crystals)) * 100
                # )
                # print(f"{int(percentage)}%")

                current_struc = random_crystals[i]

                analyzer = SpacegroupAnalyzer(current_struc)
                conv = analyzer.get_conventional_standard_structure()
                random_crystals[i] = conv

            except Exception as ex:
                print("Error calculating conventional cell of random:")
                print("(doesn't matter)")
                print(ex)

    # Get infos from icsd crystals:

    icsd_NO_wyckoffs = []
    icsd_NO_elements = []
    icsd_occupancies = []
    icsd_occupancies_weights = []
    icsd_element_repetitions = []
    icsd_wyckoff_repetitions = []

    icsd_NO_unique_wyckoffs = []
    icsd_NO_unique_wyckoffs_summed_over_els = []

    icsd_max_Zs = []
    icsd_set_wyckoffs = []

    # Just for the icsd meta-data (ids):
    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        sim = Simulation(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
    else:
        sim = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    print(f"Processing ICSD: {len(icsd_crystals)} in total.")

    for i in range(0, len(icsd_variations)):

        if spgs_to_analyze is None or icsd_labels[i][0] in spgs_to_analyze:

            (
                is_pure,
                NO_wyckoffs,
                elements,
                occupancies,
                wyckoff_repetitions,
                NO_unique_wyckoffs,
                NO_unique_wyckoffs_summed_over_els,
                unique_wyckoffs,
            ) = sim.get_wyckoff_info(icsd_metas[i][0])

            elements_unique = np.unique(elements)

            try:

                Zs = [Species(name).Z for name in elements_unique]
                icsd_max_Zs.append(max(Zs))

            except Exception as ex:

                icsd_max_Zs.append(None)

            icsd_NO_wyckoffs.append(NO_wyckoffs)
            icsd_NO_elements.append(len(elements_unique))
            icsd_occupancies.append(occupancies)
            icsd_occupancies_weights.append([1 / len(occupancies)] * len(occupancies))

            icsd_NO_unique_wyckoffs.append(NO_unique_wyckoffs)
            icsd_NO_unique_wyckoffs_summed_over_els.append(
                NO_unique_wyckoffs_summed_over_els
            )

            icsd_wyckoff_repetitions.append(wyckoff_repetitions)

            reps = []
            for el in elements_unique:
                reps.append(np.sum(np.array(elements) == el))

            icsd_element_repetitions.append(reps)

            icsd_set_wyckoffs.append(unique_wyckoffs)

        else:

            icsd_max_Zs.append(None)
            icsd_NO_wyckoffs.append(None)
            icsd_NO_elements.append(None)
            icsd_occupancies.append(None)
            icsd_occupancies_weights.append(None)
            icsd_NO_unique_wyckoffs.append(None)
            icsd_NO_unique_wyckoffs_summed_over_els.append(None)
            icsd_wyckoff_repetitions.append(None)
            icsd_element_repetitions.append(None)
            icsd_set_wyckoffs.append(None)

    # preprocess random data:

    def get_wyckoff_info(crystal):
        # returns: Number of set wyckoffs, elements

        struc = pyxtal()
        struc.from_seed(crystal)

        elements = []

        wyckoffs_per_element = {}

        for site in struc.atom_sites:
            specie_str = str(site.specie)
            elements.append(specie_str)

            wyckoff_name = f"{site.wp.multiplicity}{site.wp.letter}"

            if specie_str not in wyckoffs_per_element.keys():
                wyckoffs_per_element[specie_str] = [wyckoff_name]
            else:
                wyckoffs_per_element[specie_str].append(wyckoff_name)

        wyckoff_repetitions = []

        all_wyckoffs = []

        for key in wyckoffs_per_element.keys():
            wyckoffs_unique = np.unique(wyckoffs_per_element[key])

            all_wyckoffs.extend(wyckoffs_unique)

            for item in wyckoffs_unique:
                wyckoff_repetitions.append(
                    np.sum(np.array(wyckoffs_per_element[key]) == item)
                )

        return (
            len(struc.atom_sites),
            elements,
            wyckoff_repetitions,
            len(
                np.unique(all_wyckoffs)
            ),  # how many different wyckoff sites are occupied? "NO_unique_wyckoffs"
            len(
                all_wyckoffs
            ),  # how many different wyckoff sites are occupied summed over unique elements. "NO_unique_wyckoffs_summed_over_els"
            np.unique(all_wyckoffs),
        )

    random_NO_wyckoffs = []
    random_NO_elements = []
    random_element_repetitions = []
    random_wyckoff_repetitions = []
    random_NO_unique_wyckoffs = []
    random_NO_unique_wyckoffs_summed_over_els = []
    random_max_Zs = []
    random_set_wyckoffs = []

    for i in range(0, len(random_variations)):

        print(f"Processing random: {i} of {len(random_variations)}")

        if spgs_to_analyze is None or random_labels[i] in spgs_to_analyze:
            success = True
            try:
                (
                    NO_wyckoffs,
                    elements,
                    wyckoff_repetitions,
                    NO_unique_wyckoffs,
                    NO_unique_wyckoffs_summed_over_els,
                    unique_wyckoffs,
                ) = get_wyckoff_info(random_crystals[i])
            except Exception as ex:
                print(ex)
                success = False
        else:
            success = False

        if success:

            elements_unique = np.unique(elements)

            try:
                Zs = [Species(name).Z for name in elements_unique]
                random_max_Zs.append(max(Zs))
            except Exception as ex:

                random_max_Zs.append(None)

            random_NO_wyckoffs.append(NO_wyckoffs)
            random_NO_elements.append(len(elements_unique))

            reps = []
            for el in elements_unique:
                reps.append(np.sum(np.array(elements) == el))
            random_element_repetitions.append(reps)

            random_wyckoff_repetitions.append(wyckoff_repetitions)

            random_NO_unique_wyckoffs.append(NO_unique_wyckoffs)
            random_NO_unique_wyckoffs_summed_over_els.append(
                NO_unique_wyckoffs_summed_over_els
            )

            random_set_wyckoffs.append(unique_wyckoffs)

        else:

            random_NO_wyckoffs.append(None)
            random_NO_elements.append(None)
            random_element_repetitions.append(None)
            random_wyckoff_repetitions.append(None)
            random_NO_unique_wyckoffs.append(None)
            random_NO_unique_wyckoffs_summed_over_els.append(None)
            random_max_Zs.append(None)
            random_set_wyckoffs.append(None)

    ############## Calculate histograms:

    icsd_falsely_crystals = []
    icsd_falsely_volumes = []
    icsd_falsely_angles = []
    icsd_falsely_denseness_factors = []
    icsd_falsely_lattice_paras = []
    icsd_falsely_max_lattice_paras = []
    icsd_falsely_min_lattice_paras = []
    icsd_falsely_corn_sizes = []
    icsd_falsely_NO_wyckoffs = []
    icsd_falsely_NO_elements = []
    icsd_falsely_occupancies = []
    icsd_falsely_occupancies_weights = []
    icsd_falsely_element_repetitions = []
    icsd_falsely_NO_atoms = []
    icsd_falsely_wyckoff_repetitions = []
    icsd_falsely_magpie_features = []
    icsd_falsely_density = []
    icsd_falsely_sum_cov_vols = []
    icsd_falsely_NO_unique_wyckoffs = []
    icsd_falsely_NO_unique_wyckoffs_summed_over_els = []
    icsd_falsely_max_Zs = []
    icsd_falsely_set_wyckoffs_indices = []
    icsd_falsely_set_wyckoffs_max_indices = []
    icsd_falsely_structural_complexity = []
    icsd_falsely_chemical_ordering = []
    icsd_falsely_volumes_sum_of_intensities = []
    icsd_falsely_COM = []
    icsd_falsely_max_unscaled_intensity = []
    icsd_falsely_max_unscaled_intensity_weighted = []

    icsd_rightly_crystals = []
    icsd_rightly_volumes = []
    icsd_rightly_angles = []
    icsd_rightly_denseness_factors = []
    icsd_rightly_lattice_paras = []
    icsd_rightly_max_lattice_paras = []
    icsd_rightly_min_lattice_paras = []
    icsd_rightly_corn_sizes = []
    icsd_rightly_NO_wyckoffs = []
    icsd_rightly_NO_elements = []
    icsd_rightly_occupancies = []
    icsd_rightly_occupancies_weights = []
    icsd_rightly_element_repetitions = []
    icsd_rightly_NO_atoms = []
    icsd_rightly_wyckoff_repetitions = []
    icsd_rightly_magpie_features = []
    icsd_rightly_density = []
    icsd_rightly_sum_cov_vols = []
    icsd_rightly_NO_unique_wyckoffs = []
    icsd_rightly_NO_unique_wyckoffs_summed_over_els = []
    icsd_rightly_max_Zs = []
    icsd_rightly_set_wyckoffs_indices = []
    icsd_rightly_set_wyckoffs_max_indices = []
    icsd_rightly_structural_complexity = []
    icsd_rightly_chemical_ordering = []
    icsd_rightly_volumes_sum_of_intensities = []
    icsd_rightly_COM = []
    icsd_rightly_max_unscaled_intensity = []
    icsd_rightly_max_unscaled_intensity_weighted = []

    random_rightly_volumes = []
    random_rightly_angles = []
    random_rightly_denseness_factors = []
    random_rightly_lattice_paras = []
    random_rightly_max_lattice_paras = []
    random_rightly_min_lattice_paras = []
    random_rightly_corn_sizes = []
    random_rightly_NO_wyckoffs = []
    random_rightly_NO_elements = []
    random_rightly_element_repetitions = []
    random_rightly_NO_atoms = []
    random_rightly_wyckoff_repetitions = []
    random_rightly_magpie_features = []
    random_rightly_density = []
    random_rightly_sum_cov_vols = []
    random_rightly_NO_unique_wyckoffs = []
    random_rightly_NO_unique_wyckoffs_summed_over_els = []
    random_rightly_max_Zs = []
    random_rightly_set_wyckoffs_indices = []
    random_rightly_set_wyckoffs_max_indices = []
    random_rightly_structural_complexity = []
    random_rightly_chemical_ordering = []
    random_rightly_volumes_sum_of_intensities = []
    random_rightly_COM = []
    random_rightly_max_unscaled_intensity = []
    random_rightly_max_unscaled_intensity_weighted = []

    random_falsely_volumes = []
    random_falsely_angles = []
    random_falsely_denseness_factors = []
    random_falsely_lattice_paras = []
    random_falsely_max_lattice_paras = []
    random_falsely_min_lattice_paras = []
    random_falsely_corn_sizes = []
    random_falsely_NO_wyckoffs = []
    random_falsely_NO_elements = []
    random_falsely_element_repetitions = []
    random_falsely_NO_atoms = []
    random_falsely_wyckoff_repetitions = []
    random_falsely_magpie_features = []
    random_falsely_density = []
    random_falsely_sum_cov_vols = []
    random_falsely_NO_unique_wyckoffs = []
    random_falsely_NO_unique_wyckoffs_summed_over_els = []
    random_falsely_max_Zs = []
    random_falsely_set_wyckoffs_indices = []
    random_falsely_set_wyckoffs_max_indices = []
    random_falsely_structural_complexity = []
    random_falsely_chemical_ordering = []
    random_falsely_volumes_sum_of_intensities = []
    random_falsely_COM = []
    random_falsely_max_unscaled_intensity = []
    random_falsely_max_unscaled_intensity_weighted = []

    if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:

        spg = spgs_to_analyze[0]

        group = Group(spg, dim=3)
        names = [(str(x.multiplicity) + x.letter) for x in group]

    wrong_wyckoff_name_counter = 0

    print("Started processing falsely_indices_icsd")

    total_samples_magpie = 500
    samples_magpie_icsd_falsely = (
        total_samples_magpie
        * len(falsely_indices_icsd)
        / (len(falsely_indices_icsd) + len(rightly_indices_icsd))
    )
    samples_magpie_icsd_rightly = (
        total_samples_magpie
        * len(rightly_indices_icsd)
        / (len(falsely_indices_icsd) + len(rightly_indices_icsd))
    )
    samples_magpie_random_falsely = (
        total_samples_magpie
        * len(falsely_indices_random)
        / (len(falsely_indices_random) + len(rightly_indices_random))
    )
    samples_magpie_random_rightly = (
        total_samples_magpie
        * len(rightly_indices_random)
        / (len(falsely_indices_random) + len(rightly_indices_random))
    )

    # make this scale with the rightly / falsely proportion
    total_xrds_icsd_falsely = (
        xrds_to_show  # 200
        * len(falsely_indices_icsd)
        / (len(falsely_indices_icsd) + len(rightly_indices_icsd))
    )
    total_xrds_icsd_rightly = (
        xrds_to_show  # 200
        * len(rightly_indices_icsd)
        / (len(falsely_indices_icsd) + len(rightly_indices_icsd))
    )
    total_xrds_random_falsely = (
        xrds_to_show  # 200
        * len(falsely_indices_random)
        / (len(falsely_indices_random) + len(rightly_indices_random))
    )
    total_xrds_random_rightly = (
        xrds_to_show  # 200
        * len(rightly_indices_random)
        / (len(falsely_indices_random) + len(rightly_indices_random))
    )

    falsely_icsd_already_processed = []
    rightly_icsd_already_processed = []
    falsely_random_already_processed = []
    rightly_random_already_processed = []

    for i in falsely_indices_icsd:

        index = int(i / n_patterns_per_crystal)

        if index < len(icsd_crystals) and (
            spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze
        ):

            structure = icsd_crystals[index]

            if (
                show_sample_structures
                and counter_shown_icsd_falsely < samples_to_show_icsd
                and structure.is_ordered
                and index not in falsely_icsd_already_processed
            ):
                counter_shown_icsd_falsely += 1
                try:
                    ase_struc = AseAtomsAdaptor.get_atoms(structure)
                    write(
                        f"{out_base}icsd_falsely_structures/{icsd_metas[index][0]}.png",
                        ase_struc,
                    )
                except Exception as ex:
                    print(
                        "Something went wrong creating view of one of the structures."
                    )
                    counter_shown_icsd_falsely -= 1

            if (
                show_sample_xrds
                and counter_xrds_icsd_falsely < total_xrds_icsd_falsely
                and index not in falsely_icsd_already_processed
            ):
                (
                    patterns,
                    angles,
                    intensities,
                    max_unscaled_intensity_angle,
                ) = get_xy_patterns(
                    structure,
                    1.5406,
                    np.linspace(5, 90, 8501),
                    1,
                    (5, 90),
                    False,
                    False,
                    True,  # return angles and intensities
                    True,  # return max_unscaled_intensity_angle
                )
                pattern = patterns[0]
                icsd_falsely_volumes_sum_of_intensities.append(
                    (structure.volume, np.sum(intensities))
                )
                icsd_falsely_COM.append(
                    np.sum(np.array(angles) * np.array(intensities))
                    / np.sum(intensities)
                )
                icsd_falsely_max_unscaled_intensity.append(
                    max_unscaled_intensity_angle[0]
                )
                icsd_falsely_max_unscaled_intensity_weighted.append(
                    max_unscaled_intensity_angle[0] * max_unscaled_intensity_angle[1]
                )

                if show_individual:
                    plt.figure(
                        figsize=(
                            figure_double_width_pub * 0.95 * 0.5,
                            figure_double_width_pub * 0.7 * 0.5,
                        )
                    )
                    plt.plot(np.linspace(5, 90, 8501), pattern)
                    plt.tight_layout()
                    plt.savefig(
                        out_base + f"icsd_falsely_xrds/{icsd_metas[index][0]}_pub.pdf",
                        bbox_inches="tight",
                    )

                    plt.gcf().set_size_inches(
                        figure_double_width * 0.95 * 0.5,
                        figure_double_width * 0.7 * 0.5,
                    )
                    plt.savefig(
                        out_base + f"icsd_falsely_xrds/{icsd_metas[index][0]}.pdf",
                        bbox_inches="tight",
                    )

                counter_xrds_icsd_falsely += 1

                if xrds_icsd_falsely_average is None:
                    xrds_icsd_falsely_average = pattern
                else:
                    xrds_icsd_falsely_average += pattern

            falsely_icsd_already_processed.append(index)

            icsd_falsely_crystals.append(structure)

            volume = structure.volume

            result = get_denseness_factor(structure)
            if result is not None:
                denseness_factor, sum_cov_vols = result
            else:
                denseness_factor, sum_cov_vols = None, None
            icsd_falsely_sum_cov_vols.append(sum_cov_vols)

            try:
                icsd_falsely_density.append(structure.density)
            except Exception as ex:
                print(ex)
                icsd_falsely_density.append(None)

            icsd_falsely_NO_atoms.append(len(structure.frac_coords))

            icsd_falsely_volumes.append(volume)
            icsd_falsely_angles.append(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            icsd_falsely_corn_sizes.append(icsd_variations[index])
            icsd_falsely_NO_elements.append(icsd_NO_elements[index])
            icsd_falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            icsd_falsely_occupancies.append(icsd_occupancies[index])
            icsd_falsely_occupancies_weights.append(icsd_occupancies_weights[index])
            icsd_falsely_element_repetitions.append(icsd_element_repetitions[index])

            icsd_falsely_lattice_paras.append(
                [
                    structure.lattice.a,
                    structure.lattice.b,
                    structure.lattice.c,
                ]
            )

            icsd_falsely_max_lattice_paras.append(
                max(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )
            icsd_falsely_min_lattice_paras.append(
                min(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )

            if denseness_factor is not None:
                icsd_falsely_denseness_factors.append(denseness_factor)
            else:
                icsd_falsely_denseness_factors.append(None)

            icsd_falsely_wyckoff_repetitions.append(icsd_wyckoff_repetitions[index])

            icsd_falsely_NO_unique_wyckoffs.append(icsd_NO_unique_wyckoffs[index])
            icsd_falsely_NO_unique_wyckoffs_summed_over_els.append(
                icsd_NO_unique_wyckoffs_summed_over_els[index]
            )

            icsd_falsely_max_Zs.append(icsd_max_Zs[index])

            if analyse_complexity_ordering:
                try:
                    if structure.is_ordered:
                        icsd_falsely_chemical_ordering.append(
                            get_chemical_ordering(structure)
                        )
                    else:
                        icsd_falsely_chemical_ordering.append(None)
                except Exception as ex:
                    print("Error calculating chemical ordering:")
                    print(ex)
                    icsd_falsely_chemical_ordering.append(None)
                try:
                    icsd_falsely_structural_complexity.append(
                        get_structural_complexity(structure)
                    )
                except Exception as ex:
                    print("Error calculating structural complexity:")
                    print(ex)
                    icsd_falsely_structural_complexity.append(None)

            skip = False
            if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
                indices = []

                for name in icsd_set_wyckoffs[index]:
                    if name in names:
                        indices.append(names.index(name))
                    else:
                        print("Wrong wyckoff name! ##################################")
                        wrong_wyckoff_name_counter += 1
                        skip = True
                        break

                if not skip:
                    icsd_falsely_set_wyckoffs_indices.append(indices)
                    icsd_falsely_set_wyckoffs_max_indices.append(max(indices))
                else:
                    icsd_falsely_set_wyckoffs_indices.append(None)
                    icsd_falsely_set_wyckoffs_max_indices.append(None)
                    skip = False

            if compute_magpie_features:
                if not np.any(np.array(icsd_occupancies[index]) != 1.0):
                    try:
                        # Make sure that the proportions are kept correct:
                        if (
                            len(icsd_falsely_magpie_features)
                            < samples_magpie_icsd_falsely
                        ):
                            magpie_features = get_magpie_features(structure)

                            if (
                                magpie_features is not None
                            ):  # limit the amount of computations
                                icsd_falsely_magpie_features.append(magpie_features)

                                print(
                                    f"{len(icsd_falsely_magpie_features)} of {samples_magpie_icsd_falsely}"
                                )

                    except Exception as ex:
                        print("Error calculating magpie features.")
                        print(ex)

    if compute_magpie_features:
        if len(icsd_falsely_magpie_features) != (int(samples_magpie_icsd_falsely) + 1):
            raise Exception("total_samples_magpie was set too high.")

    print("Started processing rightly_indices_icsd")

    for i in rightly_indices_icsd:

        index = int(i / n_patterns_per_crystal)

        if index < len(icsd_crystals) and (
            spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze
        ):

            structure = icsd_crystals[index]

            if (
                show_sample_structures
                and counter_shown_icsd_rightly < samples_to_show_icsd
                and structure.is_ordered
                and index not in rightly_icsd_already_processed
            ):
                counter_shown_icsd_rightly += 1

                try:
                    ase_struc = AseAtomsAdaptor.get_atoms(structure)
                    write(
                        f"{out_base}icsd_rightly_structures/{icsd_metas[index][0]}.png",
                        ase_struc,
                    )
                except Exception as ex:
                    print(
                        "Something went wrong creating view of one of the structures."
                    )
                    counter_shown_icsd_rightly -= 1

            if (
                show_sample_xrds
                and counter_xrds_icsd_rightly < total_xrds_icsd_rightly
                and index not in rightly_icsd_already_processed
            ):

                (
                    patterns,
                    angles,
                    intensities,
                    max_unscaled_intensity_angle,
                ) = get_xy_patterns(
                    structure,
                    1.5406,
                    np.linspace(5, 90, 8501),
                    1,
                    (5, 90),
                    False,
                    False,
                    True,  # return angles and intensities
                    True,
                )
                pattern = patterns[0]
                icsd_rightly_volumes_sum_of_intensities.append(
                    (structure.volume, np.sum(intensities))
                )
                icsd_rightly_COM.append(
                    np.sum(np.array(angles) * np.array(intensities))
                    / np.sum(intensities)
                )
                icsd_rightly_max_unscaled_intensity.append(
                    max_unscaled_intensity_angle[0]
                )
                icsd_rightly_max_unscaled_intensity_weighted.append(
                    max_unscaled_intensity_angle[0] * max_unscaled_intensity_angle[1]
                )

                if show_individual:
                    plt.figure(
                        figsize=(
                            figure_double_width_pub * 0.95 * 0.5,
                            figure_double_width_pub * 0.7 * 0.5,
                        )
                    )
                    plt.plot(np.linspace(5, 90, 8501), pattern)
                    plt.tight_layout()
                    plt.savefig(
                        out_base + f"icsd_rightly_xrds/{icsd_metas[index][0]}_pub.pdf",
                        bbox_inches="tight",
                    )

                    plt.gcf().set_size_inches(
                        figure_double_width * 0.95 * 0.5,
                        figure_double_width * 0.7 * 0.5,
                    )
                    plt.savefig(
                        out_base + f"icsd_rightly_xrds/{icsd_metas[index][0]}.pdf",
                        bbox_inches="tight",
                    )

                counter_xrds_icsd_rightly += 1

                if xrds_icsd_rightly_average is None:
                    xrds_icsd_rightly_average = pattern
                else:
                    xrds_icsd_rightly_average += pattern

            rightly_icsd_already_processed.append(index)

            icsd_rightly_crystals.append(structure)

            volume = structure.volume

            result = get_denseness_factor(structure)
            if result is not None:
                denseness_factor, sum_cov_vols = result
            else:
                denseness_factor, sum_cov_vols = None, None
            icsd_rightly_sum_cov_vols.append(sum_cov_vols)

            try:
                icsd_rightly_density.append(structure.density)
            except Exception as ex:
                print(ex)
                icsd_rightly_density.append(None)

            icsd_rightly_NO_atoms.append(len(structure.frac_coords))

            icsd_rightly_volumes.append(volume)
            icsd_rightly_angles.append(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            icsd_rightly_corn_sizes.append(icsd_variations[index])
            icsd_rightly_NO_elements.append(icsd_NO_elements[index])
            icsd_rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            icsd_rightly_occupancies.append(icsd_occupancies[index])
            icsd_rightly_occupancies_weights.append(icsd_occupancies_weights[index])
            icsd_rightly_element_repetitions.append(icsd_element_repetitions[index])

            icsd_rightly_lattice_paras.append(
                [
                    structure.lattice.a,
                    structure.lattice.b,
                    structure.lattice.c,
                ]
            )

            icsd_rightly_max_lattice_paras.append(
                max(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )
            icsd_rightly_min_lattice_paras.append(
                min(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )

            if denseness_factor is not None:
                icsd_rightly_denseness_factors.append(denseness_factor)
            else:
                icsd_rightly_denseness_factors.append(None)

            icsd_rightly_wyckoff_repetitions.append(icsd_wyckoff_repetitions[index])

            icsd_rightly_NO_unique_wyckoffs.append(icsd_NO_unique_wyckoffs[index])
            icsd_rightly_NO_unique_wyckoffs_summed_over_els.append(
                icsd_NO_unique_wyckoffs_summed_over_els[index]
            )

            icsd_rightly_max_Zs.append(icsd_max_Zs[index])

            if analyse_complexity_ordering:
                try:
                    if structure.is_ordered:
                        icsd_rightly_chemical_ordering.append(
                            get_chemical_ordering(structure)
                        )
                    else:
                        icsd_rightly_chemical_ordering.append(None)
                except Exception as ex:
                    print("Error calculating chemical ordering:")
                    print(ex)
                    icsd_rightly_chemical_ordering.append(None)
                try:
                    icsd_rightly_structural_complexity.append(
                        get_structural_complexity(structure)
                    )
                except Exception as ex:
                    print("Error calculating structural complexity:")
                    print(ex)
                    icsd_rightly_structural_complexity.append(None)

            skip = False
            if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
                indices = []

                for name in icsd_set_wyckoffs[index]:
                    if name in names:
                        indices.append(names.index(name))
                    else:
                        print("Wrong wyckoff name! ##################################")
                        wrong_wyckoff_name_counter += 1
                        skip = True
                        break

                if not skip:
                    icsd_rightly_set_wyckoffs_max_indices.append(max(indices))
                    icsd_rightly_set_wyckoffs_indices.append(indices)
                else:
                    skip = False
                    icsd_rightly_set_wyckoffs_max_indices.append(None)
                    icsd_rightly_set_wyckoffs_indices.append(None)

            if compute_magpie_features:
                if not np.any(np.array(icsd_occupancies[index]) != 1.0):
                    try:
                        if (
                            len(icsd_rightly_magpie_features)
                            < samples_magpie_icsd_rightly
                        ):
                            magpie_features = get_magpie_features(structure)
                            if (
                                magpie_features is not None
                            ):  # limit the amount of computations
                                icsd_rightly_magpie_features.append(magpie_features)

                                print(
                                    f"{len(icsd_rightly_magpie_features)} of {samples_magpie_icsd_rightly}"
                                )
                    except Exception as ex:
                        print("Error calculating magpie features.")
                        print(ex)

    if compute_magpie_features:
        if len(icsd_rightly_magpie_features) != (int(samples_magpie_icsd_rightly) + 1):
            raise Exception("total_samples_magpie was set too high.")

    print("Started processing falsely_indices_random")

    for index in falsely_indices_random:
        if index < len(random_crystals) and (
            spgs_to_analyze is None or random_labels[index] in spgs_to_analyze
        ):

            structure = random_crystals[index]

            if (
                show_sample_structures
                and counter_shown_random_falsely < samples_to_show_icsd
            ):
                try:
                    counter_shown_random_falsely += 1
                    ase_struc = AseAtomsAdaptor.get_atoms(structure)
                    # view(ase_struc)
                    # input("")
                    write(
                        f"{out_base}random_falsely_structures/{counter_shown_random_falsely}.png",
                        ase_struc,
                    )
                except Exception as ex:
                    print(
                        "Something went wrong creating view of one of the structures."
                    )
                    counter_shown_random_falsely -= 1

            if (
                show_sample_xrds
                and counter_xrds_random_falsely < total_xrds_random_falsely
            ):
                (
                    patterns,
                    angles,
                    intensities,
                    max_unscaled_intensity_angle,
                ) = get_xy_patterns(
                    structure,
                    1.5406,
                    np.linspace(5, 90, 8501),
                    1,
                    (5, 90),
                    False,
                    False,
                    True,  # return angles and intensities
                    True,
                )
                pattern = patterns[0]
                random_falsely_volumes_sum_of_intensities.append(
                    (structure.volume, np.sum(intensities))
                )
                random_falsely_COM.append(
                    np.sum(np.array(angles) * np.array(intensities))
                    / np.sum(intensities)
                )
                random_falsely_max_unscaled_intensity.append(
                    max_unscaled_intensity_angle[0]
                )
                random_falsely_max_unscaled_intensity_weighted.append(
                    max_unscaled_intensity_angle[0] * max_unscaled_intensity_angle[1]
                )

                if show_individual:
                    plt.figure(
                        figsize=(
                            figure_double_width_pub * 0.95 * 0.5,
                            figure_double_width_pub * 0.7 * 0.5,
                        )
                    )
                    plt.plot(np.linspace(5, 90, 8501), pattern)
                    plt.tight_layout()
                    plt.savefig(
                        out_base
                        + f"random_falsely_xrds/{counter_xrds_random_falsely}_pub.pdf",
                        bbox_inches="tight",
                    )

                    plt.gcf().set_size_inches(
                        figure_double_width * 0.95 * 0.5,
                        figure_double_width * 0.7 * 0.5,
                    )
                    plt.savefig(
                        out_base
                        + f"random_falsely_xrds/{counter_xrds_random_falsely}.pdf",
                        bbox_inches="tight",
                    )

                counter_xrds_random_falsely += 1

                if xrds_random_falsely_average is None:
                    xrds_random_falsely_average = pattern
                else:
                    xrds_random_falsely_average += pattern

            volume = structure.volume

            result = get_denseness_factor(structure)
            if result is not None:
                denseness_factor, sum_cov_vols = result
            else:
                denseness_factor, sum_cov_vols = None, None
            random_falsely_sum_cov_vols.append(sum_cov_vols)

            try:
                random_falsely_density.append(structure.density)
            except Exception as ex:
                print(ex)
                random_falsely_density.append(None)

            random_falsely_NO_atoms.append(len(structure.frac_coords))

            random_falsely_volumes.append(volume)
            random_falsely_angles.append(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )

            random_falsely_lattice_paras.append(
                [
                    structure.lattice.a,
                    structure.lattice.b,
                    structure.lattice.c,
                ]
            )

            random_falsely_max_lattice_paras.append(
                max(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )
            random_falsely_min_lattice_paras.append(
                min(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )

            if denseness_factor is not None:
                random_falsely_denseness_factors.append(denseness_factor)
            else:
                random_falsely_denseness_factors.append(None)

            random_falsely_corn_sizes.append(random_variations[index])
            random_falsely_NO_elements.append(random_NO_elements[index])
            random_falsely_NO_wyckoffs.append(random_NO_wyckoffs[index])

            random_falsely_element_repetitions.append(random_element_repetitions[index])
            random_falsely_wyckoff_repetitions.append(random_wyckoff_repetitions[index])

            random_falsely_NO_unique_wyckoffs.append(random_NO_unique_wyckoffs[index])
            random_falsely_NO_unique_wyckoffs_summed_over_els.append(
                random_NO_unique_wyckoffs_summed_over_els[index]
            )

            random_falsely_max_Zs.append(random_max_Zs[index])

            if analyse_complexity_ordering:
                try:
                    if structure.is_ordered:
                        random_falsely_chemical_ordering.append(
                            get_chemical_ordering(structure)
                        )
                    else:
                        random_falsely_chemical_ordering.append(None)
                except Exception as ex:
                    print("Error calculating chemical ordering:")
                    print(ex)
                    random_falsely_chemical_ordering.append(None)
                try:
                    random_falsely_structural_complexity.append(
                        get_structural_complexity(structure)
                    )
                except Exception as ex:
                    print("Error calculating structural complexity:")
                    print(ex)
                    random_falsely_structural_complexity.append(None)

            skip = False
            if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
                indices = []

                for name in random_set_wyckoffs[index]:
                    if name in names:
                        indices.append(names.index(name))
                    else:
                        print("Wrong wyckoff name! ##################################")
                        wrong_wyckoff_name_counter += 1
                        skip = True
                        break

                if not skip:
                    random_falsely_set_wyckoffs_max_indices.append(max(indices))
                    random_falsely_set_wyckoffs_indices.append(indices)
                else:
                    skip = False
                    random_falsely_set_wyckoffs_indices.append(None)
                    random_falsely_set_wyckoffs_max_indices.append(None)

            if compute_magpie_features:
                try:
                    if (
                        len(random_falsely_magpie_features)
                        < samples_magpie_random_falsely
                    ):  # limit the amount of computations

                        magpie_features = get_magpie_features(structure)

                        if magpie_features is not None:
                            random_falsely_magpie_features.append(magpie_features)

                            print(
                                f"{len(random_falsely_magpie_features)} of {samples_magpie_random_falsely}"
                            )

                except Exception as ex:
                    print("Error calculating magpie features.")
                    print(ex)

    if compute_magpie_features:
        if len(random_falsely_magpie_features) != (
            int(samples_magpie_random_falsely) + 1
        ):
            raise Exception("total_samples_magpie was set too high.")

    print("Started processing rightly_indices_random")

    for index in rightly_indices_random:
        if index < len(random_crystals) and (
            spgs_to_analyze is None or random_labels[index] in spgs_to_analyze
        ):

            structure = random_crystals[index]

            if (
                show_sample_structures
                and counter_shown_random_rightly < samples_to_show_icsd
            ):
                try:
                    counter_shown_random_rightly += 1
                    ase_struc = AseAtomsAdaptor.get_atoms(structure)
                    # view(ase_struc)
                    # input("")
                    write(
                        f"{out_base}random_rightly_structures/{counter_shown_random_rightly}.png",
                        ase_struc,
                    )
                except Exception as ex:
                    print(
                        "Something went wrong creating view of one of the structures."
                    )
                    counter_shown_random_rightly -= 1

            if (
                show_sample_xrds
                and counter_xrds_random_rightly < total_xrds_random_rightly
            ):
                (
                    patterns,
                    angles,
                    intensities,
                    max_unscaled_intensity_angle,
                ) = get_xy_patterns(
                    structure,
                    1.5406,
                    np.linspace(5, 90, 8501),
                    1,
                    (5, 90),
                    False,
                    False,
                    True,  # return angles and intensities
                    True,
                )
                pattern = patterns[0]
                random_rightly_volumes_sum_of_intensities.append(
                    (structure.volume, np.sum(intensities))
                )
                random_rightly_COM.append(
                    np.sum(np.array(angles) * np.array(intensities))
                    / np.sum(intensities)
                )
                random_rightly_max_unscaled_intensity.append(
                    max_unscaled_intensity_angle[0]
                )
                random_rightly_max_unscaled_intensity_weighted.append(
                    max_unscaled_intensity_angle[0] * max_unscaled_intensity_angle[1]
                )

                if show_individual:
                    plt.figure(
                        figsize=(
                            figure_double_width_pub * 0.95 * 0.5,
                            figure_double_width_pub * 0.7 * 0.5,
                        )
                    )
                    plt.plot(np.linspace(5, 90, 8501), pattern)
                    plt.tight_layout()
                    plt.savefig(
                        out_base
                        + f"random_rightly_xrds/{counter_xrds_random_rightly}_pub.pdf",
                        bbox_inches="tight",
                    )

                    plt.gcf().set_size_inches(
                        figure_double_width * 0.95 * 0.5,
                        figure_double_width * 0.7 * 0.5,
                    )
                    plt.savefig(
                        out_base
                        + f"random_rightly_xrds/{counter_xrds_random_rightly}.pdf",
                        bbox_inches="tight",
                    )

                counter_xrds_random_rightly += 1

                if xrds_random_rightly_average is None:
                    xrds_random_rightly_average = pattern
                else:
                    xrds_random_rightly_average += pattern

            volume = structure.volume

            result = get_denseness_factor(structure)
            if result is not None:
                denseness_factor, sum_cov_vols = result
            else:
                denseness_factor, sum_cov_vols = None, None
            random_rightly_sum_cov_vols.append(sum_cov_vols)

            try:
                random_rightly_density.append(structure.density)
            except Exception as ex:
                print(ex)
                random_rightly_density.append(None)

            random_rightly_NO_atoms.append(len(structure.frac_coords))

            random_rightly_volumes.append(volume)
            random_rightly_angles.append(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )

            random_rightly_lattice_paras.append(
                [
                    structure.lattice.a,
                    structure.lattice.b,
                    structure.lattice.c,
                ]
            )

            random_rightly_max_lattice_paras.append(
                max(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )
            random_rightly_min_lattice_paras.append(
                min(structure.lattice.a, structure.lattice.b, structure.lattice.c)
            )

            if denseness_factor is not None:
                random_rightly_denseness_factors.append(denseness_factor)
            else:
                random_rightly_denseness_factors.append(None)

            random_rightly_corn_sizes.append(random_variations[index])
            random_rightly_NO_elements.append(random_NO_elements[index])
            random_rightly_NO_wyckoffs.append(random_NO_wyckoffs[index])

            random_rightly_element_repetitions.append(random_element_repetitions[index])
            random_rightly_wyckoff_repetitions.append(random_wyckoff_repetitions[index])

            random_rightly_NO_unique_wyckoffs.append(random_NO_unique_wyckoffs[index])
            random_rightly_NO_unique_wyckoffs_summed_over_els.append(
                random_NO_unique_wyckoffs_summed_over_els[index]
            )

            random_rightly_max_Zs.append(random_max_Zs[index])

            if analyse_complexity_ordering:
                try:
                    if structure.is_ordered:
                        random_rightly_chemical_ordering.append(
                            get_chemical_ordering(structure)
                        )
                    else:
                        random_rightly_chemical_ordering.append(None)
                except Exception as ex:
                    print("Error calculating chemical ordering:")
                    print(ex)
                    random_rightly_chemical_ordering.append(None)
                try:
                    random_rightly_structural_complexity.append(
                        get_structural_complexity(structure)
                    )
                except Exception as ex:
                    print("Error calculating structural complexity:")
                    print(ex)
                    random_rightly_structural_complexity.append(None)

            skip = False
            if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
                indices = []

                for name in random_set_wyckoffs[index]:
                    if name in names:
                        indices.append(names.index(name))
                    else:
                        print("Wrong wyckoff name! ##################################")
                        wrong_wyckoff_name_counter += 1
                        skip = True
                        break

                if not skip:
                    random_rightly_set_wyckoffs_indices.append(indices)
                    random_rightly_set_wyckoffs_max_indices.append(max(indices))
                else:
                    skip = False
                    random_rightly_set_wyckoffs_indices.append(None)
                    random_rightly_set_wyckoffs_max_indices.append(None)

            if compute_magpie_features:
                try:
                    if (
                        len(random_rightly_magpie_features)
                        < samples_magpie_random_rightly
                    ):  # limit the amount of computations

                        magpie_features = get_magpie_features(structure)

                        if magpie_features is not None:
                            random_rightly_magpie_features.append(magpie_features)

                        print(
                            f"{len(random_rightly_magpie_features)} of {samples_magpie_random_rightly}"
                        )

                except Exception as ex:
                    print("Error calculating magpie features.")
                    print(ex)

    if compute_magpie_features:
        if len(random_rightly_magpie_features) != (
            int(samples_magpie_random_rightly) + 1
        ):
            raise Exception("total_samples_magpie was set too high.")

    if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
        print(f"{wrong_wyckoff_name_counter} wyckoff names mismatched.")

    # combine output pngs of structures:

    if show_sample_structures:
        os.system("rm " + out_base + "icsd_rightly_structures/combined.png")
        os.system("rm " + out_base + "icsd_falsely_structures/combined.png")
        os.system("rm " + out_base + "random_rightly_structures/combined.png")
        os.system("rm " + out_base + "random_falsely_structures/combined.png")

        os.system(
            "montage -density 300 -tile 5x0 -geometry +5+5 -border 5 "
            + out_base
            + "icsd_rightly_structures/*.png"
            + " -resize 200x "
            + out_base
            + "icsd_rightly_structures/combined.png"
        )
        os.system(
            "montage -density 300 -tile 5x0 -geometry +5+5 -border 5 "
            + out_base
            + "icsd_falsely_structures/*.png"
            + " -resize 200x "
            + out_base
            + "icsd_falsely_structures/combined.png"
        )
        os.system(
            "montage -density 300 -tile 5x0 -geometry +5+5 -border 5 "
            + out_base
            + "random_rightly_structures/*.png"
            + " -resize 200x "
            + out_base
            + "random_rightly_structures/combined.png"
        )
        os.system(
            "montage -density 300 -tile 5x0 -geometry +5+5 -border 5 "
            + out_base
            + "random_falsely_structures/*.png"
            + " -resize 200x "
            + out_base
            + "random_falsely_structures/combined.png"
        )

    if show_sample_xrds:

        xrds_icsd_falsely_average /= counter_xrds_icsd_falsely
        xrds_icsd_rightly_average /= counter_xrds_icsd_rightly
        xrds_random_rightly_average /= counter_xrds_random_rightly
        xrds_random_falsely_average /= counter_xrds_random_falsely

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )

        plt.plot(
            np.linspace(5, 90, 8501),
            xrds_icsd_falsely_average,
            label="ICSD falsely",
            linestyle="dotted",
            alpha=0.5,
        )
        plt.plot(
            np.linspace(5, 90, 8501),
            xrds_icsd_rightly_average,
            label="ICSD rightly",
            linestyle="dotted",
            alpha=0.5,
        )
        plt.plot(
            np.linspace(5, 90, 8501),
            xrds_random_falsely_average,
            label="Random falsely",
            linestyle="dotted",
            alpha=0.5,
        )
        plt.plot(
            np.linspace(5, 90, 8501),
            xrds_random_rightly_average,
            label="Random rightly",
            linestyle="dotted",
            alpha=0.5,
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base + f"average_xrds_pub.pdf", bbox_inches="tight")

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(out_base + f"average_xrds.pdf", bbox_inches="tight")

    ################# 2D scatter plots ################

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_denseness_factors, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_denseness_factors, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0.5, 3.3)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_densenesses_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_densenesses_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes, random_rightly_denseness_factors, color="g", s=0.5
    )
    plt.scatter(
        random_falsely_volumes, random_falsely_denseness_factors, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0.5, 3.3)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_densenesses_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_densenesses_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_NO_unique_wyckoffs, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_NO_unique_wyckoffs, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 9)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_icsd_pub.pdf", bbox_inches="tight"
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_icsd.pdf", bbox_inches="tight"
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes, random_rightly_NO_unique_wyckoffs, color="g", s=0.5
    )
    plt.scatter(
        random_falsely_volumes, random_falsely_NO_unique_wyckoffs, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 9)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_random.pdf", bbox_inches="tight"
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        icsd_rightly_volumes,
        icsd_rightly_NO_unique_wyckoffs_summed_over_els,
        color="g",
        s=0.5,
    )
    plt.scatter(
        icsd_falsely_volumes,
        icsd_falsely_NO_unique_wyckoffs_summed_over_els,
        color="r",
        s=0.5,
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 17)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_summed_over_els_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_summed_over_els_icsd.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes,
        random_rightly_NO_unique_wyckoffs_summed_over_els,
        color="g",
        s=0.5,
    )
    plt.scatter(
        random_falsely_volumes,
        random_falsely_NO_unique_wyckoffs_summed_over_els,
        color="r",
        s=0.5,
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 17)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_summed_over_els_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_wyckoffs_summed_over_els_random.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_max_lattice_paras, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_max_lattice_paras, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_max_lattice_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_max_lattice_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes, random_rightly_max_lattice_paras, color="g", s=0.5
    )
    plt.scatter(
        random_falsely_volumes, random_falsely_max_lattice_paras, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_max_lattice_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_max_lattice_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_min_lattice_paras, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_min_lattice_paras, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_min_lattice_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_min_lattice_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes, random_rightly_min_lattice_paras, color="g", s=0.5
    )
    plt.scatter(
        random_falsely_volumes, random_falsely_min_lattice_paras, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_min_lattice_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_min_lattice_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        icsd_rightly_volumes,
        [item[0] for item in icsd_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        icsd_rightly_volumes,
        [item[1] for item in icsd_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        icsd_rightly_volumes,
        [item[2] for item in icsd_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        icsd_falsely_volumes,
        [item[0] for item in icsd_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.scatter(
        icsd_falsely_volumes,
        [item[1] for item in icsd_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.scatter(
        icsd_falsely_volumes,
        [item[2] for item in icsd_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.ylim(80, 140)
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_angles_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_angles_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_volumes,
        [item[0] for item in random_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        random_rightly_volumes,
        [item[1] for item in random_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        random_rightly_volumes,
        [item[2] for item in random_rightly_angles],
        color="g",
        s=0.5,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[0] for item in random_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[1] for item in random_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[2] for item in random_falsely_angles],
        color="r",
        s=0.5,
    )
    plt.ylim(80, 140)
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_angles_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_angles_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_density, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_density, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 17.5)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_density_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_density_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(random_rightly_volumes, random_rightly_density, color="g", s=0.5)
    plt.scatter(random_falsely_volumes, random_falsely_density, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 17.5)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_density_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_density_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        icsd_rightly_sum_cov_vols, icsd_rightly_denseness_factors, color="g", s=0.5
    )
    plt.scatter(
        icsd_falsely_sum_cov_vols, icsd_falsely_denseness_factors, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_sum_cov_vols_denseness_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_sum_cov_vols_denseness_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        random_rightly_sum_cov_vols, random_rightly_denseness_factors, color="g", s=0.5
    )
    plt.scatter(
        random_falsely_sum_cov_vols, random_falsely_denseness_factors, color="r", s=0.5
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_sum_cov_vols_denseness_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_sum_cov_vols_denseness_random.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_NO_atoms, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_NO_atoms, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 400)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_NO_atoms_icsd_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_NO_atoms_icsd.pdf", bbox_inches="tight")

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(random_rightly_volumes, random_rightly_NO_atoms, color="g", s=0.5)
    plt.scatter(random_falsely_volumes, random_falsely_NO_atoms, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 400)
    plt.tight_layout()
    plt.savefig(f"{out_base}2D_volumes_NO_atoms_random_pub.pdf", bbox_inches="tight")

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(f"{out_base}2D_volumes_NO_atoms_random.pdf", bbox_inches="tight")

    # set_wyckoffs_indices over volume

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    volumes = []
    for i, item in enumerate(icsd_rightly_set_wyckoffs_indices):
        if item is not None:
            for subitem in item:
                indices.append(subitem)
                volumes.append(icsd_rightly_volumes[i])
    plt.scatter(volumes, indices, color="g", s=0.5)
    indices = []
    volumes = []
    for i, item in enumerate(icsd_falsely_set_wyckoffs_indices):
        if item is not None:
            for subitem in item:
                indices.append(subitem)
                volumes.append(icsd_falsely_volumes[i])
    plt.scatter(volumes, indices, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_indices_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_indices_icsd.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    volumes = []
    for i, item in enumerate(random_rightly_set_wyckoffs_indices):
        if item is not None:
            for subitem in item:
                indices.append(subitem)
                volumes.append(random_rightly_volumes[i])
    plt.scatter(volumes, indices, color="g", s=0.5)
    indices = []
    volumes = []
    for i, item in enumerate(random_falsely_set_wyckoffs_indices):
        if item is not None:
            for subitem in item:
                indices.append(subitem)
                volumes.append(random_falsely_volumes[i])
    plt.scatter(volumes, indices, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_indices_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_indices_random.pdf", bbox_inches="tight"
    )

    # set_wyckoffs_max_indices over volume

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    volumes = []
    for i, item in enumerate(icsd_rightly_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            volumes.append(icsd_rightly_volumes[i])
    plt.scatter(volumes, indices, color="g", s=0.5)
    indices = []
    volumes = []
    for i, item in enumerate(icsd_falsely_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            volumes.append(icsd_falsely_volumes[i])
    plt.scatter(volumes, indices, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_max_indices_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_max_indices_icsd.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    volumes = []
    for i, item in enumerate(random_rightly_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            volumes.append(random_rightly_volumes[i])
    plt.scatter(volumes, indices, color="g", s=0.5)
    indices = []
    volumes = []
    for i, item in enumerate(random_falsely_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            volumes.append(random_falsely_volumes[i])
    plt.scatter(volumes, indices, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_max_indices_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_set_wyckoffs_max_indices_random.pdf",
        bbox_inches="tight",
    )

    # set_wyckoffs_max_indices over NO_wyckoffs

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    NO_wyckoffs = []
    for i, item in enumerate(icsd_rightly_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            NO_wyckoffs.append(icsd_rightly_NO_wyckoffs[i])
    plt.scatter(NO_wyckoffs, indices, color="g", s=0.5)
    indices = []
    NO_wyckoffs = []
    for i, item in enumerate(icsd_falsely_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            NO_wyckoffs.append(icsd_falsely_NO_wyckoffs[i])
    plt.scatter(NO_wyckoffs, indices, color="r", s=0.5)
    plt.xlim(0, 100)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_NO_wyckoffs_set_wyckoffs_max_indices_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_NO_wyckoffs_set_wyckoffs_max_indices_icsd.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    indices = []
    NO_wyckoffs = []
    for i, item in enumerate(random_rightly_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            NO_wyckoffs.append(random_rightly_NO_wyckoffs[i])
    plt.scatter(NO_wyckoffs, indices, color="g", s=0.5)
    indices = []
    NO_wyckoffs = []
    for i, item in enumerate(random_falsely_set_wyckoffs_max_indices):
        if item is not None:
            indices.append(item)
            NO_wyckoffs.append(random_falsely_NO_wyckoffs[i])
    plt.scatter(NO_wyckoffs, indices, color="r", s=0.5)
    plt.xlim(0, 100)
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_NO_wyckoffs_set_wyckoffs_max_indices_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_NO_wyckoffs_set_wyckoffs_max_indices_random.pdf",
        bbox_inches="tight",
    )

    # structural_complexity over volume (shannon entropy of occupations)

    if analyse_complexity_ordering:

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )
        plt.scatter(
            icsd_rightly_volumes, icsd_rightly_structural_complexity, color="g", s=0.5
        )
        plt.scatter(
            icsd_falsely_volumes, icsd_falsely_structural_complexity, color="r", s=0.5
        )
        plt.xlim(0, 7000)
        plt.tight_layout()
        plt.savefig(
            f"{out_base}2D_volumes_structural_complexity_icsd_pub.pdf",
            bbox_inches="tight",
        )

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(
            f"{out_base}2D_volumes_structural_complexity_icsd.pdf",
            bbox_inches="tight",
        )

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )
        plt.scatter(
            random_rightly_volumes,
            random_rightly_structural_complexity,
            color="g",
            s=0.5,
        )
        plt.scatter(
            random_falsely_volumes,
            random_falsely_structural_complexity,
            color="r",
            s=0.5,
        )
        plt.xlim(0, 7000)
        plt.tight_layout()
        plt.savefig(
            f"{out_base}2D_volumes_structural_complexity_random_pub.pdf",
            bbox_inches="tight",
        )

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(
            f"{out_base}2D_volumes_structural_complexity_random.pdf",
            bbox_inches="tight",
        )

    # chemical_ordering over volume (shannon entropy of occupations)

    if analyse_complexity_ordering:

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )
        plt.scatter(
            icsd_rightly_volumes, icsd_rightly_chemical_ordering, color="g", s=0.5
        )
        plt.scatter(
            icsd_falsely_volumes, icsd_falsely_chemical_ordering, color="r", s=0.5
        )
        plt.xlim(0, 7000)
        plt.tight_layout()
        plt.savefig(
            f"{out_base}2D_volumes_chemical_ordering_icsd_pub.pdf",
            bbox_inches="tight",
        )

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(
            f"{out_base}2D_volumes_chemical_ordering_icsd.pdf",
            bbox_inches="tight",
        )

        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )
        plt.scatter(
            random_rightly_volumes, random_rightly_chemical_ordering, color="g", s=0.5
        )
        plt.scatter(
            random_falsely_volumes, random_falsely_chemical_ordering, color="r", s=0.5
        )
        plt.xlim(0, 7000)
        plt.tight_layout()
        plt.savefig(
            f"{out_base}2D_volumes_chemical_ordering_random_pub.pdf",
            bbox_inches="tight",
        )

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(
            f"{out_base}2D_volumes_chemical_ordering_random.pdf",
            bbox_inches="tight",
        )

    # number of unique elements over the volume

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(icsd_rightly_volumes, icsd_rightly_NO_elements, color="g", s=0.5)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_NO_elements, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_elements_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_elements_icsd.pdf",
        bbox_inches="tight",
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(random_rightly_volumes, random_rightly_NO_elements, color="g", s=0.5)
    plt.scatter(random_falsely_volumes, random_falsely_NO_elements, color="r", s=0.5)
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_elements_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_NO_unique_elements_random.pdf",
        bbox_inches="tight",
    )

    # sum of intensities over the volume

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        [item[0] for item in icsd_rightly_volumes_sum_of_intensities],
        [item[1] for item in icsd_rightly_volumes_sum_of_intensities],
        color="g",
        s=0.5,
    )
    plt.scatter(
        [item[0] for item in icsd_falsely_volumes_sum_of_intensities],
        [item[1] for item in icsd_falsely_volumes_sum_of_intensities],
        color="r",
        s=0.5,
    )
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_sum_of_intensities_icsd_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_sum_of_intensities_icsd.pdf", bbox_inches="tight"
    )

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.scatter(
        [item[0] for item in random_rightly_volumes_sum_of_intensities],
        [item[1] for item in random_rightly_volumes_sum_of_intensities],
        color="g",
        s=0.5,
    )
    plt.scatter(
        [item[0] for item in random_falsely_volumes_sum_of_intensities],
        [item[1] for item in random_falsely_volumes_sum_of_intensities],
        color="r",
        s=0.5,
    )
    plt.xlim(0, 7000)
    plt.tight_layout()
    plt.savefig(
        f"{out_base}2D_volumes_sum_of_intensities_random_pub.pdf",
        bbox_inches="tight",
    )

    plt.gcf().set_size_inches(
        figure_double_width * 0.95 * 0.5,
        figure_double_width * 0.7 * 0.5,
    )
    plt.savefig(
        f"{out_base}2D_volumes_sum_of_intensities_random.pdf",
        bbox_inches="tight",
    )

    ##### Angles 3D scatter plot

    plt.close("all")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        [item[0] for item in random_rightly_angles],
        [item[1] for item in random_rightly_angles],
        [item[2] for item in random_rightly_angles],
        label="Random rightly",
    )
    ax.scatter(
        [item[0] for item in random_falsely_angles],
        [item[1] for item in random_falsely_angles],
        [item[2] for item in random_falsely_angles],
        label="Random falsely",
    )
    ax.scatter(
        [item[0] for item in icsd_rightly_angles],
        [item[1] for item in icsd_rightly_angles],
        [item[2] for item in icsd_rightly_angles],
        label="ICSD rightly",
    )
    ax.scatter(
        [item[0] for item in icsd_falsely_angles],
        [item[1] for item in icsd_falsely_angles],
        [item[2] for item in icsd_falsely_angles],
        label="ICSD falsely",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{out_base}3D_angles_scatter.pdf",
        bbox_inches="tight",
    )
    # plt.show()

    ##### Lattice paras 3D scatter plot

    plt.close("all")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        [item[0] for item in random_rightly_lattice_paras],
        [item[1] for item in random_rightly_lattice_paras],
        [item[2] for item in random_rightly_lattice_paras],
        label="Random rightly",
    )
    ax.scatter(
        [item[0] for item in random_falsely_lattice_paras],
        [item[1] for item in random_falsely_lattice_paras],
        [item[2] for item in random_falsely_lattice_paras],
        label="Random falsely",
    )
    ax.scatter(
        [item[0] for item in icsd_rightly_lattice_paras],
        [item[1] for item in icsd_rightly_lattice_paras],
        [item[2] for item in icsd_rightly_lattice_paras],
        label="ICSD rightly",
    )
    ax.scatter(
        [item[0] for item in icsd_falsely_lattice_paras],
        [item[1] for item in icsd_falsely_lattice_paras],
        [item[2] for item in icsd_falsely_lattice_paras],
        label="ICSD falsely",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{out_base}3D_lattice_paras_scatter.pdf",
        bbox_inches="tight",
    )
    # plt.show()

    ################# hist plotting ################

    def create_histogram(
        tag,
        data_icsd,  # rightly, falsely
        data_random,  # rightly, falsely
        xlabel,
        labels,
        is_int=False,
        only_proportions=False,
        min_is_zero=True,
        fixed_x_min=None,
        fixed_y_max=None,
        weights_icsd=None,
        weights_random=None,
        N_bins_continuous=60,
        force_int_bins=False,
        zoom=False,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        zoom_value=None,
    ):
        # determine range on x axis:
        min = 10**9
        max = 0

        for item in [
            data_icsd[0] if data_icsd is not None else None,
            data_icsd[1] if data_icsd is not None else None,
            data_random[0] if data_random is not None else None,
            data_random[1] if data_random is not None else None,
        ]:

            if item is None:
                continue

            if len(item) == 0:
                continue

            for j in reversed(range(0, len(item))):
                if item[j] is None:
                    del item[j]

            new_min = np.min(item)
            new_max = np.max(item)

            if new_min < min:
                min = new_min

            if new_max > max:
                max = new_max

        if min == 10**9:
            return  # no data present

        if fixed_y_max is not None:
            max = fixed_y_max

        if fixed_x_min is not None:
            min = fixed_x_min

        if min_is_zero:
            min = 0

        if not is_int:
            bins = np.linspace(
                min,
                max,
                N_bins_continuous,
            )
        else:
            bins = (
                np.arange(
                    min,
                    max + 2,
                    1
                    if force_int_bins or ((max - min) < N_bins_continuous)
                    else int((max - min) / N_bins_continuous),
                )
                - 0.5
            )

        bin_width = bins[1] - bins[0]

        hists = []

        for i, data in enumerate([data_icsd, data_random]):

            if data is None:
                continue

            weights = [weights_icsd, weights_random][i]

            for j, item in enumerate(data):
                hist, edges = np.histogram(
                    item, bins, weights=weights[j] if weights is not None else None
                )
                hists.append(hist)

            # to handle rightly and falsely:
            total_hist = hists[-2] + hists[-1]

            hists[-2] = hists[-2] / total_hist
            hists[-1] = hists[-1] / total_hist

            hists[-2] = np.nan_to_num(hists[-2])
            hists[-1] = np.nan_to_num(hists[-1])

            if not only_proportions:  # scale with probability distribution
                total_hist = total_hist / (np.sum(total_hist) * bin_width)
                hists[-2] = hists[-2] * total_hist
                hists[-1] = hists[-1] * total_hist

        # Figure size
        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95 * 0.5,
                figure_double_width_pub * 0.7 * 0.5,
            )
        )

        ax1 = plt.gca()

        ax1.set_xlabel(xlabel)

        if not only_proportions:
            ax1.set_ylabel("Probability density")
        else:
            ax1.set_ylabel("proportion for each bin")

        if zoom:
            axins = zoomed_inset_axes(ax1, zoom_value, loc="upper right")

        counter = 0
        for i, data in enumerate([data_icsd, data_random]):

            if data is None:
                continue

            for ax in [ax1, axins] if zoom else [ax1]:

                if i == 0:

                    # falsely
                    h1 = ax.bar(
                        bins[:-1],
                        hists[counter * 2 + 1],  # height; 1,3
                        bottom=0,  # bottom
                        color="r",
                        label=labels[counter * 2 + 1],  # 1,3
                        width=bin_width,
                        align="edge",
                    )

                    # rightly
                    h2 = ax.bar(
                        bins[:-1],
                        hists[counter * 2],  # height; 0,2
                        bottom=hists[counter * 2 + 1],  # bottom; 1,3
                        color="g",
                        label=labels[counter * 2],  # 0,2
                        width=bin_width,
                        align="edge",
                    )

                elif i == 1:

                    # middle (falsely):
                    ax.step(
                        # bins[:-1],
                        bins[:],
                        # hists[counter * 2 + 1],
                        np.append(
                            hists[counter * 2 + 1], hists[counter * 2 + 1][-1]
                        ),  # 1,3
                        color="blueviolet",
                        label=labels[counter * 2 + 1],
                        where="post",
                    )

                    # top (rightly):
                    ax.step(
                        # bins[:-1],
                        bins[:],
                        # hists[counter * 2]
                        # + hists[counter * 2 + 1],  # top coordinate, not height
                        np.append(
                            (hists[counter * 2] + hists[counter * 2 + 1]),  # 0,2 + 1,3
                            (hists[counter * 2] + hists[counter * 2 + 1])[
                                -1
                            ],  # 0,2 + 1,3
                        ),  # top coordinate, not height
                        color="b",
                        label=labels[counter * 2],
                        where="post",
                    )

            counter += 1

        ymin, ymax = ax1.get_ylim()  # get current limits

        if min_is_zero:
            ax1.set_xlim(left=0, right=None)

        if not only_proportions:
            ax1.set_ylim(bottom=-1 * ymax / 25, top=ymax)
        else:
            ax1.set_ylim(bottom=-0.1, top=1.1)

        if zoom:

            # sub region of the original plot
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            plt.xticks(visible=False)
            plt.yticks(visible=False)

            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.legend()
        plt.tight_layout()

        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if zoom:
            plt.draw()

        plt.savefig(
            f"{out_base}{tag}{'_prop' if only_proportions else ''}_pub.pdf",
            bbox_inches="tight",
        )

        plt.gcf().set_size_inches(
            figure_double_width * 0.95 * 0.5,
            figure_double_width * 0.7 * 0.5,
        )
        plt.savefig(
            f"{out_base}{tag}{'_prop' if only_proportions else ''}.pdf",
            bbox_inches="tight",
        )

    # actual plotting:

    if False:
        volumes_too_high_icsd = [
            volume for volume in icsd_rightly_volumes if volume > 7000
        ]
        volumes_too_high_icsd.extend(
            [volume for volume in icsd_falsely_volumes if volume > 7000]
        )
        volumes_too_high_random = [
            volume for volume in random_rightly_volumes if volume > 7000
        ]
        volumes_too_high_random.extend(
            [volume for volume in random_falsely_volumes if volume > 7000]
        )

        print("Volumes too high icsd:")
        print(volumes_too_high_icsd)
        print("Volumes too high random:")
        print(volumes_too_high_random)

    for flag in [True, False]:
        create_histogram(
            "volumes",
            [icsd_rightly_volumes, icsd_falsely_volumes],
            [random_rightly_volumes, random_falsely_volumes],
            r"Volume / $Å^3$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
            fixed_x_min=0 if fix_important_ranges else None,
            fixed_y_max=7000 if fix_important_ranges else None,
            zoom=zoom if not flag else False,
            x1=1500,
            x2=3000,
            y1=0,
            y2=0.2 * 1e-3,
            zoom_value=3.0 * 1.35,
        )

    for flag in [True, False]:
        create_histogram(
            "angles_all",
            [
                [j for i in icsd_rightly_angles for j in i],
                [j for i in icsd_falsely_angles for j in i],
            ],
            [
                [j for i in random_rightly_angles for j in i],
                [j for i in random_falsely_angles for j in i],
            ],
            r"angles all / °",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "angle_0",
            [
                [item[0] for item in icsd_rightly_angles],
                [item[0] for item in icsd_falsely_angles],
            ],
            [
                [item[0] for item in random_rightly_angles],
                [item[0] for item in random_falsely_angles],
            ],
            r"angle 0 / °",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "angle_1",
            [
                [item[1] for item in icsd_rightly_angles],
                [item[1] for item in icsd_falsely_angles],
            ],
            [
                [item[1] for item in random_rightly_angles],
                [item[1] for item in random_falsely_angles],
            ],
            r"angle 1 / °",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "angle_2",
            [
                [item[2] for item in icsd_rightly_angles],
                [item[2] for item in icsd_falsely_angles],
            ],
            [
                [item[2] for item in random_rightly_angles],
                [item[2] for item in random_falsely_angles],
            ],
            r"angle 2 / °",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "denseness_factors",
            [
                [item for item in icsd_rightly_denseness_factors if item is not None],
                [item for item in icsd_falsely_denseness_factors if item is not None],
            ],
            [
                [item for item in random_rightly_denseness_factors if item is not None],
                [item for item in random_falsely_denseness_factors if item is not None],
            ],
            "Density factor",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
            fixed_x_min=0.0 if fix_important_ranges else None,
            fixed_y_max=4.0 if fix_important_ranges else None,
        )

    for flag in [True, False]:
        create_histogram(
            "corn_sizes",
            [
                [j for i in icsd_rightly_corn_sizes if i is not None for j in i],
                [j for i in icsd_falsely_corn_sizes if i is not None for j in i],
            ],
            [random_rightly_corn_sizes, random_falsely_corn_sizes],
            r"Corn size / nm",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=False,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_wyckoffs",
            [icsd_rightly_NO_wyckoffs, icsd_falsely_NO_wyckoffs],
            [random_rightly_NO_wyckoffs, random_falsely_NO_wyckoffs],
            "Number atoms in asym. unit",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
            force_int_bins=True,
            fixed_x_min=0 if fix_important_ranges else None,
            fixed_y_max=30 if fix_important_ranges else None,
            zoom=zoom if not flag else False,
            x1=10,
            x2=20,
            y1=0,
            y2=0.5 * 1e-1,
            zoom_value=2.2,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_elements",
            [icsd_rightly_NO_elements, icsd_falsely_NO_elements],
            [random_rightly_NO_elements, random_falsely_NO_elements],
            "Number of unique elements on wyckoff sites",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_all",
            [
                [j for i in icsd_rightly_lattice_paras if i is not None for j in i],
                [j for i in icsd_falsely_lattice_paras if i is not None for j in i],
            ],
            [
                [j for i in random_rightly_lattice_paras if i is not None for j in i],
                [j for i in random_falsely_lattice_paras if i is not None for j in i],
            ],
            r"Lattice parameters all / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
            fixed_x_min=0 if fix_important_ranges else None,
            fixed_y_max=55 if fix_important_ranges else None,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_0_over_1",
            [
                [item[0] / item[1] for item in icsd_rightly_lattice_paras],
                [item[0] / item[1] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[0] / item[1] for item in random_rightly_lattice_paras],
                [item[0] / item[1] for item in random_falsely_lattice_paras],
            ],
            "Lattice parameters 0 over 1",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_0_over_2",
            [
                [item[0] / item[2] for item in icsd_rightly_lattice_paras],
                [item[0] / item[2] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[0] / item[2] for item in random_rightly_lattice_paras],
                [item[0] / item[2] for item in random_falsely_lattice_paras],
            ],
            "Lattice parameters 0 over 2",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_1_over_2",
            [
                [item[1] / item[2] for item in icsd_rightly_lattice_paras],
                [item[1] / item[2] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[1] / item[2] for item in random_rightly_lattice_paras],
                [item[1] / item[2] for item in random_falsely_lattice_paras],
            ],
            "Lattice parameters 1 over 2",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_para_0",
            [
                [item[0] for item in icsd_rightly_lattice_paras],
                [item[0] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[0] for item in random_rightly_lattice_paras],
                [item[0] for item in random_falsely_lattice_paras],
            ],
            r"Lattice parameter 0 / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_para_1",
            [
                [item[1] for item in icsd_rightly_lattice_paras],
                [item[1] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[1] for item in random_rightly_lattice_paras],
                [item[1] for item in random_falsely_lattice_paras],
            ],
            r"Lattice parameter 1 / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_para_2",
            [
                [item[2] for item in icsd_rightly_lattice_paras],
                [item[2] for item in icsd_falsely_lattice_paras],
            ],
            [
                [item[2] for item in random_rightly_lattice_paras],
                [item[2] for item in random_falsely_lattice_paras],
            ],
            r"Lattice parameter 2 / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_max",
            [
                icsd_rightly_max_lattice_paras,
                icsd_falsely_max_lattice_paras,
            ],
            [random_rightly_max_lattice_paras, random_falsely_max_lattice_paras],
            r"max lattice parameter / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras_min",
            [
                icsd_rightly_min_lattice_paras,
                icsd_falsely_min_lattice_paras,
            ],
            [random_rightly_min_lattice_paras, random_falsely_min_lattice_paras],
            r"min lattice parameter / $Å$",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "occupancies_weighted",
            [
                [j for i in icsd_rightly_occupancies if i is not None for j in i],
                [j for i in icsd_falsely_occupancies if i is not None for j in i],
            ],
            None,
            "occupancy",
            [
                "ICSD correctly",
                "ICSD incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
            weights_icsd=[
                [j for i in icsd_rightly_occupancies_weights for j in i],
                [j for i in icsd_falsely_occupancies_weights for j in i],
            ],
        )

    for flag in [True, False]:
        create_histogram(
            "occupancies",
            [
                [j for i in icsd_rightly_occupancies if i is not None for j in i],
                [j for i in icsd_falsely_occupancies if i is not None for j in i],
            ],
            None,
            "occupancy",
            [
                "ICSD correctly",
                "ICSD incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "element_repetitions",
            [
                [
                    j
                    for i in icsd_rightly_element_repetitions
                    if i is not None
                    for j in i
                ],
                [
                    j
                    for i in icsd_falsely_element_repetitions
                    if i is not None
                    for j in i
                ],
            ],
            [
                [
                    j
                    for i in random_rightly_element_repetitions
                    if i is not None
                    for j in i
                ],
                [
                    j
                    for i in random_falsely_element_repetitions
                    if i is not None
                    for j in i
                ],
            ],
            "Number of element repetitions on wyckoff sites",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "wyckoff_repetitions",
            [
                [
                    j
                    for i in icsd_rightly_wyckoff_repetitions
                    if i is not None
                    for j in i
                ],
                [
                    j
                    for i in icsd_falsely_wyckoff_repetitions
                    if i is not None
                    for j in i
                ],
            ],
            [
                [
                    j
                    for i in random_rightly_wyckoff_repetitions
                    if i is not None
                    for j in i
                ],
                [
                    j
                    for i in random_falsely_wyckoff_repetitions
                    if i is not None
                    for j in i
                ],
            ],
            "Number of wyckoff repetitions per element",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_unique_wyckoffs",
            [
                icsd_rightly_NO_unique_wyckoffs,
                icsd_falsely_NO_unique_wyckoffs,
            ],
            [
                random_rightly_NO_unique_wyckoffs,
                random_falsely_NO_unique_wyckoffs,
            ],
            "Number of unique wyckoff sites",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_unique_wyckoffs_summed_over_els",
            [
                icsd_rightly_NO_unique_wyckoffs_summed_over_els,
                icsd_falsely_NO_unique_wyckoffs_summed_over_els,
            ],
            [
                random_rightly_NO_unique_wyckoffs_summed_over_els,
                random_falsely_NO_unique_wyckoffs_summed_over_els,
            ],
            "Number of unique wyckoff sites summed over elements",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_atoms",
            [
                icsd_rightly_NO_atoms,
                icsd_falsely_NO_atoms,
            ],
            [random_rightly_NO_atoms, random_falsely_NO_atoms],
            "Number of atoms in the unit cell",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "max_Zs",
            [
                [item for item in icsd_rightly_max_Zs if item is not None],
                [item for item in icsd_falsely_max_Zs if item is not None],
            ],
            [
                [item for item in random_rightly_max_Zs if item is not None],
                [item for item in random_falsely_max_Zs if item is not None],
            ],
            "max Z",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "sum_of_intensities",
            [
                [item[1] for item in icsd_rightly_volumes_sum_of_intensities],
                [item[1] for item in icsd_falsely_volumes_sum_of_intensities],
            ],
            [
                [item[1] for item in random_rightly_volumes_sum_of_intensities],
                [item[1] for item in random_falsely_volumes_sum_of_intensities],
            ],
            r"sum of intensities",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "COM",
            [
                icsd_rightly_COM,
                icsd_falsely_COM,
            ],
            [
                random_rightly_COM,
                random_falsely_COM,
            ],
            r"COM",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "sum_cov_vols",
            [
                icsd_rightly_sum_cov_vols,
                icsd_falsely_sum_cov_vols,
            ],
            [
                random_rightly_sum_cov_vols,
                random_falsely_sum_cov_vols,
            ],
            r"sum of covalent volumes",
            [
                "ICSD correctly",
                "ICSD incorrectly",
                "Synthetic correctly",
                "Synthetic incorrectly",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    if show_sample_xrds:
        for flag in [True, False]:
            create_histogram(
                "max_unscaled_intensity_angle",
                [
                    [
                        item
                        for item in icsd_rightly_max_unscaled_intensity
                        if item < 10**11
                    ],
                    [
                        item
                        for item in icsd_falsely_max_unscaled_intensity
                        if item < 10**11
                    ],
                ],
                [
                    [
                        item
                        for item in random_rightly_max_unscaled_intensity
                        if item < 10**11
                    ],
                    [
                        item
                        for item in random_falsely_max_unscaled_intensity
                        if item < 10**11
                    ],
                ],
                r"max_unscaled_intensity_angle",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                min_is_zero=True,
                fixed_y_max=10**10,
            )

        for flag in [True, False]:
            create_histogram(
                "max_unscaled_intensity_angle_weighted",
                [
                    [
                        item
                        for item in icsd_rightly_max_unscaled_intensity_weighted
                        if item < 10**11
                    ],
                    [
                        item
                        for item in icsd_falsely_max_unscaled_intensity_weighted
                        if item < 10**11
                    ],
                ],
                [
                    [
                        item
                        for item in random_rightly_max_unscaled_intensity_weighted
                        if item < 10**11
                    ],
                    [
                        item
                        for item in random_falsely_max_unscaled_intensity_weighted
                        if item < 10**11
                    ],
                ],
                r"max_unscaled_intensity_angle_weighted",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                min_is_zero=True,
            )

    if not spgs_to_analyze is None and len(spgs_to_analyze) == 1:
        for flag in [True, False]:
            create_histogram(
                "set_wyckoff_indices",
                [
                    [
                        j
                        for i in icsd_rightly_set_wyckoffs_indices
                        if i is not None
                        for j in i
                    ],
                    [
                        j
                        for i in icsd_falsely_set_wyckoffs_indices
                        if i is not None
                        for j in i
                    ],
                ],
                [
                    [
                        j
                        for i in random_rightly_set_wyckoffs_indices
                        if i is not None
                        for j in i
                    ],
                    [
                        j
                        for i in random_falsely_set_wyckoffs_indices
                        if i is not None
                        for j in i
                    ],
                ],
                "wyckoff index",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=True,
                only_proportions=flag,
                min_is_zero=True,
            )

        for flag in [True, False]:
            create_histogram(
                "set_wyckoff_max_indices",
                [
                    [
                        item
                        for item in icsd_rightly_set_wyckoffs_max_indices
                        if item is not None
                    ],
                    [
                        item
                        for item in icsd_falsely_set_wyckoffs_max_indices
                        if item is not None
                    ],
                ],
                [
                    [
                        item
                        for item in random_rightly_set_wyckoffs_max_indices
                        if item is not None
                    ],
                    [
                        item
                        for item in random_falsely_set_wyckoffs_max_indices
                        if item is not None
                    ],
                ],
                "max wyckoff index",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=True,
                only_proportions=flag,
                min_is_zero=True,
            )

    if analyse_complexity_ordering:

        for flag in [True, False]:
            create_histogram(
                "structural_complexity",
                [
                    [
                        item
                        for item in icsd_rightly_structural_complexity
                        if item is not None
                    ],
                    [
                        item
                        for item in icsd_falsely_structural_complexity
                        if item is not None
                    ],
                ],
                [
                    [
                        item
                        for item in random_rightly_structural_complexity
                        if item is not None
                    ],
                    [
                        item
                        for item in random_falsely_structural_complexity
                        if item is not None
                    ],
                ],
                r"structural complexity",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                min_is_zero=True,
            )

        for flag in [True, False]:
            create_histogram(
                "chemical ordering",
                [
                    [
                        item
                        for item in icsd_rightly_chemical_ordering
                        if item is not None
                    ],
                    [
                        item
                        for item in icsd_falsely_chemical_ordering
                        if item is not None
                    ],
                ],
                [
                    [
                        item
                        for item in random_rightly_chemical_ordering
                        if item is not None
                    ],
                    [
                        item
                        for item in random_falsely_chemical_ordering
                        if item is not None
                    ],
                ],
                r"chemical ordering",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                min_is_zero=True,
            )

    # Info about wyckoff positions in cif file format:
    # => where in the cif file is written what kind of wyckoff site we are dealing with?
    # The general wyckoff site is always written in the cif file!
    # This is because the special ones are only special cases of the general wyckoff position!
    # Only the general wyckoff position is needed to generate all the coordinates.

    ################# Analysing additional structural features (Magpie) ################

    # mean_effective_coord_number,
    # mean_coord_number,
    # L2_norm,
    # L3_norm,
    # max_packing_efficiency,
    # mean_bond_length_variation,

    if compute_magpie_features:

        for flag in [True, False]:
            create_histogram(
                "mean_effective_coord_number",
                [
                    [item[0] for item in icsd_rightly_magpie_features],
                    [item[0] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[0] for item in random_rightly_magpie_features],
                    [item[0] for item in random_falsely_magpie_features],
                ],
                r"mean effective coordination number",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )

        for flag in [True, False]:
            create_histogram(
                "mean_coord_number",
                [
                    [item[1] for item in icsd_rightly_magpie_features],
                    [item[1] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[1] for item in random_rightly_magpie_features],
                    [item[1] for item in random_falsely_magpie_features],
                ],
                r"mean coordination number",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )

        for flag in [True, False]:
            create_histogram(
                "stoichometry_L2_norm",
                [
                    [item[2] for item in icsd_rightly_magpie_features],
                    [item[2] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[2] for item in random_rightly_magpie_features],
                    [item[2] for item in random_falsely_magpie_features],
                ],
                r"stoichometry L2 norm",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )

        for flag in [True, False]:
            create_histogram(
                "stoichometry_L3_norm",
                [
                    [item[3] for item in icsd_rightly_magpie_features],
                    [item[3] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[3] for item in random_rightly_magpie_features],
                    [item[3] for item in random_falsely_magpie_features],
                ],
                r"stoichometry L3 norm",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )

        for flag in [True, False]:
            create_histogram(
                "max_packing_efficiency",
                [
                    [item[4] for item in icsd_rightly_magpie_features],
                    [item[4] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[4] for item in random_rightly_magpie_features],
                    [item[4] for item in random_falsely_magpie_features],
                ],
                r"max packing efficiency",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )

        for flag in [True, False]:
            create_histogram(
                "mean_bond_length_variation",
                [
                    [item[5] for item in icsd_rightly_magpie_features],
                    [item[5] for item in icsd_falsely_magpie_features],
                ],
                [
                    [item[5] for item in random_rightly_magpie_features],
                    [item[5] for item in random_falsely_magpie_features],
                ],
                r"mean bond length variation",
                [
                    "ICSD correctly",
                    "ICSD incorrectly",
                    "Synthetic correctly",
                    "Synthetic incorrectly",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )
