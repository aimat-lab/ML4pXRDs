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
from train_dataset.utils.analyse_magpie import get_magpie_features
from train_dataset.utils.denseness_factor import get_denseness_factor

if __name__ == "__main__":

    if len(sys.argv) > 2:
        # Probably running directly from the training script, so take arguments

        in_base = sys.argv[1]
        tag = sys.argv[2]

        spgs_to_analyze = [int(spg) for spg in sys.argv[3:]]

        if len(spgs_to_analyze) == 0:
            spgs_to_analyze = None  # all spgs
        else:
            tag += "/" + "_".join([str(spg) for spg in spgs_to_analyze])

    else:

        # in_base = "classifier_spgs/runs_from_cluster/initial_tests/10-03-2022_14-34-51/"
        # in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/initial_tests/17-03-2022_10-11-11/"
        # tag = "magpie_10-03-2022_14-34-51"
        # tag = "volumes_densenesses_4-spg"
        # tag = "look_at_structures"
        in_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/initial_tests/20-03-2022_02-06-52/"

        tag = "4-spg-2D-scatters"
        # tag = "volumes_densenesses_2-spg_test/15"

        spgs_to_analyze = [14, 104, 176, 129]
        # spgs_to_analyze = [15]
        # spgs_to_analyze = None  # analyse all space groups; alternative: list of spgs

    compute_magpie_features = False

    show_sample_structures = False
    samples_to_show_icsd = 50
    counter_shown_icsd_rightly = 0
    counter_shown_icsd_falsely = 0
    counter_shown_random_rightly = 0
    counter_shown_random_falsely = 0

    out_base = "comparison_plots/" + tag + "/"
    os.system("mkdir -p " + out_base)

    if show_sample_structures:
        os.system("mkdir -p " + out_base + "icsd_rightly_structures")
        os.system("mkdir -p " + out_base + "icsd_falsely_structures")
        os.system("mkdir -p " + out_base + "random_rightly_structures")
        os.system("mkdir -p " + out_base + "random_falsely_structures")

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
    random_labels = [spgs[index] for index in random_labels]

    with open(in_base + "rightly_falsely_icsd.pickle", "rb") as file:
        rightly_indices_icsd, falsely_indices_icsd = pickle.load(file)

    with open(in_base + "rightly_falsely_random.pickle", "rb") as file:
        rightly_indices_random, falsely_indices_random = pickle.load(file)

    # limit the range:
    if True:
        random_crystals = random_crystals[0:600]
        random_labels = random_labels[0:600]
        random_variations = random_variations[0:600]
        icsd_crystals = icsd_crystals[0:600]
        icsd_labels = icsd_labels[0:600]
        icsd_variations = icsd_variations[0:600]
        icsd_metas = icsd_metas[0:600]

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

    # this is actually not really needed, but just in case...
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

        (
            is_pure,
            NO_wyckoffs,
            elements,
            occupancies,
            wyckoff_repetitions,
        ) = sim.get_wyckoff_info(icsd_metas[i][0])

        elements_unique = np.unique(elements)

        icsd_NO_wyckoffs.append(NO_wyckoffs)
        icsd_NO_elements.append(len(elements_unique))
        icsd_occupancies.append(occupancies)
        icsd_occupancies_weights.append([1 / len(occupancies)] * len(occupancies))

        icsd_wyckoff_repetitions.append(wyckoff_repetitions)

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

        for key in wyckoffs_per_element.keys():
            wyckoffs_unique = np.unique(wyckoffs_per_element[key])

            for item in wyckoffs_unique:
                wyckoff_repetitions.append(
                    np.sum(np.array(wyckoffs_per_element[key]) == item)
                )

        return len(struc.atom_sites), elements, wyckoff_repetitions

    random_NO_wyckoffs = []
    random_NO_elements = []
    random_element_repetitions = []
    random_wyckoff_repetitions = []

    for i in range(0, len(random_variations)):

        print(f"Processing random: {i} of {len(random_variations)}")

        success = True
        try:
            NO_wyckoffs, elements, wyckoff_repetitions = get_wyckoff_info(
                random_crystals[i]
            )
        except Exception as ex:
            print(ex)
            success = False

        if success:

            elements_unique = np.unique(elements)

            random_NO_wyckoffs.append(NO_wyckoffs)
            random_NO_elements.append(len(elements_unique))

            reps = []
            for el in elements_unique:
                reps.append(np.sum(np.array(elements) == el))
            random_element_repetitions.append(reps)

            random_wyckoff_repetitions.append(wyckoff_repetitions)

        else:

            random_NO_wyckoffs.append(None)
            random_NO_elements.append(None)
            random_element_repetitions.append(None)
            random_wyckoff_repetitions.append(None)

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

    for i in falsely_indices_icsd:

        index = int(i / 5)

        if index < len(icsd_crystals) and (
            spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze
        ):

            structure = icsd_crystals[index]

            if (
                show_sample_structures
                and counter_shown_icsd_falsely < samples_to_show_icsd
                and structure.is_ordered
                and (i % 5) == 0
            ):
                counter_shown_icsd_falsely += 1
                ase_struc = AseAtomsAdaptor.get_atoms(structure)
                write(
                    f"{out_base}icsd_falsely_structures/{icsd_metas[index][0]}.png",
                    ase_struc,
                )

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
                [structure.lattice.a, structure.lattice.b, structure.lattice.b]
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

        index = int(i / 5)

        if index < len(icsd_crystals) and (
            spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze
        ):

            structure = icsd_crystals[index]

            if (
                show_sample_structures
                and counter_shown_icsd_rightly < samples_to_show_icsd
                and structure.is_ordered
                and (i % 5) == 0
            ):
                counter_shown_icsd_rightly += 1
                ase_struc = AseAtomsAdaptor.get_atoms(structure)
                write(
                    f"{out_base}icsd_rightly_structures/{icsd_metas[index][0]}.png",
                    ase_struc,
                )

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
                [structure.lattice.a, structure.lattice.b, structure.lattice.b]
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
                counter_shown_random_falsely += 1
                ase_struc = AseAtomsAdaptor.get_atoms(structure)
                write(
                    f"{out_base}random_falsely_structures/{counter_shown_random_falsely}.png",
                    ase_struc,
                )

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
                [structure.lattice.a, structure.lattice.b, structure.lattice.c]
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
                counter_shown_random_rightly += 1
                ase_struc = AseAtomsAdaptor.get_atoms(structure)
                write(
                    f"{out_base}random_rightly_structures/{counter_shown_random_rightly}.png",
                    ase_struc,
                )

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
                [structure.lattice.a, structure.lattice.b, structure.lattice.c]
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

    ################# volumes_denseness_factors ################

    plt.figure()
    plt.scatter(icsd_rightly_volumes, icsd_rightly_denseness_factors, color="g", s=1)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_denseness_factors, color="r", s=1)
    plt.xlim(0, 7000)
    plt.ylim(0.5, 3.3)
    plt.savefig(
        f"{out_base}2D_volumes_densenesses_icsd.png", bbox_inches="tight", dpi=300
    )

    plt.figure()
    plt.scatter(
        random_rightly_volumes, random_rightly_denseness_factors, color="g", s=1
    )
    plt.scatter(
        random_falsely_volumes, random_falsely_denseness_factors, color="r", s=1
    )
    plt.xlim(0, 7000)
    plt.ylim(0.5, 3.3)
    plt.savefig(
        f"{out_base}2D_volumes_densenesses_random.png", bbox_inches="tight", dpi=300
    )

    plt.figure()
    plt.scatter(
        icsd_rightly_volumes, [item[0] for item in icsd_rightly_angles], color="g", s=1
    )
    plt.scatter(
        icsd_rightly_volumes, [item[1] for item in icsd_rightly_angles], color="g", s=1
    )
    plt.scatter(
        icsd_rightly_volumes, [item[2] for item in icsd_rightly_angles], color="g", s=1
    )
    plt.scatter(
        icsd_falsely_volumes, [item[0] for item in icsd_falsely_angles], color="r", s=1
    )
    plt.scatter(
        icsd_falsely_volumes, [item[1] for item in icsd_falsely_angles], color="r", s=1
    )
    plt.scatter(
        icsd_falsely_volumes, [item[2] for item in icsd_falsely_angles], color="r", s=1
    )
    plt.ylim(80, 140)
    plt.xlim(0, 7000)
    plt.savefig(f"{out_base}2D_volumes_angles_icsd.png", bbox_inches="tight", dpi=300)

    plt.figure()
    plt.scatter(
        random_rightly_volumes,
        [item[0] for item in random_rightly_angles],
        color="g",
        s=1,
    )
    plt.scatter(
        random_rightly_volumes,
        [item[1] for item in random_rightly_angles],
        color="g",
        s=1,
    )
    plt.scatter(
        random_rightly_volumes,
        [item[2] for item in random_rightly_angles],
        color="g",
        s=1,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[0] for item in random_falsely_angles],
        color="r",
        s=1,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[1] for item in random_falsely_angles],
        color="r",
        s=1,
    )
    plt.scatter(
        random_falsely_volumes,
        [item[2] for item in random_falsely_angles],
        color="r",
        s=1,
    )
    plt.ylim(80, 140)
    plt.xlim(0, 7000)
    plt.savefig(f"{out_base}2D_volumes_angles_random.png", bbox_inches="tight", dpi=300)

    plt.figure()
    plt.scatter(icsd_rightly_volumes, icsd_rightly_density, color="g", s=1)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_density, color="r", s=1)
    plt.xlim(0, 7000)
    plt.ylim(0, 17.5)
    plt.savefig(f"{out_base}2D_volumes_density_icsd.png", bbox_inches="tight", dpi=300)

    plt.figure()
    plt.scatter(random_rightly_volumes, random_rightly_density, color="g", s=1)
    plt.scatter(random_falsely_volumes, random_falsely_density, color="r", s=1)
    plt.xlim(0, 7000)
    plt.ylim(0, 17.5)
    plt.savefig(
        f"{out_base}2D_volumes_density_random.png", bbox_inches="tight", dpi=300
    )

    plt.figure()
    plt.scatter(
        icsd_rightly_sum_cov_vols, icsd_rightly_denseness_factors, color="g", s=1
    )
    plt.scatter(
        icsd_falsely_sum_cov_vols, icsd_falsely_denseness_factors, color="r", s=1
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 3.5)
    plt.savefig(
        f"{out_base}2D_sum_cov_vols_denseness_icsd.png", bbox_inches="tight", dpi=300
    )

    plt.figure()
    plt.scatter(
        random_rightly_sum_cov_vols, random_rightly_denseness_factors, color="g", s=1
    )
    plt.scatter(
        random_falsely_sum_cov_vols, random_falsely_denseness_factors, color="r", s=1
    )
    plt.xlim(0, 7000)
    plt.ylim(0, 3.5)
    plt.savefig(
        f"{out_base}2D_sum_cov_vols_denseness_random.png", bbox_inches="tight", dpi=300
    )

    plt.figure()
    plt.scatter(icsd_rightly_volumes, icsd_rightly_NO_atoms, color="g", s=1)
    plt.scatter(icsd_falsely_volumes, icsd_falsely_NO_atoms, color="r", s=1)
    plt.xlim(0, 7000)
    plt.ylim(0, 400)
    plt.savefig(f"{out_base}2D_volumes_NO_atoms_icsd.png", bbox_inches="tight", dpi=300)

    plt.figure()
    plt.scatter(random_rightly_volumes, random_rightly_NO_atoms, color="g", s=1)
    plt.scatter(random_falsely_volumes, random_falsely_NO_atoms, color="r", s=1)
    plt.xlim(0, 7000)
    plt.ylim(0, 400)
    plt.savefig(
        f"{out_base}2D_volumes_NO_atoms_random.png", bbox_inches="tight", dpi=300
    )

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

            new_min = np.min(item)
            new_max = np.max(item)

            if new_min < min:
                min = new_min

            if new_max > max:
                max = new_max

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
                    if (max - min) < N_bins_continuous
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
        plt.figure()
        ax1 = plt.gca()

        ax1.set_xlabel(xlabel)

        if not only_proportions:
            ax1.set_ylabel("probability density")
        else:
            ax1.set_ylabel("proportion for each bin")

        counter = 0
        for i, data in enumerate([data_icsd, data_random]):

            if data is None:
                continue

            if i == 0:

                # falsely
                h1 = ax1.bar(
                    bins[:-1],
                    hists[counter * 2 + 1],  # height; 1,3
                    bottom=0,  # bottom
                    color="r",
                    label=labels[counter * 2 + 1],  # 1,3
                    width=bin_width,
                    align="edge",
                )

                # rightly
                h2 = ax1.bar(
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
                ax1.step(
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
                ax1.step(
                    # bins[:-1],
                    bins[:],
                    # hists[counter * 2]
                    # + hists[counter * 2 + 1],  # top coordinate, not height
                    np.append(
                        (hists[counter * 2] + hists[counter * 2 + 1]),  # 0,2 + 1,3
                        (hists[counter * 2] + hists[counter * 2 + 1])[-1],  # 0,2 + 1,3
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

        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{out_base}{tag}{'_prop' if only_proportions else ''}.png",
            bbox_inches="tight",
        )

    # actual plotting:

    for flag in [True, False]:
        create_histogram(
            "volumes",
            [icsd_rightly_volumes, icsd_falsely_volumes],
            [random_rightly_volumes, random_falsely_volumes],
            r"volume / $Å^3$",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "angles",
            [
                [j for i in icsd_rightly_angles for j in i],
                [j for i in icsd_falsely_angles for j in i],
            ],
            [
                [j for i in random_rightly_angles for j in i],
                [j for i in random_falsely_angles for j in i],
            ],
            r"angle / °",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
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
            "denseness factor",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "corn_sizes",
            [
                [j for i in icsd_rightly_corn_sizes for j in i],
                [j for i in icsd_falsely_corn_sizes for j in i],
            ],
            [random_rightly_corn_sizes, random_falsely_corn_sizes],
            "corn size",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
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
            "Number of set wyckoff sites",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "NO_elements",
            [icsd_rightly_NO_elements, icsd_falsely_NO_elements],
            [random_rightly_NO_elements, random_falsely_NO_elements],
            "Number of unique elements on wyckoff sites",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "lattice_paras",
            [
                [j for i in icsd_rightly_lattice_paras for j in i],
                [j for i in icsd_falsely_lattice_paras for j in i],
            ],
            [
                [j for i in random_rightly_lattice_paras for j in i],
                [j for i in random_falsely_lattice_paras for j in i],
            ],
            r"lattice parameter / $Å$",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
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
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
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
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "occupancies_weighted",
            [
                [j for i in icsd_rightly_occupancies for j in i],
                [j for i in icsd_falsely_occupancies for j in i],
            ],
            None,
            "occupancy",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
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
                [j for i in icsd_rightly_occupancies for j in i],
                [j for i in icsd_falsely_occupancies for j in i],
            ],
            None,
            "occupancy",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
            ],
            is_int=False,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "element_repetitions",
            [
                [j for i in icsd_rightly_element_repetitions for j in i],
                [j for i in icsd_falsely_element_repetitions for j in i],
            ],
            [
                [j for i in random_rightly_element_repetitions for j in i],
                [j for i in random_falsely_element_repetitions for j in i],
            ],
            "Number of element repetitions on wyckoff sites",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=True,
            only_proportions=flag,
            min_is_zero=True,
        )

    for flag in [True, False]:
        create_histogram(
            "wyckoff_repetitions",
            [
                [j for i in icsd_rightly_wyckoff_repetitions for j in i],
                [j for i in icsd_falsely_wyckoff_repetitions for j in i],
            ],
            [
                [j for i in random_rightly_wyckoff_repetitions for j in i],
                [j for i in random_falsely_wyckoff_repetitions for j in i],
            ],
            "Number of wyckoff repetitions per element",
            [
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
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
                "ICSD correctly classified",
                "ICSD incorrectly classified",
                "Random correctly classified",
                "Random incorrectly classified",
            ],
            is_int=True,
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
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
                    "ICSD correctly classified",
                    "ICSD incorrectly classified",
                    "Random correctly classified",
                    "Random incorrectly classified",
                ],
                is_int=False,
                only_proportions=flag,
                N_bins_continuous=12,
            )
