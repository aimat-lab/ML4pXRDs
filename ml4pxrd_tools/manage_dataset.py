"""
This module contains functions to handle the generation and usage of the dataset
and extracted statistics.

To generate a new dataset with prototype-based split, you first have to change
`path_to_icsd_directory_cluster` or `path_to_icsd_directory_local` (depends on
if you run this script on a cluster using slurm or not) in this script. It
should point to your directory containing the ICSD database. Furthermore, you
first need to run the simulation of the ICSD data (see README.md) and point
`path_to_patterns` (see below) to the directory containing your simulated 
patterns.

Then, you can run this file to generate the dataset: `python manage_dataset.py`
"""

import os
from ml4pxrd_tools.simulation.icsd_simulator import ICSDSimulator
import math
from sklearn.model_selection import GroupShuffleSplit
import time
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from pyxtal import pyxtal
from training.analysis.denseness_factor import get_denseness_factor
import pickle
from glob import glob
from scipy.stats import kde
import statsmodels.api as sm
from pyxtal.symmetry import get_pbc_and_lattice
from ml4pxrd_tools.generation.all_elements import all_elements

# Path to the ICSD directory that contains the "ICSD_data_from_API.csv" file
# and the "cif" directory (which contains all the ICSD cif files)
# We provide two separate variables for local execution and execution on
# a cluster using slurm.
path_to_icsd_directory_local = os.path.expanduser("~/Dokumente/Big_Files/ICSD/")
path_to_icsd_directory_cluster = os.path.expanduser("~/Databases/ICSD/")

path_to_patterns = (
    "patterns/icsd_vecsei/"  # relative to the main directory of this repository
)


def get_wyckoff_info(pyxtal_crystal):
    """Returns number of atoms in asymmetric unit and list of
    elements in asymmetric unit.

    Args:
        pyxtal_crystal (pyxtal.pyxtal): Pyxtal crystal structure object

    Returns:
        tuple: (number of atoms in asymmetric unit, list of elements in asymmetric unit)
    """

    elements = []

    for site in pyxtal_crystal.atom_sites:
        specie_str = str(site.specie)
        elements.append(specie_str)

    return len(pyxtal_crystal.atom_sites), elements


def prepare_dataset(per_element=False, max_volume=7000, max_NO_wyckoffs=100):
    """Prepare a dataset for training on patterns from synthetic crystals and
    testing on ICSD patterns. This includes both the generation of statistical
    data extracted from the ICSD, as well as the training dataset split. We use
    a 70:30 split while we ensure that the same structure type is only either in
    the statistics dataset or the test dataset (see description in our
    publication). In order to run this function, simulation data from the ICSD
    is needed. Please refer to ml4pxrd_tools.simulation.icsd_simulator.py to
    simulate the ICSD patterns. The final dataset split including statistics
    information will be saved in the directory `prepared_dataset` in the
    main directory of the repository.

    Args:
        per_element (bool, optional): If this setting is True,
        NO_repetitions_prob_per_spg_per_element and
        probability_per_spg_per_element_per_wyckoff
            are indexed using the element. If False, they are independent of the
            element. Defaults to False.
        test_max_volume (int, optional): Maximum volume in statistics and test
        dataset. Defaults to 7000. test_max_NO_wyckoffs (int, optional): Maximum
        number of atoms in asymmetric unit in statistics and test dataset.
        Defaults to 100.

    Returns:
        bool|None: True if successful, else None.
    """

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prepared_dataset",
    )

    if os.path.exists(output_dir):
        print("Please remove existing prepared_dataset directory first.")
        return None

    spgs = range(1, 231)  # all spgs

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        path_to_icsd_directory = path_to_icsd_directory_cluster
    else:
        path_to_icsd_directory = path_to_icsd_directory_local

    sim = ICSDSimulator(
        os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
        os.path.join(path_to_icsd_directory, "cif/"),
        output_dir=path_to_patterns,
    )

    sim.load(load_patterns_angles_intensities=False)

    ##### Statistics / test split:

    group_labels = []

    counter = 0

    for i, meta in enumerate(sim.sim_metas):

        print(f"{i/len(sim.sim_metas)*100}%")

        index = sim.icsd_ids.index(meta[0])

        group_label = sim.icsd_structure_types[index]

        if not isinstance(group_label, str):
            if math.isnan(group_label):
                group_label = f"alone_{counter}"
                counter += 1
            else:
                raise Exception("Something went wrong.")

        group_labels.append(group_label)

    gss = GroupShuffleSplit(1, test_size=0.3, train_size=0.7)

    train_metas_splitted_indices, test_metas_splitted_indices = list(
        gss.split(X=[item[0] for item in sim.sim_metas], groups=group_labels)
    )[0]

    train_metas_splitted = [sim.sim_metas[i][0] for i in train_metas_splitted_indices]
    test_metas_splitted = [sim.sim_metas[i][0] for i in test_metas_splitted_indices]

    # Check if they are distinct (regarding the structure types)

    prototypes_test = [
        sim.icsd_structure_types[sim.icsd_ids.index(meta)]
        for meta in test_metas_splitted
    ]

    overlap_counter = 0
    for meta in train_metas_splitted:
        prototype_statistics = sim.icsd_structure_types[sim.icsd_ids.index(meta)]

        if meta in test_metas_splitted or (
            isinstance(prototype_statistics, str)
            and prototype_statistics in prototypes_test
        ):
            overlap_counter += 1
    print(
        f"{overlap_counter} of {len(train_metas_splitted)} samples in statistics dataset have a prototype that also exists in the test dataset.",
        flush=True,
    )  # This should be zero, otherwise something went wrong!

    if overlap_counter > 0:
        raise Exception("Something went wrong while splitting train / test.")

    statistics_crystals = []
    statistics_metas = []
    statistics_variations = []
    statistics_labels = []

    test_crystals = []
    test_metas = []
    test_variations = []
    test_labels = []

    for i, meta in enumerate(sim.sim_metas):
        if meta[0] in train_metas_splitted:
            statistics_crystals.append(sim.sim_crystals[i])
            statistics_metas.append(sim.sim_metas[i])
            statistics_variations.append(sim.sim_variations[i])
            statistics_labels.append(sim.sim_labels[i])
        elif meta[0] in test_metas_splitted:
            test_crystals.append(sim.sim_crystals[i])
            test_metas.append(sim.sim_metas[i])
            test_variations.append(sim.sim_variations[i])
            test_labels.append(sim.sim_labels[i])

    ##### Calculate the statistics from the sim_statistics part of the simulation:

    if per_element:
        counts_per_spg_per_element_per_wyckoff = {}
    else:
        counts_per_spg_per_wyckoff = {}

    counter_per_spg_per_element = {}

    NO_wyckoffs_per_spg = {}
    NO_unique_elements_per_spg = {}

    if per_element:
        NO_repetitions_per_spg_per_element = {}
    else:
        NO_repetitions_per_spg = {}

    denseness_factors_per_spg = {}

    all_data_per_spg = {}

    # Pre-process the symmetry groups:

    for spg_number in spgs:

        if per_element:
            counts_per_spg_per_element_per_wyckoff[spg_number] = {}
        else:
            counts_per_spg_per_wyckoff[spg_number] = {}

        NO_wyckoffs_per_spg[spg_number] = []
        NO_unique_elements_per_spg[spg_number] = []

        if per_element:
            NO_repetitions_per_spg_per_element[spg_number] = {}
        else:
            NO_repetitions_per_spg[spg_number] = []

        counter_per_spg_per_element[spg_number] = {}

        denseness_factors_per_spg[spg_number] = []

        all_data_per_spg[spg_number] = []

    # Analyze the statistics:

    start = time.time()
    nan_counter_statistics = 0

    statistics_match_metas = []
    statistics_match_labels = []

    for i in range(len(statistics_crystals)):
        crystal = statistics_crystals[i]

        if (i % 100) == 0:
            print(f"{i / len(statistics_crystals) * 100} % processed.")

        try:

            # Get conventional structure
            analyzer = SpacegroupAnalyzer(crystal)
            conv = analyzer.get_conventional_standard_structure()
            crystal = conv
            statistics_crystals[i] = conv

            _, NO_wyckoffs, _, _, _, _, _, _ = sim.get_wyckoff_info(
                statistics_metas[i][0]
            )

            if np.any(np.isnan(statistics_variations[i][0])):
                nan_counter_statistics += 1
                continue

            if (
                NO_wyckoffs > max_NO_wyckoffs or crystal.volume > max_volume
            ):  # only consider matching structures for statistics
                continue

            statistics_match_metas.append(statistics_metas[i])
            statistics_match_labels.append(statistics_labels[i])

            struc = pyxtal()
            struc.from_seed(crystal)
            spg_number = (
                struc.group.number
            )  # use the group as calculated by pyxtal for statistics

            NO_wyckoffs, elements = get_wyckoff_info(struc)

            NO_wyckoffs_per_spg[spg_number].append(len(struc.atom_sites))

            try:
                denseness_factor = get_denseness_factor(crystal)
                denseness_factors_per_spg[spg_number].append(denseness_factor)
            except Exception as ex:
                print("Exception while calculating denseness factor.")
                print(ex)

            elements_unique = np.unique(elements)

            for el in elements_unique:
                if "+" in el or "-" in el or "." in el or "," in el:
                    print("Non-clean element string detected.")

            NO_unique_elements_per_spg[spg_number].append(len(elements_unique))

            for el in elements_unique:
                reps = np.sum(np.array(elements) == el)

                if per_element:
                    if el in NO_repetitions_per_spg_per_element[spg_number].keys():
                        NO_repetitions_per_spg_per_element[spg_number][el].append(reps)
                    else:
                        NO_repetitions_per_spg_per_element[spg_number][el] = [reps]
                else:
                    NO_repetitions_per_spg[spg_number].append(reps)

        except Exception as ex:

            print(f"Error processing structure:")
            print(ex)

            continue

        specie_strs = []

        # We store the full wyckoff position occupations in a convenient format in all_data_entry
        # to later be able to reuse this without rerunning the whole pre-processing again
        all_data_entry = {}
        all_data_entry["occupations"] = []
        all_data_entry["lattice_parameters"] = struc.lattice.get_para()

        for site in struc.atom_sites:

            specie_str = str(site.specie)
            specie_strs.append(specie_str)

            name = str(site.wp.multiplicity) + site.wp.letter  # wyckoff name

            all_data_entry["occupations"].append((specie_str, name, site.position))

            # Store the wyckoff occupations => to later calculate the wyckoff occupation probabilities
            if per_element:

                if (
                    specie_str
                    in counts_per_spg_per_element_per_wyckoff[spg_number].keys()
                ):
                    if (
                        name
                        in counts_per_spg_per_element_per_wyckoff[spg_number][
                            specie_str
                        ].keys()
                    ):
                        counts_per_spg_per_element_per_wyckoff[spg_number][specie_str][
                            name
                        ] += 1
                    else:
                        counts_per_spg_per_element_per_wyckoff[spg_number][specie_str][
                            name
                        ] = 1
                else:
                    counts_per_spg_per_element_per_wyckoff[spg_number][specie_str] = {}
                    counts_per_spg_per_element_per_wyckoff[spg_number][specie_str][
                        name
                    ] = 1

            else:

                if name in counts_per_spg_per_wyckoff[spg_number].keys():
                    counts_per_spg_per_wyckoff[spg_number][name] += 1
                else:
                    counts_per_spg_per_wyckoff[spg_number][name] = 1

        # Store the frequency of which elements are present
        for specie_str in np.unique(specie_strs):
            if specie_str in counter_per_spg_per_element[spg_number].keys():
                counter_per_spg_per_element[spg_number][specie_str] += 1
            else:
                counter_per_spg_per_element[spg_number][specie_str] = 1

        all_data_per_spg[spg_number].append(all_data_entry)

    NO_unique_elements_prob_per_spg = {}
    if per_element:
        NO_repetitions_prob_per_spg_per_element = {}
    else:
        NO_repetitions_prob_per_spg = {}

    for spg in NO_wyckoffs_per_spg.keys():
        if len(NO_wyckoffs_per_spg[spg]) > 0:

            bincounted_NO_unique_elements = np.bincount(NO_unique_elements_per_spg[spg])
            NO_unique_elements_prob_per_spg[spg] = bincounted_NO_unique_elements[
                1:
            ] / np.sum(bincounted_NO_unique_elements[1:])

            if per_element:

                NO_repetitions_prob_per_spg_per_element[spg] = {}

                for el in NO_repetitions_per_spg_per_element[spg].keys():

                    bincounted_NO_repetitions = np.bincount(
                        NO_repetitions_per_spg_per_element[spg][el]
                    )

                    NO_repetitions_prob_per_spg_per_element[spg][
                        el
                    ] = bincounted_NO_repetitions[1:] / np.sum(
                        bincounted_NO_repetitions[1:]
                    )

            else:

                bincounted_NO_repetitions = np.bincount(NO_repetitions_per_spg[spg])

                NO_repetitions_prob_per_spg[spg] = bincounted_NO_repetitions[
                    1:
                ] / np.sum(bincounted_NO_repetitions[1:])

        else:

            NO_unique_elements_prob_per_spg[spg] = []

            if per_element:
                NO_repetitions_prob_per_spg_per_element[spg] = {}
            else:
                NO_repetitions_prob_per_spg[spg] = []

    print(f"Took {time.time() - start} s to calculate the statistics.", flush=True)

    #####

    # Find the meta ids of the match and match_pure test datasets
    # match: volume and number of atoms in asymmetric unit not too high
    # match_pure: same as match, but only crystals without partial occupancies

    test_match_metas = []
    test_match_pure_metas = []

    print("Processing test dataset...", flush=True)
    start = time.time()

    corrected_labels = []  # Labels as given by SpacegroupAnalyzer
    count_mismatches = 0  # Count mismatches between the space group determined by SpacegroupAnalyzer and given by ICSD

    nan_counter_test = 0

    for i in reversed(range(len(test_crystals))):
        crystal = test_crystals[i]

        if (i % 100) == 0:
            print(f"{(len(test_crystals) - i) / len(test_crystals) * 100} % processed.")

        try:

            is_pure, NO_wyckoffs, _, _, _, _, _, _ = sim.get_wyckoff_info(
                test_metas[i][0]
            )

            # Get conventional structure
            analyzer = SpacegroupAnalyzer(crystal)
            conv = analyzer.get_conventional_standard_structure()
            test_crystals[i] = conv
            crystal = conv

            if not (
                (max_volume is not None and test_crystals[i].volume > max_volume)
                or (max_NO_wyckoffs is not None and NO_wyckoffs > max_NO_wyckoffs)
            ) and not np.any(np.isnan(test_variations[i][0])):

                test_match_metas.append(test_metas[i])
                if is_pure:
                    test_match_pure_metas.append(test_metas[i])

            if np.any(
                np.isnan(test_variations[i][0])
            ):  # If the simulation of the pattern was unsuccessful
                del test_crystals[i]
                del test_metas[i]
                del test_variations[i]
                del test_labels[i]
                nan_counter_test += 1
                continue

            spg_number_icsd = test_labels[i][0]

            analyzer = SpacegroupAnalyzer(
                crystal,
                # symprec=1e-8,
                symprec=1e-4,  # for now (as in Pyxtal), use higher value than for perfect generated crystals
                angle_tolerance=5.0,
            )

            spg_analyzer = analyzer.get_space_group_number()

            if spg_analyzer != spg_number_icsd:
                count_mismatches += 1

            corrected_labels.append(spg_analyzer)

        except Exception as ex:

            print(f"Error processing structure, skipping in test set:")
            print(ex)

            corrected_labels.append(None)

    print(
        f"{count_mismatches/len(test_crystals)*100}% mismatches between spgs provided by ICSD and analyzed by SpacegroupAnalyzer in test set."
    )

    print(f"Took {time.time() - start} s to process the test dataset.")

    print(f"Size of statistics dataset: {len(statistics_crystals)}")
    print(f"Size of statistics_match dataset: {len(statistics_match_metas)}")
    print(f"Size of test dataset: {len(test_crystals)}")
    print(f"Size of test_match dataset: {len(test_match_metas)}")
    print(
        f"Size of test_match_pure_metas dataset: {len(test_match_pure_metas)}",
        flush=True,
    )
    print(f"Nan-counter test dataset: {nan_counter_test}")
    print(f"Nan-counter statistics dataset: {nan_counter_statistics}")

    os.system("mkdir -p " + output_dir)

    with open(f"{output_dir}/meta", "wb") as file:
        pickle.dump(
            (
                counter_per_spg_per_element,
                counts_per_spg_per_element_per_wyckoff
                if per_element
                else counts_per_spg_per_wyckoff,
                NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element
                if per_element
                else NO_repetitions_prob_per_spg,
                denseness_factors_per_spg,
                per_element,
                statistics_metas,
                statistics_labels,
                statistics_match_metas,
                statistics_match_labels,
                test_metas,
                test_labels,
                list(reversed(corrected_labels)),
                test_match_metas,
                test_match_pure_metas,
            ),
            file,
        )

    # Split array in parts to lower memory requirements:

    for i in range(0, int(len(test_crystals) / 1000) + 1):
        with open(f"{output_dir}/test_crystals_{i}", "wb") as file:
            pickle.dump(test_crystals[i * 1000 : (i + 1) * 1000], file)

    for i in range(0, int(len(statistics_crystals) / 1000) + 1):
        with open(f"{output_dir}/statistics_crystals_{i}", "wb") as file:
            pickle.dump(statistics_crystals[i * 1000 : (i + 1) * 1000], file)

    with open(f"{output_dir}/all_data_per_spg", "wb") as file:
        pickle.dump(all_data_per_spg, file)

    return True


def load_dataset_info(
    minimum_NO_statistics_crystals_per_spg=50,
    check_for_sum_formula_overlap=False,
    load_public_statistics_only=False,
    save_public_statistics=False,
):
    """Load prepared dataset from the prepared_dataset directory. This includes
    statistics data and information about the statistics / test split.

    Args:
        minimum_NO_statistics_crystals_per_spg (int, optional): Minimum number of crystals in the statistics dataset per space group.
            Space groups where this is not fulfilled are skipped. Defaults to 50.
        check_for_sum_formula_overlap (bool, optional): Analyze the overlap of sum formulas between the statistics and test dataset. Defaults to False.
        load_public_statistics_only (bool, optional): Whether or not to load only the publically available statistics data (non-licensed). Defaults to False.
        save_public_statistics (bool, optional): Whether or not to save the publically available statistics data separately. Defaults to False.

    Returns:
        tuple: (public_statistics, dataset with statistics and test split)
    """

    repository_dir = os.path.dirname(os.path.dirname(__file__))

    if load_public_statistics_only:

        with open(
            os.path.join(repository_dir, "public_statistics"),
            "rb",
        ) as file:

            public_statistics = pickle.load(file)
            return public_statistics

    with open(
        os.path.join(repository_dir, "prepared_dataset/meta"),
        "rb",
    ) as file:

        data = pickle.load(file)

        per_element = data[5]

        counter_per_spg_per_element = data[0]
        if per_element:
            counts_per_spg_per_element_per_wyckoff = data[1]
        else:
            counts_per_spg_per_wyckoff = data[1]
        NO_unique_elements_prob_per_spg = data[2]

        if per_element:
            NO_repetitions_prob_per_spg_per_element = data[3]
        else:
            NO_repetitions_prob_per_spg = data[3]
        denseness_factors_per_spg = data[4]

        statistics_metas = data[6]
        statistics_labels = data[7]
        statistics_match_metas = data[8]
        statistics_match_labels = data[9]
        test_metas = data[10]
        test_labels = data[11]
        corrected_labels = data[12]
        test_match_metas = data[13]
        test_match_pure_metas = data[14]

    if check_for_sum_formula_overlap:

        # Check for overlap in sum formulas between
        # test_metas and statistics_metas

        jobid = os.getenv("SLURM_JOB_ID")
        if jobid is not None and jobid != "":
            path_to_icsd_directory = path_to_icsd_directory_cluster
        else:
            path_to_icsd_directory = path_to_icsd_directory_local

        sim = ICSDSimulator(
            os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
            os.path.join(path_to_icsd_directory, "cif/"),
        )
        sim.output_dir = path_to_patterns

        statistics_sum_formulas = [
            sim.icsd_sumformulas[sim.icsd_ids.index(meta[0])]
            for meta in statistics_metas
        ]
        statistics_sum_formulas_set = set(statistics_sum_formulas)

        overlap_counter = 0
        overlap_matching_spgs_counter = 0

        for meta in test_metas:

            test_index = sim.icsd_ids.index(meta[0])
            test_sum_formula = sim.icsd_sumformulas[test_index]

            if test_sum_formula in statistics_sum_formulas_set:

                overlap_counter += 1

                # test_prototype = sim.icsd_structure_types[test_index]
                test_spg = sim.get_space_group_number(meta[0])

                # statistics_index = sim.icsd_ids.index(
                #    statistics_metas[statistics_sum_formulas.index(test_sum_formula)][0]
                # )
                # statistics_prototype = sim.icsd_structure_types[statistics_index]
                # statistics_spg = sim.get_space_group_number(
                #    sim.icsd_ids[statistics_index]
                # )
                # print(
                #    f"Overlap: sum formular {test_sum_formula}, prototypes {test_prototype} (test) and {statistics_prototype} (statistics), spgs {test_spg} (test) and {statistics_spg} (statistics)"
                # )

                matching_indices = [
                    item[0]
                    for item in enumerate(statistics_sum_formulas)
                    if item[1] == test_sum_formula
                ]

                for matching_index in matching_indices:
                    if (
                        sim.get_space_group_number(statistics_metas[matching_index][0])
                        == test_spg
                    ):
                        overlap_matching_spgs_counter += 1
                        break

        print(
            f"{overlap_counter} of {len(test_metas)} sample sum formulas in the test set are also in the train set."
        )
        print(
            f"{overlap_matching_spgs_counter} of {len(test_metas)} sample sum formulas WITH THE SAME SPG in the test set are also in the train set."
        )
        exit()

    with open(
        os.path.join(repository_dir, "prepared_dataset/all_data_per_spg"),
        "rb",
    ) as file:
        all_data_per_spg = pickle.load(file)

    ##### Load crystals

    test_crystals_files = sorted(
        glob(os.path.join(repository_dir, "prepared_dataset/test_crystals_*")),
        key=lambda x: int(os.path.basename(x).replace("test_crystals_", "")),
    )
    statistics_crystals_files = sorted(
        glob(os.path.join(repository_dir, "prepared_dataset/statistics_crystals_*")),
        key=lambda x: int(os.path.basename(x).replace("statistics_crystals_", "")),
    )

    test_crystals = []
    for file in test_crystals_files:
        with open(file, "rb") as file:
            test_crystals.extend(pickle.load(file))

    statistics_crystals = []
    for file in statistics_crystals_files:
        with open(file, "rb") as file:
            statistics_crystals.extend(pickle.load(file))

    ##### Determine well-represented spgs and print some information about the dataset

    print("Info about statistics (prepared) dataset:")
    total = 0
    total_below_X = 0
    represented_spgs = []
    for spg in denseness_factors_per_spg.keys():
        total += len(
            [item for item in denseness_factors_per_spg[spg] if item is not None]
        )
        if (
            len([item for item in denseness_factors_per_spg[spg] if item is not None])
            < minimum_NO_statistics_crystals_per_spg
        ):
            total_below_X += 1
        else:
            represented_spgs.append(spg)
    print(f"{total} total entries.")
    print(
        f"{total_below_X} spgs below {minimum_NO_statistics_crystals_per_spg} entries."
    )  # => these spgs will be excluded when training etc.

    ##### Determine the KDE for the denseness factor (either non-conditional (1D) or conditioned on the sum of atomic volumes)

    denseness_factors_density_per_spg = {}
    denseness_factors_conditional_sampler_seeds_per_spg = {}

    for spg in denseness_factors_per_spg.keys():

        denseness_factors = [
            item[0] for item in denseness_factors_per_spg[spg] if item is not None
        ]
        sums_cov_volumes = [
            item[1] for item in denseness_factors_per_spg[spg] if item is not None
        ]

        ########## 1D densities: (non-conditional)

        if len(denseness_factors) >= minimum_NO_statistics_crystals_per_spg:
            denseness_factors_density = kde.gaussian_kde(denseness_factors)
        else:
            denseness_factors_density = None

        denseness_factors_density_per_spg[spg] = denseness_factors_density

        ########## 2D densities (p(factor | volume)) (conditioned on the sum of atomic volumes):

        if len(denseness_factors) < minimum_NO_statistics_crystals_per_spg:
            denseness_factors_conditional_sampler_seeds_per_spg[spg] = None
            continue

        entries = [
            entry for entry in denseness_factors_per_spg[spg] if entry is not None
        ]
        entries = np.array(entries)

        conditional_density = sm.nonparametric.KDEMultivariateConditional(
            endog=[denseness_factors],
            exog=[sums_cov_volumes],
            dep_type="c",
            indep_type="c",
            bw=[
                0.0530715103,
                104.043070,
            ],  # bw pre-computed for performance reasons using normal reference method
        )

        # Store information that will later be needed to sample from this conditional
        # distribution.
        sampler_seed = (
            conditional_density,
            min(denseness_factors),
            max(denseness_factors),
            max(sums_cov_volumes),
        )
        denseness_factors_conditional_sampler_seeds_per_spg[spg] = sampler_seed

    ##### Convert counts to probabilities

    for spg in counter_per_spg_per_element.keys():
        for element in counter_per_spg_per_element[spg].keys():
            if element not in all_elements:
                raise Exception(f"Element {element} not supported.")

        # convert to relative entries
        total = 0
        for key in counter_per_spg_per_element[spg].keys():
            total += counter_per_spg_per_element[spg][key]
        for key in counter_per_spg_per_element[spg].keys():
            counter_per_spg_per_element[spg][key] /= total

    probability_per_spg_per_element = counter_per_spg_per_element

    if per_element:
        for spg in counts_per_spg_per_element_per_wyckoff.keys():
            for el in counts_per_spg_per_element_per_wyckoff[spg].keys():
                total = 0
                for wyckoff_site in counts_per_spg_per_element_per_wyckoff[spg][
                    el
                ].keys():
                    total += counts_per_spg_per_element_per_wyckoff[spg][el][
                        wyckoff_site
                    ]

                for wyckoff_site in counts_per_spg_per_element_per_wyckoff[spg][
                    el
                ].keys():
                    if total > 0:
                        counts_per_spg_per_element_per_wyckoff[spg][el][
                            wyckoff_site
                        ] /= total
                    else:
                        counts_per_spg_per_element_per_wyckoff[spg][el][
                            wyckoff_site
                        ] = 0
            probability_per_spg_per_element_per_wyckoff = (
                counts_per_spg_per_element_per_wyckoff
            )
    else:
        for spg in counts_per_spg_per_wyckoff.keys():
            total = 0
            for wyckoff_site in counts_per_spg_per_wyckoff[spg].keys():
                total += counts_per_spg_per_wyckoff[spg][wyckoff_site]

            for wyckoff_site in counts_per_spg_per_wyckoff[spg].keys():
                if total > 0:
                    counts_per_spg_per_wyckoff[spg][wyckoff_site] /= total
                else:
                    counts_per_spg_per_wyckoff[spg][wyckoff_site] = 0
        probability_per_spg_per_wyckoff = counts_per_spg_per_wyckoff

    ##### Calculate lattice parameter KDEs

    scaled_paras_per_lattice_type = {}

    for spg in all_data_per_spg.keys():

        _, lattice_type = get_pbc_and_lattice(spg, 3)

        for entry in all_data_per_spg[spg]:

            a, b, c, alpha, beta, gamma = entry["lattice_parameters"]

            volume = (
                a
                * b
                * c
                * np.sqrt(
                    1
                    + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                    - np.cos(alpha) ** 2
                    - np.cos(beta) ** 2
                    - np.cos(gamma) ** 2
                )
            )

            cbrt_volume = np.cbrt(volume)

            a /= cbrt_volume
            b /= cbrt_volume
            c /= cbrt_volume

            if lattice_type in ["cubic", "Cubic"]:

                paras = [a]

                assert (
                    (a == b)
                    and (b == c)
                    and (alpha == np.pi / 2)
                    and (alpha == beta)
                    and (alpha == gamma)
                )

                continue  # For cubic unit cells, there is no free parameter after fixing the volume

            elif lattice_type in ["hexagonal", "trigonal", "Hexagonal", "Trigonal"]:
                paras = [a, c]

                assert (
                    (a == b)
                    and (alpha == np.pi / 2)
                    and (alpha == beta)
                    and (gamma == np.pi * 2 / 3)
                )

            elif lattice_type in ["tetragonal", "Tetragonal"]:
                paras = [a, c]

                assert (
                    (a == b)
                    and (alpha == np.pi / 2)
                    and (alpha == beta)
                    and (alpha == gamma)
                )

            elif lattice_type in ["orthorhombic", "Orthorhombic"]:

                paras = [a, b, c]

                assert (alpha == np.pi / 2) and (alpha == beta) and (alpha == gamma)

            elif lattice_type in ["monoclinic", "Monoclinic"]:

                paras = [a, b, c, beta]

                assert (alpha == np.pi / 2) and (alpha == gamma)

            elif lattice_type == "triclinic":
                paras = [a, b, c, alpha, beta, gamma]

            else:

                raise Exception(f"Invalid lattice type {lattice_type}")

            if lattice_type in scaled_paras_per_lattice_type.keys():
                scaled_paras_per_lattice_type[lattice_type].append(paras)
            else:
                scaled_paras_per_lattice_type[lattice_type] = [paras]

    lattice_paras_density_per_lattice_type = {}
    for lattice_type in scaled_paras_per_lattice_type.keys():

        input_array = np.array(scaled_paras_per_lattice_type[lattice_type]).T
        density = kde.gaussian_kde(input_array)

        # print(density.covariance)
        # print(density.covariance.shape)
        # print(density.covariance_factor)
        # print(density.resample(1).T)

        lattice_paras_density_per_lattice_type[lattice_type] = density

    ##### Calculate the probability distribution of how spgs are represented in ICSD
    # This uses only the well-represented spgs in `represented_spgs`

    probability_per_spg = {}
    for i, label in enumerate(statistics_match_labels):
        if label[0] in represented_spgs:
            if label[0] in probability_per_spg.keys():
                probability_per_spg[label[0]] += 1
            else:
                probability_per_spg[label[0]] = 1
    total = np.sum(list(probability_per_spg.values()))
    for key in probability_per_spg.keys():
        probability_per_spg[key] /= total

    public_statistics = (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff
        if per_element
        else probability_per_spg_per_wyckoff,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element
        if per_element
        else NO_repetitions_prob_per_spg,
        denseness_factors_density_per_spg,
        denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type,
        per_element,
        represented_spgs,
        probability_per_spg,
    )

    split_dataset = (
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
    )

    if save_public_statistics:
        with open(
            os.path.join(repository_dir, "public_statistics"),
            "wb",
        ) as file:
            pickle.dump(public_statistics, file)

    return (public_statistics, split_dataset)


def show_dataset_statistics():
    """Loads the prepared dataset and prints the size of the statistics and test dataset.
    Additionally, for each spg the number of samples in the statistics and test dataset are printed.
    """

    prepared_dataset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "prepared_dataset"
    )

    with open(os.path.join(prepared_dataset_dir, "meta"), "rb") as file:

        data = pickle.load(file)

        per_element = data[5]

        counter_per_spg_per_element = data[0]
        if per_element:
            counts_per_spg_per_element_per_wyckoff = data[1]
        else:
            counts_per_spg_per_wyckoff = data[1]

        NO_unique_elements_prob_per_spg = data[2]

        if per_element:
            NO_repetitions_prob_per_spg_per_element = data[3]
        else:
            NO_repetitions_prob_per_spg = data[3]
        denseness_factors_per_spg = data[4]

        statistics_metas = data[6]
        statistics_labels = data[7]
        statistics_match_metas = data[8]
        statistics_match_labels = data[9]
        test_metas = data[10]
        test_labels = data[11]
        corrected_labels = data[12]
        test_match_metas = data[13]
        test_match_pure_metas = data[14]

    test_match_labels = []
    test_metas_flat = [meta[0] for meta in test_metas]
    for meta in test_match_metas:
        test_match_labels.append(test_labels[test_metas_flat.index(meta[0])])

    samples_per_spg_statistics = {}
    samples_per_spg_test = {}

    for label in statistics_match_labels:
        if label[0] in samples_per_spg_statistics.keys():
            samples_per_spg_statistics[label[0]] += 1
        else:
            samples_per_spg_statistics[label[0]] = 1

    print(f"Size of statistics match dataset: {len(statistics_match_labels)}")
    print(f"Size of test match dataset: {len(test_match_labels)}")

    for label in test_match_labels:
        if label[0] in samples_per_spg_test.keys():
            samples_per_spg_test[label[0]] += 1
        else:
            samples_per_spg_test[label[0]] = 1

    all_spgs = np.unique(
        sorted(
            list(samples_per_spg_statistics.keys()) + list(samples_per_spg_test.keys())
        )
    )

    for spg in all_spgs:
        N_test = samples_per_spg_test[spg] if spg in samples_per_spg_test.keys() else 0
        N_statistics = (
            samples_per_spg_statistics[spg]
            if spg in samples_per_spg_statistics.keys()
            else 0
        )
        # if N_test > N_statistics:
        print(f"spg {spg}: statistics: {N_statistics} test: {N_test}")


if __name__ == "__main__":
    prepare_dataset(per_element=False)
