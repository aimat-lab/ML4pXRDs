import os
from ml4pxrd_tools.simulation.simulator import Simulator
import math
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
import time
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from pyxtal import pyxtal
from training.analysis.denseness_factor import get_denseness_factor
import pickle
from glob import glob
from scipy.stats import kde
import statsmodels.api as sm
from pyxtal.symmetry import Group
from sklearn.neighbors import KernelDensity
from pyxtal.symmetry import get_pbc_and_lattice
from ml4pxrd_tools.generation.all_elements import all_elements


def get_wyckoff_info(pyxtal_crystal):
    # returns: Number of set wyckoffs, elements

    elements = []

    for site in pyxtal_crystal.atom_sites:
        specie_str = str(site.specie)
        elements.append(specie_str)

    return len(pyxtal_crystal.atom_sites), elements


def prepare_training(
    per_element=False, validation_max_volume=7000, validation_max_NO_wyckoffs=100
):

    if os.path.exists("prepared_training"):
        print("Please remove existing prepared_training folder first.")
        exit()

    spgs = range(1, 231)

    jobid = os.getenv("SLURM_JOB_ID")
    path_to_patterns = "./patterns/icsd_vecsei/"

    if jobid is not None and jobid != "":
        sim = Simulator(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
        sim.output_dir = path_to_patterns

    else:  # local
        sim = Simulator(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        sim.output_dir = path_to_patterns

    sim.load(load_patterns_angles_intensities=False)

    ########## Train (statistics) / test splitting:

    ### Four strategies: random, structure type full, main structure type, sum formula
    strategy = "structure type full"

    if strategy != "random":

        group_labels = []

        counter = 0

        for i, meta in enumerate(sim.sim_metas):

            print(f"{i/len(sim.sim_metas)*100}%")

            index = sim.icsd_ids.index(meta[0])

            if strategy == "structure type full":
                group_label = sim.icsd_structure_types[index]
            elif strategy == "main structure type":
                group_label = sim.icsd_structure_types[index].split("#")[
                    0
                ]  # only use the main part of the structure type
            elif strategy == "sum formula":
                group_label = sim.icsd_sumformulas[index]
            else:
                raise Exception("Grouping strategy not supported.")

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

        train_metas_splitted = [
            sim.sim_metas[i][0] for i in train_metas_splitted_indices
        ]
        test_metas_splitted = [sim.sim_metas[i][0] for i in test_metas_splitted_indices]

    else:

        (train_metas_splitted, test_metas_splitted) = train_test_split(  # statistics
            sim.sim_metas, test_size=0.3
        )

    # Check if they are distinct

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
    print(f"{overlap_counter} of {len(train_metas_splitted)} overlapped.", flush=True)

    # Check if train_metas_splitted and test_metas_splitted together yield sim.sim_metas

    all_metas = train_metas_splitted + test_metas_splitted
    if sorted(all_metas) != sorted([item[0] for item in sim.sim_metas]):
        raise Exception("Something went wrong while splitting train / test.")

    if overlap_counter > 0:
        raise Exception("Something went wrong while splitting train / test.")

    ##########

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
        else:
            raise Exception("Something went wrong while splitting train / test.")

    ##########

    # Calculate the statistics from the sim_statistics part of the simulation:

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

    # pre-process the symmetry groups:

    for spg_number in spgs:

        # group = Group(spg_number, dim=3)
        # names = [(str(x.multiplicity) + x.letter) for x in group]

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

    # Analyse the statistics:

    start = time.time()
    nan_counter_statistics = 0

    statistics_match_metas = []
    statistics_match_labels = []

    for i in range(len(statistics_crystals)):
        crystal = statistics_crystals[i]

        if (i % 100) == 0:
            print(f"{i / len(statistics_crystals) * 100} % processed.")

        try:

            # spg_number = sim_statistics.sim_labels[i][0]

            # for site in crystal.sites:
            #    if site.species_string == "D-" or site.species_string == "D+":
            #        print("Oh")
            #
            #        calc = XRDCalculator(1.5)
            #        calc.get_pattern(crystal)
            #
            # This error also manifests itself in the comparison script when trying to get the covalent
            # radius of D (which is unknown element).

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
                NO_wyckoffs > validation_max_NO_wyckoffs
                or crystal.volume > validation_max_volume
            ):  # only consider matching structures for statistics, too
                continue

            statistics_match_metas.append(statistics_metas[i])
            statistics_match_labels.append(statistics_labels[i])

            struc = pyxtal()
            struc.from_seed(crystal)
            spg_number = (
                struc.group.number
            )  # use the group as calculated by pyxtal for statistics; this should be fine.

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
                    print("Ohoh")

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

        all_data_entry = {}
        all_data_entry["occupations"] = []
        all_data_entry["lattice_parameters"] = struc.lattice.get_para()

        for site in struc.atom_sites:

            specie_str = str(site.specie)
            specie_strs.append(specie_str)

            name = str(site.wp.multiplicity) + site.wp.letter  # wyckoff name

            all_data_entry["occupations"].append((specie_str, name, site.position))

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

        for specie_str in np.unique(specie_strs):
            if specie_str in counter_per_spg_per_element[spg_number].keys():
                counter_per_spg_per_element[spg_number][specie_str] += 1
            else:
                counter_per_spg_per_element[spg_number][specie_str] = 1

        all_data_per_spg[spg_number].append(all_data_entry)

    NO_wyckoffs_prob_per_spg = {}

    NO_unique_elements_prob_per_spg = {}
    if per_element:
        NO_repetitions_prob_per_spg_per_element = {}
    else:
        NO_repetitions_prob_per_spg = {}

    for spg in NO_wyckoffs_per_spg.keys():
        if len(NO_wyckoffs_per_spg[spg]) > 0:

            bincounted_NO_wyckoffs = np.bincount(NO_wyckoffs_per_spg[spg])
            bincounted_NO_unique_elements = np.bincount(NO_unique_elements_per_spg[spg])

            NO_wyckoffs_prob_per_spg[spg] = bincounted_NO_wyckoffs[1:] / np.sum(
                bincounted_NO_wyckoffs[1:]
            )
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
            NO_wyckoffs_prob_per_spg[spg] = []
            NO_unique_elements_prob_per_spg[spg] = []

            if per_element:
                NO_repetitions_prob_per_spg_per_element[spg] = {}
            else:
                NO_repetitions_prob_per_spg[spg] = []

    print(f"Took {time.time() - start} s to calculate the statistics.", flush=True)

    ##########

    # Find the meta ids of the two test datasets:
    test_match_metas = []
    test_match_pure_metas = []

    print("Processing test dataset...", flush=True)
    start = time.time()

    corrected_labels = []
    count_mismatches = 0

    nan_counter_test = 0

    for i in reversed(range(len(test_crystals))):
        crystal = test_crystals[i]

        if (i % 100) == 0:
            print(f"{i / len(test_crystals) * 100} % processed.")

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
                (
                    validation_max_volume is not None
                    and test_crystals[i].volume > validation_max_volume
                )
                or (
                    validation_max_NO_wyckoffs is not None
                    and NO_wyckoffs > validation_max_NO_wyckoffs
                )
            ) and not np.any(np.isnan(test_variations[i][0])):

                test_match_metas.append(test_metas[i])
                if is_pure:
                    test_match_pure_metas.append(test_metas[i])

            if np.any(np.isnan(test_variations[i][0])):
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

    print(f"{count_mismatches/len(test_crystals)*100}% mismatches in test set.")

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

    os.system("mkdir -p prepared_training")

    with open("prepared_training/meta", "wb") as file:
        pickle.dump(
            (
                counter_per_spg_per_element,
                counts_per_spg_per_element_per_wyckoff
                if per_element
                else counts_per_spg_per_wyckoff,
                NO_wyckoffs_prob_per_spg,
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
        with open(f"prepared_training/test_crystals_{i}", "wb") as file:
            pickle.dump(test_crystals[i * 1000 : (i + 1) * 1000], file)

    for i in range(0, int(len(statistics_crystals) / 1000) + 1):
        with open(f"prepared_training/statistics_crystals_{i}", "wb") as file:
            pickle.dump(statistics_crystals[i * 1000 : (i + 1) * 1000], file)

    with open("prepared_training/all_data_per_spg", "wb") as file:
        pickle.dump(all_data_per_spg, file)


def load_dataset_info(X=50, check_for_sum_formula_overlap=False):

    with open(
        os.path.join(os.path.dirname(__file__), "prepared_training/meta"), "rb"
    ) as file:
        data = pickle.load(file)

        per_element = data[6]

        counter_per_spg_per_element = data[0]
        if per_element:
            counts_per_spg_per_element_per_wyckoff = data[1]
        else:
            counts_per_spg_per_wyckoff = data[1]
        NO_wyckoffs_prob_per_spg = data[2]
        NO_unique_elements_prob_per_spg = data[3]

        if per_element:
            NO_repetitions_prob_per_spg_per_element = data[4]
        else:
            NO_repetitions_prob_per_spg = data[4]
        denseness_factors_per_spg = data[5]

        statistics_metas = data[7]
        statistics_labels = data[8]
        statistics_match_metas = data[9]
        statistics_match_labels = data[10]
        test_metas = data[11]
        test_labels = data[12]
        corrected_labels = data[13]
        test_match_metas = data[14]
        test_match_pure_metas = data[15]

    if check_for_sum_formula_overlap:
        # Check for overlap in sum formulas between
        # test_metas and statistics_metas

        path_to_patterns = "./patterns/icsd_vecsei/"
        jobid = os.getenv("SLURM_JOB_ID")
        if jobid is not None and jobid != "":
            sim = Simulator(
                os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
                os.path.expanduser("~/Databases/ICSD/cif/"),
            )
            sim.output_dir = path_to_patterns
        else:  # local
            sim = Simulator(
                "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
                "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
            )
            sim.output_dir = path_to_patterns

        # statistics_metas = statistics_metas[0:1000]
        # test_metas = test_metas[0:1000]

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
        os.path.join(os.path.dirname(__file__), "prepared_training/all_data_per_spg"),
        "rb",
    ) as file:
        all_data_per_spg = pickle.load(file)

    # Split array in parts to lower memory requirements:
    test_crystals_files = sorted(
        glob(
            os.path.join(os.path.dirname(__file__), "prepared_training/test_crystals_*")
        ),
        key=lambda x: int(os.path.basename(x).replace("test_crystals_", "")),
    )
    statistics_crystals_files = sorted(
        glob(
            os.path.join(
                os.path.dirname(__file__), "prepared_training/statistics_crystals_*"
            )
        ),
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
            < X
        ):
            total_below_X += 1
        else:
            represented_spgs.append(spg)
    print(f"{total} total entries.")
    print(f"{total_below_X} spgs below {X} entries.")

    denseness_factors_density_per_spg = {}
    denseness_factors_conditional_sampler_seeds_per_spg = {}

    for spg in denseness_factors_per_spg.keys():

        denseness_factors = [
            item[0] for item in denseness_factors_per_spg[spg] if item is not None
        ]
        sums_cov_volumes = [
            item[1] for item in denseness_factors_per_spg[spg] if item is not None
        ]

        ########## 1D densities:

        if len(denseness_factors) >= X:
            denseness_factors_density = kde.gaussian_kde(denseness_factors)
        else:
            denseness_factors_density = None

        denseness_factors_density_per_spg[spg] = denseness_factors_density

        if (
            False
            and spg in list(range(1, 231, 3))
            and denseness_factors_density is not None
        ):
            grid = np.linspace(
                min(denseness_factors_per_spg[spg]),
                max(denseness_factors_per_spg[spg]),
                300,
            )
            plt.figure()
            plt.plot(grid, denseness_factors_density(grid))
            plt.hist(denseness_factors_per_spg[spg], density=True, bins=60)
            plt.savefig(f"denseness_factors_fit_{spg}.png")

        ########## 2D densities (p(factor | volume)):

        if len(denseness_factors) < X:
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
            bw=[0.0530715103, 104.043070],  # TODO: Maybe change bandwidth in the future
        )

        # print(conditional_density.bw)

        sampler_seed = (
            conditional_density,
            min(denseness_factors),
            max(denseness_factors),
            max(sums_cov_volumes),
        )
        denseness_factors_conditional_sampler_seeds_per_spg[spg] = sampler_seed

        if False:
            start = time.time()
            sample(2000)
            stop = time.time()
            print(f"Sampling once took {stop-start}s")

    if False:
        for spg in [2, 15]:

            denseness_factors = [
                item[0] for item in denseness_factors_per_spg[spg] if item is not None
            ]
            sums_cov_volumes = [
                item[1] for item in denseness_factors_per_spg[spg] if item is not None
            ]
            entries = [
                entry for entry in denseness_factors_per_spg[spg] if entry is not None
            ]
            entries = np.array(entries)

            seed = denseness_factors_conditional_sampler_seeds_per_spg[spg]

            start = time.time()
            samples = [
                [sample_denseness_factor(volume, seed), volume]
                for volume in sums_cov_volumes[0:1000]
            ]
            stop = time.time()
            print(f"Sampling took {stop-start}s")

            plt.scatter(
                [item[1] for item in entries[0:1000]],
                [item[0] for item in entries[0:1000]],
                label="Original",
            )
            plt.scatter(
                [item[1] for item in samples],
                [item[0] for item in samples],
                label="Resampled",
            )
            plt.legend()
            plt.show()

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

    kde_per_spg = {}

    for spg in all_data_per_spg.keys():

        if len(all_data_per_spg[spg]) < X:
            kde_per_spg[spg] = None
            continue

        group = Group(spg, dim=3)
        names = [(str(x.multiplicity) + x.letter) for x in group]

        data = np.zeros(shape=(len(all_data_per_spg[spg]), len(names)))

        for i, entry in enumerate(all_data_per_spg[spg]):
            for subentry in entry["occupations"]:
                index = names.index(subentry[1])
                data[i, index] += 1

        kd = KernelDensity(bandwidth=0.5, kernel="gaussian")

        kd.fit(data)

        kde_per_spg[spg] = kd

        # for sample in kd.sample(15):
        #    print([int(item) if item >= 0 else 0 for item in np.round(sample)])

    ########## Calculate lattice parameter kdes:

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

    return (  # We reproduce all the probabilities from the ICSD, but all are independently drawn.
        # The only correlation considered is having multiple elements of the same type (less spread in number of unique elements).
        # This is the main assumption of my work.
        # There are no correlations in coordinate space.
        # P(wyckoff, element) = P(wyckoff|element)P(element)
        # We just want to resemble the occupation of wyckoff sites realistically in the most straightforward way.
        # More than that is not needed for merely extracting symmetry information.
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff
        if per_element
        else probability_per_spg_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element
        if per_element
        else NO_repetitions_prob_per_spg,
        denseness_factors_density_per_spg,
        kde_per_spg,
        all_data_per_spg,
        denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type,
        per_element,
        represented_spgs,
        (
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
        ),
        probability_per_spg,
    )


def show_dataset_statistics():

    with open(
        os.path.join(os.path.dirname(__file__), "prepared_training/meta"), "rb"
    ) as file:
        data = pickle.load(file)

        per_element = data[6]

        counter_per_spg_per_element = data[0]
        if per_element:
            counts_per_spg_per_element_per_wyckoff = data[1]
        else:
            counts_per_spg_per_wyckoff = data[1]
        NO_wyckoffs_prob_per_spg = data[2]
        NO_unique_elements_prob_per_spg = data[3]

        if per_element:
            NO_repetitions_prob_per_spg_per_element = data[4]
        else:
            NO_repetitions_prob_per_spg = data[4]
        denseness_factors_per_spg = data[5]

        statistics_metas = data[7]
        statistics_labels = data[8]
        statistics_match_metas = data[9]
        statistics_match_labels = data[10]
        test_metas = data[11]
        test_labels = data[12]
        corrected_labels = data[13]
        test_match_metas = data[14]
        test_match_pure_metas = data[15]

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
    prepare_training(per_element=False)
