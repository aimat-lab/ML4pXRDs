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
        in_base = "classifier_spgs/runs_from_cluster/initial_tests/06-03-2022_23-58-34/"
        # tag = "magpie_10-03-2022_14-34-51"
        tag = "volumes_densenesses_4-spg"

        spgs_to_analyze = [14, 104, 176, 129]
        # spgs_to_analyze = None  # analyse all space groups; alternative: list of spgs

    compute_magpie_features = False

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
    random_labels = [spgs[index] for index in random_labels]

    with open(in_base + "rightly_falsely_icsd.pickle", "rb") as file:
        rightly_indices_icsd, falsely_indices_icsd = pickle.load(file)

    with open(in_base + "rightly_falsely_random.pickle", "rb") as file:
        rightly_indices_random, falsely_indices_random = pickle.load(file)

    # limit the range:
    if False:
        random_crystals = random_crystals[0:300]
        random_labels = random_labels[0:300]
        random_variations = random_variations[0:300]
        icsd_crystals = icsd_crystals[0:300]
        icsd_labels = icsd_labels[0:300]
        icsd_variations = icsd_variations[0:300]
        icsd_metas = icsd_metas[0:300]

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
    icsd_falsely_corn_sizes = []
    icsd_falsely_NO_wyckoffs = []
    icsd_falsely_NO_elements = []
    icsd_falsely_occupancies = []
    icsd_falsely_occupancies_weights = []
    icsd_falsely_element_repetitions = []
    icsd_falsely_NO_atoms = []
    icsd_falsely_wyckoff_repetitions = []
    icsd_falsely_magpie_features = []

    icsd_rightly_crystals = []
    icsd_rightly_volumes = []
    icsd_rightly_angles = []
    icsd_rightly_denseness_factors = []
    icsd_rightly_lattice_paras = []
    icsd_rightly_corn_sizes = []
    icsd_rightly_NO_wyckoffs = []
    icsd_rightly_NO_elements = []
    icsd_rightly_occupancies = []
    icsd_rightly_occupancies_weights = []
    icsd_rightly_element_repetitions = []
    icsd_rightly_NO_atoms = []
    icsd_rightly_wyckoff_repetitions = []
    icsd_rightly_magpie_features = []

    random_rightly_crystals = []
    random_rightly_volumes = []
    random_rightly_angles = []
    random_rightly_denseness_factors = []
    random_rightly_lattice_paras = []
    random_rightly_corn_sizes = []
    random_rightly_NO_wyckoffs = []
    random_rightly_NO_elements = []
    random_rightly_element_repetitions = []
    random_rightly_NO_atoms = []
    random_rightly_wyckoff_repetitions = []
    random_rightly_magpie_features = []

    random_falsely_crystals = []
    random_falsely_volumes = []
    random_falsely_angles = []
    random_falsely_denseness_factors = []
    random_falsely_lattice_paras = []
    random_falsely_corn_sizes = []
    random_falsely_NO_wyckoffs = []
    random_falsely_NO_elements = []
    random_falsely_element_repetitions = []
    random_falsely_NO_atoms = []
    random_falsely_wyckoff_repetitions = []
    random_falsely_magpie_features = []

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

    falsely_icsd_volumes_densenesses = []

    for i in falsely_indices_icsd:

        index = int(i / 5)

        if spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze:

            structure = icsd_crystals[index]

            icsd_falsely_crystals.append(structure)

            volume = structure.volume

            denseness_factor = get_denseness_factor(structure)

            icsd_falsely_NO_atoms.append(len(structure.frac_coords))

            icsd_falsely_volumes.append(volume)
            icsd_falsely_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            icsd_falsely_corn_sizes.extend(icsd_variations[index])
            icsd_falsely_NO_elements.append(icsd_NO_elements[index])
            icsd_falsely_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            icsd_falsely_occupancies.extend(icsd_occupancies[index])
            icsd_falsely_occupancies_weights.extend(icsd_occupancies_weights[index])
            icsd_falsely_element_repetitions.extend(icsd_element_repetitions[index])

            icsd_falsely_lattice_paras.append(structure.lattice.a)
            icsd_falsely_lattice_paras.append(structure.lattice.b)
            icsd_falsely_lattice_paras.append(structure.lattice.c)

            if denseness_factor is not None:
                icsd_falsely_denseness_factors.append(denseness_factor)

                falsely_icsd_volumes_densenesses.append((volume, denseness_factor))

            icsd_falsely_wyckoff_repetitions.extend(icsd_wyckoff_repetitions[index])

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

    rightly_icsd_volumes_densenesses = []

    for i in rightly_indices_icsd:

        index = int(i / 5)

        if spgs_to_analyze is None or icsd_labels[index][0] in spgs_to_analyze:

            structure = icsd_crystals[index]

            icsd_rightly_crystals.append(structure)

            volume = structure.volume

            denseness_factor = get_denseness_factor(structure)

            icsd_rightly_NO_atoms.append(len(structure.frac_coords))

            icsd_rightly_volumes.append(volume)
            icsd_rightly_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )
            icsd_rightly_corn_sizes.extend(icsd_variations[index])
            icsd_rightly_NO_elements.append(icsd_NO_elements[index])
            icsd_rightly_NO_wyckoffs.append(icsd_NO_wyckoffs[index])
            icsd_rightly_occupancies.extend(icsd_occupancies[index])
            icsd_rightly_occupancies_weights.extend(icsd_occupancies_weights[index])
            icsd_rightly_element_repetitions.extend(icsd_element_repetitions[index])

            icsd_rightly_lattice_paras.append(structure.lattice.a)
            icsd_rightly_lattice_paras.append(structure.lattice.b)
            icsd_rightly_lattice_paras.append(structure.lattice.c)

            if denseness_factor is not None:
                icsd_rightly_denseness_factors.append(denseness_factor)

                rightly_icsd_volumes_densenesses.append((volume, denseness_factor))

            icsd_rightly_wyckoff_repetitions.extend(icsd_wyckoff_repetitions[index])

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

    falsely_icsd_volumes_densenesses = []

    for index in falsely_indices_random:

        structure = random_crystals[index]

        random_falsely_crystals.append(structure)

        if spgs_to_analyze is None or random_labels[index] in spgs_to_analyze:

            volume = structure.volume

            denseness_factor = get_denseness_factor(structure)

            random_falsely_NO_atoms.append(len(structure.frac_coords))

            random_falsely_volumes.append(volume)
            random_falsely_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )

            random_falsely_lattice_paras.append(structure.lattice.a)
            random_falsely_lattice_paras.append(structure.lattice.b)
            random_falsely_lattice_paras.append(structure.lattice.c)

            if denseness_factor is not None:
                random_falsely_denseness_factors.append(denseness_factor)

            random_falsely_corn_sizes.append(random_variations[index])
            random_falsely_NO_elements.append(random_NO_elements[index])
            random_falsely_NO_wyckoffs.append(random_NO_wyckoffs[index])
            random_falsely_element_repetitions.extend(random_element_repetitions[index])

            random_falsely_wyckoff_repetitions.extend(random_wyckoff_repetitions[index])

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

        structure = random_crystals[index]

        random_rightly_crystals.append(structure)

        if spgs_to_analyze is None or random_labels[index] in spgs_to_analyze:

            volume = structure.volume

            denseness_factor = get_denseness_factor(structure)

            random_rightly_NO_atoms.append(len(structure.frac_coords))

            random_rightly_volumes.append(volume)
            random_rightly_angles.extend(
                [
                    structure.lattice.alpha,
                    structure.lattice.beta,
                    structure.lattice.gamma,
                ]
            )

            random_rightly_lattice_paras.append(structure.lattice.a)
            random_rightly_lattice_paras.append(structure.lattice.b)
            random_rightly_lattice_paras.append(structure.lattice.c)

            if denseness_factor is not None:
                random_rightly_denseness_factors.append(denseness_factor)

            random_rightly_corn_sizes.append(random_variations[index])
            random_rightly_NO_elements.append(random_NO_elements[index])
            random_rightly_NO_wyckoffs.append(random_NO_wyckoffs[index])
            random_rightly_element_repetitions.extend(random_element_repetitions[index])

            random_rightly_wyckoff_repetitions.extend(random_wyckoff_repetitions[index])

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

    ################# volumes_denseness_factors ################

    rightly_volumes = [item[0] for item in rightly_icsd_volumes_densenesses]
    rightly_densenesses = [item[1] for item in rightly_icsd_volumes_densenesses]

    falsely_volumes = [item[0] for item in falsely_icsd_volumes_densenesses]
    falsely_densenesses = [item[1] for item in falsely_icsd_volumes_densenesses]

    plt.scatter(rightly_volumes, rightly_densenesses, color="r", s=5)
    plt.scatter(falsely_volumes, falsely_densenesses, color="g", s=5)
    plt.savefig(
        f"{out_base}volumes_densenesses.png",
        bbox_inches="tight",
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
                    hists[counter * 2 + 1],  # height
                    bottom=0,  # bottom
                    color="r",
                    label=labels[counter * 2 + 1],
                    width=bin_width,
                    align="edge",
                )

                # rightly
                h2 = ax1.bar(
                    bins[:-1],
                    hists[counter * 2],  # height
                    bottom=hists[counter * 2 + 1],  # bottom
                    color="g",
                    label=labels[counter * 2],
                    width=bin_width,
                    align="edge",
                )

            elif i == 1:

                # middle (falsely):
                ax1.step(
                    # bins[:-1],
                    bins[:],
                    # hists[counter * 2 + 1],
                    np.append(hists[counter * 2 + 1], hists[counter * 2 + 1][-1]),
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
                        (hists[counter * 2] + hists[counter * 2 + 1]),
                        (hists[counter * 2] + hists[counter * 2 + 1])[-1],
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
            [icsd_rightly_angles, icsd_falsely_angles],
            [random_rightly_angles, random_falsely_angles],
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
                icsd_rightly_denseness_factors,
                icsd_falsely_denseness_factors,
            ],
            [random_rightly_denseness_factors, random_falsely_denseness_factors],
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
            [icsd_rightly_corn_sizes, icsd_falsely_corn_sizes],
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
                icsd_rightly_lattice_paras,
                icsd_falsely_lattice_paras,
            ],
            [random_rightly_lattice_paras, random_falsely_lattice_paras],
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
            "occupancies_weighted",
            [icsd_rightly_occupancies, icsd_falsely_occupancies],
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
                icsd_rightly_occupancies_weights,
                icsd_falsely_occupancies_weights,
            ],
        )

    for flag in [True, False]:
        create_histogram(
            "occupancies",
            [icsd_rightly_occupancies, icsd_falsely_occupancies],
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
                icsd_rightly_element_repetitions,
                icsd_falsely_element_repetitions,
            ],
            [random_rightly_element_repetitions, random_falsely_element_repetitions],
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
                icsd_rightly_wyckoff_repetitions,
                icsd_falsely_wyckoff_repetitions,
            ],
            [random_rightly_wyckoff_repetitions, random_falsely_wyckoff_repetitions],
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
