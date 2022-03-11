from pyxtal import pyxtal
import matplotlib.pyplot as plt
import numpy as np
import random
from pyxtal.symmetry import Group
import time
import pickle
from dataset_simulations.simulation import Simulation
from pymatgen.io.cif import CifParser
import os
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.diffraction.xrd import XRDCalculator  # for debugging

# import warnings
# with warnings.catch_warnings():
#    warnings.simplefilter("error")

# extracted from pyxtal element.py:
all_elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
][0:94]


def track_job(job, update_interval=5):
    while job._number_left > 0:
        print(
            "Tasks remaining: {0} (chunk size {1})".format(
                job._number_left * job._chunksize, job._chunksize
            ),
            flush=True,
        )
        time.sleep(update_interval)


def generate_structure(
    _,
    group_object,
    multiplicities,
    names,
    letters,
    dofs,
    max_NO_elements=10,  # This doesn't have any effect if NO_wyckoffs_probability is set
    seed=-1,
    do_distance_checks=True,
    fixed_volume=None,
    do_merge_checks=True,
    use_icsd_statistics=False,
    probability_per_spg_per_element=None,
    probability_per_spg_per_wyckoff=None,
    max_volume=None,
    return_original_pyxtal_object=False,
    NO_wyckoffs_prob_per_spg=None,
    do_symmetry_checks=True,
    set_NO_elements_to_max=False,
    force_wyckoff_indices=True,
    use_element_repetitions_instead_of_NO_wyckoffs=False,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg=None,
    verbose=False,
):

    if use_icsd_statistics and (
        probability_per_spg_per_element is None
        or probability_per_spg_per_wyckoff is None
    ):
        raise Exception("Statistics data needed if use_icsd_statistics = True.")

    if use_element_repetitions_instead_of_NO_wyckoffs and (
        NO_unique_elements_prob_per_spg is None or NO_repetitions_prob_per_spg is None
    ):
        raise Exception(
            "Statistics data needed if use_element_repetitions_instead_of_NO_wyckoffs = True."
        )

    if seed != -1:
        np.random.seed(seed)
        random.seed(seed)

    if not use_element_repetitions_instead_of_NO_wyckoffs:

        if set_NO_elements_to_max:
            NO_elements = max_NO_elements
        elif NO_wyckoffs_prob_per_spg is None:
            NO_elements = random.randint(1, max_NO_elements)
        else:

            while True:

                NO_wyckoffs_probability = NO_wyckoffs_prob_per_spg[group_object.number]

                if np.sum(NO_wyckoffs_probability[0:max_NO_elements]) < 0.01:
                    raise Exception(
                        "Requested spg number has very small probability <= max_NO_elements."
                    )

                if len(NO_wyckoffs_probability) == 0:
                    raise Exception(
                        "Requested spg number with no probability entries in NO_wyckoffs_prob_per_spg."
                    )  # This should not happen!

                NO_elements = np.random.choice(
                    range(1, len(NO_wyckoffs_probability) + 1),
                    size=1,
                    p=NO_wyckoffs_probability,
                )[0]

                if NO_elements <= max_NO_elements:
                    break

    else:

        # pick number of unique elements to sample:

        NO_unique_elements = np.random.choice(
            range(1, len(NO_unique_elements_prob_per_spg[group_object.number]) + 1),
            size=1,
            p=NO_unique_elements_prob_per_spg[group_object.number],
        )[0]

        if verbose:
            print(f"NO_unique_elements: {NO_unique_elements}")

    tries_counter = 0

    while True:

        # If trying 20 times to generate a crystal with the given NO_elements / element repetitions fails, then pick a new
        # NO_elements / element repetitions and return that. This should always return at some point.
        if tries_counter > 20:

            if not use_element_repetitions_instead_of_NO_wyckoffs:
                print(
                    f"Failed generating crystal of spg {group_object.number} with {NO_elements} set wyckoff positions 10 times. Choosing new NO_elements repetitions now."
                )
            else:
                print(
                    f"Failed generating crystal of spg {group_object.number} with {NO_unique_elements} unique elements 10 times. Choosing new element repetitions now."
                )

            return generate_structure(
                _,
                group_object,
                multiplicities,
                names,
                letters,
                dofs,
                max_NO_elements,
                seed,
                do_distance_checks,
                fixed_volume,
                do_merge_checks,
                use_icsd_statistics,
                probability_per_spg_per_element,
                probability_per_spg_per_wyckoff,
                max_volume,
                return_original_pyxtal_object,
                NO_wyckoffs_prob_per_spg,
                do_symmetry_checks,
                set_NO_elements_to_max,
                force_wyckoff_indices=force_wyckoff_indices,
                use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions_instead_of_NO_wyckoffs,
                NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg=NO_repetitions_prob_per_spg,
            )

        number_of_atoms_per_site = np.zeros(len(names))

        chosen_elements = []
        chosen_numbers = []
        chosen_wyckoff_positions = []
        chosen_wyckoff_letters = []
        chosen_wyckoff_indices = []

        set_wyckoffs_counter = 0

        unique_elements_counter = 0
        current_element = None
        current_picked_repetition = None
        current_repetition_counter = 0

        while True:  # for

            # break conditions:
            if not use_element_repetitions_instead_of_NO_wyckoffs:
                if set_wyckoffs_counter >= NO_elements:
                    break
            else:
                if unique_elements_counter >= NO_unique_elements:
                    break

            counter_collisions = 0
            while True:

                if counter_collisions > 50:
                    print("More than 50 collisions setting an atom")

                if not use_icsd_statistics:
                    chosen_index = random.randint(0, len(number_of_atoms_per_site) - 1)
                else:
                    probability_per_wyckoff = probability_per_spg_per_wyckoff[
                        group_object.number
                    ]

                    chosen_wyckoff = np.random.choice(
                        list(probability_per_wyckoff.keys()),
                        1,
                        p=list(probability_per_wyckoff.values()),
                    )[0]
                    chosen_index = names.index(chosen_wyckoff)

                """
                # always first choose the general Wyckoff site:
                chosen_index = (
                    random.randint(0, len(number_of_atoms) - 1) if i > 0 else 0
                )
                """

                """ See this from the documentation
                PyXtal starts with the largest available WP, which is the general position of the space group.
                If the number of atoms required is equal to or greater than the size of the general position,
                the algorithm proceeds. If fewer atoms are needed, the next largest WP (or set of WP’s) is
                chosen, in order of descending multiplicity. This is done to ensure that larger positions are
                preferred over smaller ones; this reflects the greater prevalence of larger multiplicities
                both statistically and in nature.
                """

                if (
                    dofs[chosen_index] == 0
                    and int(number_of_atoms_per_site[chosen_index]) == 1
                ):
                    counter_collisions += 1
                    continue  # try again

                number_of_atoms_per_site[chosen_index] += 1

                if not use_element_repetitions_instead_of_NO_wyckoffs:
                    if not use_icsd_statistics:
                        chosen_elements.append(random.choice(all_elements))
                    else:
                        chosen_element = np.random.choice(
                            list(
                                probability_per_spg_per_element[
                                    group_object.number
                                ].keys()
                            ),
                            1,
                            p=list(
                                probability_per_spg_per_element[
                                    group_object.number
                                ].values()
                            ),
                        )[0]

                        chosen_elements.append(chosen_element)
                else:

                    if (
                        current_repetition_counter == current_picked_repetition
                        or current_picked_repetition is None
                    ):

                        current_picked_repetition = np.random.choice(
                            range(
                                1,
                                len(NO_repetitions_prob_per_spg[group_object.number])
                                + 1,
                            ),
                            1,
                            p=list(NO_repetitions_prob_per_spg[group_object.number]),
                        )[0]

                        current_repetition_counter = 1

                        while True:

                            current_element = np.random.choice(
                                list(
                                    probability_per_spg_per_element[
                                        group_object.number
                                    ].keys()
                                ),
                                1,
                                p=list(
                                    probability_per_spg_per_element[
                                        group_object.number
                                    ].values()
                                ),
                            )[0]

                            if current_element not in chosen_elements:
                                break

                        unique_elements_counter += 1

                    else:

                        current_repetition_counter += 1

                    chosen_elements.append(current_element)

                chosen_numbers.append(multiplicities[chosen_index])
                chosen_wyckoff_positions.append([names[chosen_index]])
                chosen_wyckoff_letters.append([letters[chosen_index]])
                chosen_wyckoff_indices.append(chosen_index)

                break

            if not use_element_repetitions_instead_of_NO_wyckoffs:
                set_wyckoffs_counter += 1

        if use_element_repetitions_instead_of_NO_wyckoffs:
            if len(chosen_numbers) > max_NO_elements:
                print(
                    f"Too many total number of set wyckoff sites for spg {group_object.number}, regenerating..."
                )
                continue  # but do not increase tries_counter, this is totally fine and expected!

        if verbose:
            print(f"Number of chosen wyckoff sites: {len(chosen_numbers)}")

        my_crystal = pyxtal()

        try:

            # If use_icsd_statistic is False, for now do not pass wyckoff sites into pyxtal.
            volume_ok = my_crystal.from_random(
                wyckoff_indices_per_specie=chosen_wyckoff_indices
                if force_wyckoff_indices
                else None,
                use_given_wyckoff_sites=force_wyckoff_indices,
                dim=3,
                group=group_object,
                species=chosen_elements,
                numIons=chosen_numbers,
                # sites=chosen_wyckoff_positions,
                my_seed=seed,
                # factor=1.1,
                # factor=np.random.uniform(0.7, 5.0),
                # factor=np.random.uniform(0.7, 3.0),
                # factor=np.random.uniform(0.7, 1.2),
                factor=np.random.uniform(
                    0.7, 2.2
                ),  # trying to match the denseness factor distribution of ICSD
                do_distance_checks=do_distance_checks,
                fixed_volume=fixed_volume,
                do_merge_checks=do_merge_checks,
                max_volume=max_volume,
                max_count=5,
            )

            if not volume_ok:
                tries_counter += 1

                if not use_element_repetitions_instead_of_NO_wyckoffs:
                    print(
                        f"Volume too high, regenerating. (NO_wyckoffs: {NO_elements})"
                    )
                else:
                    print(
                        f"Volume too high, regenerating. (Number of unique elements: {NO_unique_elements})"
                    )

                continue

        except Exception as ex:
            print(flush=True)
            print(ex, flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)

            tries_counter += 1

            continue

        if not my_crystal.valid:
            print(flush=True)
            print("Generated a non-valid crystal. Something went wrong.", flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)

            tries_counter += 1

            continue

        try:

            # Only for comparing the debug code with the original code:
            # for site in my_crystal.atom_sites:
            #    site.coords = filtered_coords(site.coords)

            crystal = my_crystal.to_pymatgen()

            if do_symmetry_checks:

                # Make sure that the space group is actually correct / unique
                analyzer = SpacegroupAnalyzer(
                    crystal,
                    symprec=1e-8,
                    angle_tolerance=5.0,
                )

                checked_spg = analyzer.get_space_group_number()
                if checked_spg != group_object.number:

                    if not use_element_repetitions_instead_of_NO_wyckoffs:
                        print(
                            f"Mismatch in space group number, skipping structure. Generated: {group_object.number} Checked: {checked_spg}; NO_elements: {NO_elements}"
                        )
                    else:
                        print(
                            f"Mismatch in space group number, skipping structure. Generated: {group_object.number} Checked: {checked_spg}; Number of unique elements: {NO_unique_elements}"
                        )

                    tries_counter += 1

                    continue

        except Exception as ex:

            print(flush=True)
            print(ex, flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)

            tries_counter += 1

            continue

        # print(spacegroup_number)
        # print(chosen_elements)
        # print(chosen_numbers)
        # vis = StructureVis()
        # vis.set_structure(crystal)
        # vis.show()

        if not return_original_pyxtal_object:
            return crystal
        else:
            return crystal, my_crystal


def generate_structures(
    spacegroup_number,
    N,
    max_NO_elements=10,
    seed=-1,
    do_distance_checks=True,
    fixed_volume=None,
    do_merge_checks=True,
    use_icsd_statistics=False,
    probability_per_spg_per_element=None,
    probability_per_spg_per_wyckoff=None,
    max_volume=None,
    return_original_pyxtal_object=False,
    NO_wyckoffs_prob_per_spg=None,
    do_symmetry_checks=True,
    set_NO_elements_to_max=False,
    force_wyckoff_indices=True,
    use_element_repetitions_instead_of_NO_wyckoffs=False,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg=None,
    verbose=False,
):

    group = Group(spacegroup_number, dim=3)

    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)
    letters = [x.letter for x in group]

    # print(flush=True)
    # print(f"Current group: {spacegroup_number}", flush=True)
    # print(names, flush=True)
    # print(multiplicities, flush=True)
    # print(flush=True)

    result = [
        generate_structure(
            None,
            group_object=group,
            multiplicities=multiplicities,
            names=names,
            letters=letters,
            dofs=dofs,
            max_NO_elements=max_NO_elements,
            seed=seed,
            do_distance_checks=do_distance_checks,
            fixed_volume=fixed_volume,
            do_merge_checks=do_merge_checks,
            use_icsd_statistics=use_icsd_statistics,
            probability_per_spg_per_element=probability_per_spg_per_element,
            probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
            max_volume=max_volume,
            return_original_pyxtal_object=return_original_pyxtal_object,
            NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
            do_symmetry_checks=do_symmetry_checks,
            set_NO_elements_to_max=set_NO_elements_to_max,
            force_wyckoff_indices=force_wyckoff_indices,
            use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions_instead_of_NO_wyckoffs,
            NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg=NO_repetitions_prob_per_spg,
            verbose=verbose,
        )
        for i in range(0, N)
    ]

    return result


def get_wyckoff_info(pyxtal_crystal):
    # returns: Number of set wyckoffs, elements

    elements = []

    for site in pyxtal_crystal.atom_sites:
        specie_str = str(site.specie)
        elements.append(specie_str)

    return len(pyxtal_crystal.atom_sites), elements


def prepare_training(files_to_use_for_test_set=40):  # roughly 30%

    spgs = range(1, 231)

    jobid = os.getenv("SLURM_JOB_ID")
    path_to_patterns = "./patterns/icsd_vecsei/"

    if jobid is not None and jobid != "":
        sim_test = Simulation(
            "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/ws/uvgnh/Databases/ICSD/cif/",
        )
        sim_test.output_dir = path_to_patterns

        sim_statistics = Simulation(
            "/home/ws/uvgnh/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/ws/uvgnh/Databases/ICSD/cif/",
        )
        sim_statistics.output_dir = path_to_patterns
    else:
        sim_test = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        sim_test.output_dir = path_to_patterns

        sim_statistics = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        sim_statistics.output_dir = path_to_patterns

    sim_test.load(
        load_patterns_angles_intensities=False, start=0, stop=files_to_use_for_test_set
    )
    sim_statistics.load(
        load_patterns_angles_intensities=False, start=files_to_use_for_test_set
    )

    # Calculate the statistics from the sim_statistics part of the simulation:

    counts_per_spg_per_wyckoff = {}
    counter_per_spg_per_element = {}

    NO_wyckoffs_per_spg = {}
    NO_unique_elements_per_spg = {}
    NO_repetitions_per_spg_per_element = {}

    # pre-process the symmetry groups:

    for spg_number in spgs:

        group = Group(spg_number, dim=3)
        names = [(str(x.multiplicity) + x.letter) for x in group]

        counts_per_spg_per_wyckoff[spg_number] = {}
        for name in names:
            counts_per_spg_per_wyckoff[spg_number][name] = 0

        NO_wyckoffs_per_spg[spg_number] = []
        NO_unique_elements_per_spg[spg_number] = []
        NO_repetitions_per_spg_per_element[spg_number] = {}

        counter_per_spg_per_element[spg_number] = {}

    # Analyse the statistics:

    start = time.time()

    for i, crystal in enumerate(sim_statistics.sim_crystals):

        if (i % 100) == 0:
            print(f"{i / len(sim_statistics.sim_crystals) * 100} % processed.")

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

            struc = pyxtal()
            struc.from_seed(crystal)

            spg_number = (
                struc.group.number
            )  # use the group as calculated by pyxtal for statistics; this should be fine.

            NO_wyckoffs_per_spg[spg_number].append(len(struc.atom_sites))

            success = True
            try:
                NO_wyckoffs, elements = get_wyckoff_info(struc)
            except Exception as ex:
                print(ex)
                success = False

            if success:

                elements_unique = np.unique(elements)

                for el in elements_unique:
                    if "+" in el or "-" in el or "." in el or "," in el:
                        print("Ohoh")

                NO_unique_elements_per_spg[spg_number].append(len(elements_unique))

                for el in elements_unique:
                    reps = np.sum(np.array(elements) == el)

                    if el in NO_repetitions_per_spg_per_element[spg_number].keys():
                        NO_repetitions_per_spg_per_element[spg_number][el].append(reps)
                    else:
                        NO_repetitions_per_spg_per_element[spg_number][el] = [reps]

        except Exception as ex:

            print(f"Error reading structure:")
            print(ex)

            continue

        for site in struc.atom_sites:

            specie_str = str(site.specie)
            if specie_str in counter_per_spg_per_element[spg_number].keys():
                counter_per_spg_per_element[spg_number][specie_str] += 1
            else:
                counter_per_spg_per_element[spg_number][specie_str] = 1

            name = str(site.wp.multiplicity) + site.wp.letter
            counts_per_spg_per_wyckoff[spg_number][name] += 1

    represented_spgs = []
    NO_wyckoffs_prob_per_spg = {}

    NO_unique_elements_prob_per_spg = {}
    NO_repetitions_prob_per_spg_per_element = {}

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

            represented_spgs.append(spg)
        else:
            NO_wyckoffs_prob_per_spg[spg] = []
            NO_unique_elements_prob_per_spg[spg] = []
            NO_repetitions_prob_per_spg_per_element[spg] = {}

    print(f"Took {time.time() - start} s to calculate the statistics.")

    print("Processing test dataset...")
    start = time.time()

    corrected_labels = []
    count_mismatches = 0

    for i, crystal in enumerate(sim_test.sim_crystals):

        if (i % 100) == 0:
            print(f"{i / len(sim_test.sim_crystals) * 100} % processed.")

        try:

            spg_number_icsd = sim_test.sim_labels[i][0]

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

    print(f"{count_mismatches/len(sim_test.sim_crystals)*100}% mismatches in test set.")

    print(f"Took {time.time() - start} s to process the test dataset.")

    with open("prepared_training", "wb") as file:
        pickle.dump(
            (
                counter_per_spg_per_element,
                counts_per_spg_per_wyckoff,
                NO_wyckoffs_prob_per_spg,
                corrected_labels,
                files_to_use_for_test_set,
                represented_spgs,
                NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element,
            ),
            file,
        )


def load_dataset_info():

    with open(
        os.path.join(os.path.dirname(__file__), "prepared_training"),
        "rb",
    ) as file:
        data = pickle.load(file)
        counter_per_spg_per_element = data[0]
        counts_per_spg_per_wyckoff = data[1]
        NO_wyckoffs_prob_per_spg = data[2]
        corrected_labels = data[3]
        files_to_use_for_test_set = data[4]
        represented_spgs = data[5]
        NO_unique_elements_prob_per_spg = data[6]
        NO_repetitions_prob_per_spg = data[7]

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

    for spg in counts_per_spg_per_wyckoff.keys():
        total = 0
        for wyckoff_site in counts_per_spg_per_wyckoff[spg].keys():
            total += counts_per_spg_per_wyckoff[spg][wyckoff_site]
        for wyckoff_site in counts_per_spg_per_wyckoff[spg].keys():
            if total > 0:
                counts_per_spg_per_wyckoff[spg][wyckoff_site] /= total
            else:  # if no observations are present in the ICSD, make it evenly distributed
                counts_per_spg_per_wyckoff[spg][wyckoff_site] = 1 / len(
                    counts_per_spg_per_wyckoff[spg].keys()
                )
    probability_per_spg_per_wyckoff = counts_per_spg_per_wyckoff

    return (
        probability_per_spg_per_element,
        probability_per_spg_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
        corrected_labels,
        files_to_use_for_test_set,
        represented_spgs,  # spgs represented in the statistics dataset (70%)
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg,
    )


if __name__ == "__main__":

    if True:
        (
            probability_per_spg_per_element,
            probability_per_spg_per_wyckoff,
            NO_wyckoffs_prob_per_spg,
            corrected_labels,
            files_to_use_for_test_set,
            represented_spgs,
            NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg,
        ) = load_dataset_info()

        for i in range(0, 20 * 100):
            # for spg in represented_spgs:
            for spg in represented_spgs:
                # for spg in [2]:

                print(spg)

                generate_structures(
                    spg,
                    1,
                    100,
                    -1,
                    True,  # distance checks enabled!
                    None,
                    False,
                    True,
                    probability_per_spg_per_element,
                    probability_per_spg_per_wyckoff,
                    7000,
                    False,
                    NO_wyckoffs_prob_per_spg,
                    True,
                    False,
                    True,
                    True,
                    NO_unique_elements_prob_per_spg,
                    NO_repetitions_prob_per_spg,
                )

    if False:

        (
            probability_per_element,
            probability_per_spg_per_wyckoff,
            NO_wyckoffs_prob_per_spg,
            corrected_labels,
            files_to_use_for_test_set,
            represented_spgs,
        ) = load_dataset_info()

        for i in reversed(range(len(represented_spgs))):
            if np.sum(NO_wyckoffs_prob_per_spg[represented_spgs[i]][0:100]) <= 0.01:
                print(f"Excluded spg {represented_spgs[i]}")
                del represented_spgs[i]

        for i in range(0, 5):
            for spg in represented_spgs:

                print(spg)
                structure, orig_pyxtal_obj = generate_structures(
                    spg,
                    1,
                    100,
                    -1,
                    False,
                    None,
                    False,
                    True,
                    probability_per_element,
                    probability_per_spg_per_wyckoff,
                    7000,
                    True,
                    NO_wyckoffs_prob_per_spg,
                    True,
                    False,
                    True,
                )[0]

                # orig_NO_wyckoffs = len(orig_pyxtal_obj.atom_sites)

                # pyxtal_obj = pyxtal()
                # pyxtal_obj.from_seed(structure)

                # new_NO_wyckoffs = len(pyxtal_obj.atom_sites)

                # if orig_NO_wyckoffs != new_NO_wyckoffs:
                #    print("Ohoh")
                #    exit()

    if False:
        prepare_training()

    if False:

        data = load_dataset_info()
        print()

    if False:
        data = load_dataset_info()

        plt.plot(data[2])
        plt.show()

    if False:

        N = 10000

        # Compare the amount of spg skips due to wrong spg

        (
            probability_per_element,
            probability_per_spg_per_wyckoff,
            NO_wyckoffs_probability,
            corrected_labels,
            files_to_use_for_test_set,
        ) = load_dataset_info()

        spg_sets = [[2, 15], [14, 104, 129, 176]]

        for evenly_distributed in [True, False]:

            print(
                "Evenly distributed"
                if evenly_distributed
                else "Following ICSD distribution"
            )

            for spg_set in spg_sets:

                print(f"Spg set:")
                print(spg_set)

                for i in range(0, int(N / len(spg_set))):

                    for spg in spg_set:

                        generate_structures(
                            spacegroup_number=spg,
                            N=1,
                            max_NO_elements=100,
                            do_distance_checks=False,
                            do_merge_checks=False,
                            use_icsd_statistics=True,
                            probability_per_element=probability_per_element,
                            probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
                            max_volume=7000,
                            NO_wyckoffs_probability=NO_wyckoffs_probability
                            if not evenly_distributed
                            else None,
                            do_symmetry_checks=True,
                        )

    if False:

        mistakes = {}
        skipped = {}

        (
            probability_per_element,
            probability_per_spg_per_wyckoff,
        ) = load_wyckoff_statistics()

        counter_mistakes = 0
        counter_skipped = 0

        timings = []

        N = 50
        for i in range(0, N):

            print(i)

            # spgs = [2, 15]
            spgs = range(1, 231)
            for spg in spgs:
                structure = generate_structures(
                    spg,
                    1,
                    100,
                    do_distance_checks=False,
                    do_merge_checks=False,
                    use_icsd_statistics=True,
                    probability_per_element=probability_per_element,
                    probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
                )[0]

                try:

                    start = time.time()

                    analyzer = SpacegroupAnalyzer(
                        # structure, symprec=1e-4, angle_tolerance=5.0
                        structure,
                        symprec=1e-8,
                        angle_tolerance=5.0,
                    )
                    group_number = analyzer.get_space_group_number()

                    timings.append(time.time() - start)

                    # pyxtal_structure = pyxtal()
                    # pyxtal_structure.from_seed(structure)
                    # group_number = pyxtal_structure.group.number

                except Exception as ex:

                    print(ex)

                    counter_skipped += 1

                    if spg in skipped.keys():
                        skipped[spg] += 1
                    else:
                        skipped[spg] = 1

                    # try:
                    #    pyxtal_structure = pyxtal()
                    #    pyxtal_structure.from_seed(structure)
                    #    print(pyxtal_structure.group.number)
                    # except Exception as ex:
                    #    print(ex)

                    continue

                if spg != group_number:

                    counter_mistakes += 1

                    if spg in mistakes.keys():
                        mistakes[spg] += 1
                    else:
                        mistakes[spg] = 1

        print(f"{counter_mistakes / (len(spgs)*N) * 100}% mistakes")
        print(f"{counter_skipped / (len(spgs)*N) * 100}% skipped")

        print(f"Average timing: {np.mean(timings)}")

        counts_mistakes = [
            x[1]
            for x in sorted(zip(mistakes.keys(), mistakes.values()), key=lambda x: x[0])
        ]
        plt.bar(sorted(mistakes.keys()), counts_mistakes)
        plt.title("Mistakes")
        plt.show()

        counts_skipped = [
            x[1]
            for x in sorted(zip(skipped.keys(), skipped.values()), key=lambda x: x[0])
        ]
        plt.bar(sorted(skipped.keys()), counts_skipped)
        plt.title("Skipped")
        plt.show()

    if False:

        parser = CifParser("example.cif")
        structures_prim = parser.get_structures()[0]
        structures_conv = parser.get_structures(primitive=False)[0]

        print()

    if False:

        (
            probability_per_element,
            probability_per_spg_per_wyckoff,
        ) = load_wyckoff_statistics()

        structures = generate_structures(
            225,
            1,
            2,
            do_distance_checks=False,
            do_merge_checks=False,
            use_icsd_statistics=True,
            probability_per_element=probability_per_element,
            probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
            return_original_pyxtal_object=True,
        )

        pymatgen_structure = structures[0][0]
        pyxtal_structure = structures[0][1]

        struc = pyxtal()
        struc.from_seed(pymatgen_structure)

        pymatgen_s = struc.to_pymatgen()
        pymatgen_p = pymatgen_s.get_primitive_structure()

        struc_1 = pyxtal()
        struc_1.from_seed(pymatgen_p)

        print(pymatgen_structure.lattice)
        print()
        print(pyxtal_structure.lattice)
        print()
        print(struc.lattice)
        print()
        print(pymatgen_s.lattice)
        print()
        print(pymatgen_p.lattice)
        print()
        print(struc_1.lattice)
        print()
        print()

        print(pymatgen_structure.composition)
        print()
        print(pyxtal_structure.formula)
        print()
        print(struc.formula)
        print()
        print(pymatgen_s.composition)
        print()
        print(pymatgen_p.composition)
        print()
        print(struc_1.formula)

        print()

    if False:
        analyse_set_wyckoffs([2, 15, 14, 104, 129, 176], load_only=1)

    if False:

        (
            probability_per_element,
            probability_per_spg_per_wyckoff,
        ) = load_wyckoff_statistics()

        volumes = []

        for spg in range(1, 231):

            print(f"Spg {spg}")

            structures = generate_structures(
                spg,
                2,
                100,
                do_distance_checks=False,
                do_merge_checks=False,
                use_icsd_statistics=True,
                probability_per_element=probability_per_element,
                probability_per_spg_per_wyckoff=probability_per_spg_per_wyckoff,
                max_volume=7000,
            )

            for structure in structures:
                volumes.append(structure.volume)

        volumes = np.array(volumes)
        print(f"Volumes <= 7000: {np.sum(volumes <= 7000)}")
        print(f"Volumes > 7000: {np.sum(volumes > 7000)}")

        bins = np.linspace(
            np.min(volumes),
            np.max(volumes),
            60,
        )
        bin_width = bins[1] - bins[0]
        hist, edges = np.histogram(volumes, bins, density=True)

        plt.bar(bins[:-1], hist, width=bin_width)
        plt.show()

    if False:

        NO_chosen_elements = 50

        seed = 5215
        number_per_spg = 1
        low = 1
        high = 231
        np.random.seed(seed)
        random.seed(seed)

        structure_seeds = np.random.randint(0, 10000, 230 * number_per_spg)

        # To pre-compile functions:
        for spg in range(low, high):
            generate_structures(
                spg,
                number_per_spg,
                seed=int(structure_seeds[spg - 1]),
                do_distance_checks=False,
            )

        start = time.time()
        for spg in range(low, high):
            generate_structures(
                spg,
                number_per_spg,
                seed=int(structure_seeds[spg - 1]),
                do_distance_checks=False,
                max_NO_elements=NO_chosen_elements,
            )
        stop = time.time()
        print(f"No distance checks: {stop-start} s", flush=True)

        if False:

            start = time.time()
            for spg in range(low, high):
                generate_structures(
                    spg,
                    number_per_spg,
                    seed=int(structure_seeds[spg - 1]),
                    do_distance_checks=True,
                    max_NO_elements=NO_chosen_elements,
                )
            stop = time.time()
            print(f"With distance checks: {stop-start} s", flush=True)

    if False:

        seed = 5215
        number_per_spg = 1

        low = 1
        high = 13

        np.random.seed(seed)
        random.seed(seed)

        structure_seeds = np.random.randint(0, 10000, 230 * number_per_spg)

        # generate_structures(13, 100, seed=seed)

        # To pre-compile functions:
        for spg in range(low, high):
            generate_structures(spg, number_per_spg, seed=int(structure_seeds[spg - 1]))

        start = time.time()

        results = []
        timings = []
        for spg in range(low, high):
            start_inner = time.time()
            results.extend(
                generate_structures(
                    spg, number_per_spg, seed=int(structure_seeds[spg - 1])
                )
            )
            timings.append(time.time() - start_inner)

        with open("compare_debug", "wb") as file:
            coords = []
            for crystal in results:
                coords.append(crystal.cart_coords)
            pickle.dump(coords, file)

        stop = time.time()

        print(f"Total job took {stop-start} s", flush=True)

    if False:

        with open("compare_debug", "rb") as file:
            coords_debug = pickle.load(file)

        with open("compare_original", "rb") as file:
            coords_original = pickle.load(file)

        for i, coor in enumerate(coords_debug):
            for j, coordinate in enumerate(coor):
                compare_to = coords_original[i][j]

                if np.sum(np.square(coordinate - compare_to)) > 10 ** (-10):
                    print(f"Oh oh {i} {j}")

                    # if j == 0:
                    #    print(coords_original[i])
                    #    print(coords_debug[i])
