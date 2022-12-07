from sympy import E
from pyxtal import pyxtal
import matplotlib.pyplot as plt
import numpy as np
import random
from pyxtal.symmetry import Group
import time
import pickle
from tools.simulation.simulator import Simulator
from pymatgen.io.cif import CifParser
import os
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tools.generation.structure_generation import generate_pyxtal_object
from all_elements import all_elements


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
    probability_per_spg_per_element_per_wyckoff=None,
    max_volume=None,
    return_original_pyxtal_object=False,
    NO_wyckoffs_prob_per_spg=None,
    do_symmetry_checks=True,
    set_NO_elements_to_max=False,
    force_wyckoff_indices=True,
    use_element_repetitions_instead_of_NO_wyckoffs=False,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg_per_element=None,
    verbose=False,
    denseness_factors_density_per_spg=None,
    kde_per_spg=None,
    all_data_per_spg=None,
    use_coordinates_directly=False,
    use_lattice_paras_directly=False,
    use_alternative_structure_generator_implementation=True,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    per_element=False,
    verbosity=2,
):

    if use_icsd_statistics and (
        probability_per_spg_per_element is None
        or probability_per_spg_per_element_per_wyckoff is None
    ):
        raise Exception("Statistics data needed if use_icsd_statistics = True.")

    if use_element_repetitions_instead_of_NO_wyckoffs and (
        NO_unique_elements_prob_per_spg is None
        or NO_repetitions_prob_per_spg_per_element is None
    ):
        raise Exception(
            "Statistics data needed if use_element_repetitions_instead_of_NO_wyckoffs = True."
        )

    if seed != -1:
        np.random.seed(seed)
        random.seed(seed)

    if (
        not use_element_repetitions_instead_of_NO_wyckoffs
        and kde_per_spg is None
        and all_data_per_spg is None
    ):

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

    elif (
        use_element_repetitions_instead_of_NO_wyckoffs or kde_per_spg is not None
    ) and all_data_per_spg is None:

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
                probability_per_spg_per_element_per_wyckoff,
                max_volume,
                return_original_pyxtal_object,
                NO_wyckoffs_prob_per_spg,
                do_symmetry_checks,
                set_NO_elements_to_max,
                force_wyckoff_indices=force_wyckoff_indices,
                use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions_instead_of_NO_wyckoffs,
                NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
                verbose=verbose,
                denseness_factors_density_per_spg=denseness_factors_density_per_spg,
                kde_per_spg=kde_per_spg,
                all_data_per_spg=all_data_per_spg,
                use_coordinates_directly=use_coordinates_directly,
                use_lattice_paras_directly=use_lattice_paras_directly,
                use_alternative_structure_generator_implementation=use_alternative_structure_generator_implementation,
                denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                per_element=per_element,
                verbosity=verbosity,
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

        if kde_per_spg is None and all_data_per_spg is None:

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
                        print(
                            "More than 50 collisions setting an atom, continuing with next unique element."
                        )

                        current_repetition_counter = current_picked_repetition  # force that loop goes to next unique element
                        break

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

                            current_picked_repetition = np.random.choice(
                                range(
                                    1,
                                    len(
                                        NO_repetitions_prob_per_spg_per_element[
                                            group_object.number
                                        ][current_element]
                                        if per_element
                                        else NO_repetitions_prob_per_spg_per_element[
                                            group_object.number
                                        ]
                                    )
                                    + 1,
                                ),
                                1,
                                p=list(
                                    NO_repetitions_prob_per_spg_per_element[
                                        group_object.number
                                    ][current_element]
                                    if per_element
                                    else NO_repetitions_prob_per_spg_per_element[
                                        group_object.number
                                    ]
                                ),
                            )[0]
                            current_repetition_counter = 1

                            unique_elements_counter += 1

                        else:

                            current_repetition_counter += 1

                        chosen_elements.append(current_element)

                    if not use_icsd_statistics:
                        chosen_index = random.randint(
                            0, len(number_of_atoms_per_site) - 1
                        )
                    else:
                        probability_per_wyckoff = (
                            probability_per_spg_per_element_per_wyckoff[
                                group_object.number
                            ][current_element]
                            if per_element
                            else probability_per_spg_per_element_per_wyckoff[
                                group_object.number
                            ]
                        )

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

                        # need to reset the changes that were made above, since this didn't work out:
                        chosen_elements.pop()
                        current_repetition_counter -= 1

                        continue  # try again

                    number_of_atoms_per_site[chosen_index] += 1

                    chosen_numbers.append(multiplicities[chosen_index])
                    chosen_wyckoff_positions.append([names[chosen_index]])
                    chosen_wyckoff_letters.append([letters[chosen_index]])
                    chosen_wyckoff_indices.append(chosen_index)

                    break

                if not use_element_repetitions_instead_of_NO_wyckoffs:
                    set_wyckoffs_counter += 1

        elif kde_per_spg is not None and all_data_per_spg is None:

            # sample wyckoff occupations directly from the ICSD:

            occupations = [
                int(item) if item >= 0 else 0
                for item in np.round(kde_per_spg[group_object.number].sample(1)[0])
            ]

            for i in range(0, len(occupations)):

                occ = occupations[i]

                if dofs[i] == 0:
                    occ = min(1, occ)

                number_of_atoms_per_site[i] = occ

            NO_wyckoffs = np.sum(number_of_atoms_per_site)

            if NO_wyckoffs < 1.0:
                print("NO_wyckoffs = 0.0, regenerating...")
                continue

            if NO_unique_elements > NO_wyckoffs:
                NO_unique_elements = int(NO_wyckoffs)

            # Choose unique elements:
            unique_elements = []
            for i in range(0, NO_unique_elements):
                while True:
                    current_element = np.random.choice(
                        list(
                            probability_per_spg_per_element[group_object.number].keys()
                        ),
                        1,
                        p=list(
                            probability_per_spg_per_element[
                                group_object.number
                            ].values()
                        ),
                    )[0]

                    if current_element not in unique_elements:
                        unique_elements.append(current_element)
                        break

            a = []
            for i in range(0, len(unique_elements) - 1):
                while True:
                    chosen_frac = np.random.randint(1, NO_wyckoffs)
                    if chosen_frac not in a:
                        a.append(chosen_frac)
                        break
                    else:
                        continue

            a = sorted(a)

            N_per_element = np.append(a, NO_wyckoffs) - np.insert(a, 0, 0)
            assert (
                np.sum(N_per_element) == NO_wyckoffs
            )  # important for break condition below

            chosen_elements = []
            for i, current_N in enumerate(list(N_per_element)):
                chosen_elements.extend([unique_elements[i]] * int(current_N))

            random.shuffle(chosen_elements)

            chosen = np.zeros(len(names))
            current_wyckoff_index = 0

            for i, el in enumerate(chosen_elements):

                while True:

                    if (
                        chosen[current_wyckoff_index]
                        < number_of_atoms_per_site[current_wyckoff_index]
                    ):
                        chosen[current_wyckoff_index] += 1

                        chosen_wyckoff_indices.append(current_wyckoff_index)
                        chosen_numbers.append(multiplicities[current_wyckoff_index])

                        break

                    else:

                        current_wyckoff_index += 1
                        continue

        else:

            all_data = all_data_per_spg[group_object.number]
            chosen_entry = random.choice(all_data)

            for item in chosen_entry["occupations"]:
                el = item[0]
                wyckoff_name = item[1]
                # position = item[2]
                wyckoff_index = names.index(wyckoff_name)

                chosen_elements.append(el)
                chosen_numbers.append(multiplicities[wyckoff_index])
                chosen_wyckoff_indices.append(wyckoff_index)

        if (
            use_element_repetitions_instead_of_NO_wyckoffs
            or kde_per_spg is not None
            or all_data_per_spg is not None
        ):
            if len(chosen_numbers) > max_NO_elements:
                if verbosity != 2:
                    print(
                        f"Too many total number of set wyckoff sites for spg {group_object.number}, regenerating..."
                    )
                continue  # but do not increase tries_counter, this is totally fine and expected!

        if verbose:
            print(f"Number of chosen wyckoff sites: {len(chosen_numbers)}")

        if (
            denseness_factors_density_per_spg is None
            or denseness_factors_density_per_spg[group_object.number] is None
            or (
                denseness_factors_conditional_sampler_seeds_per_spg is not None
                and denseness_factors_conditional_sampler_seeds_per_spg[
                    group_object.number
                ]
                is None
            )
        ):
            # factor = np.random.uniform(0.7, 2.2)
            factor = np.random.uniform(0.7, 2.13)
        elif denseness_factors_conditional_sampler_seeds_per_spg is None:
            while True:
                factor = denseness_factors_density_per_spg[
                    group_object.number
                ].resample(1)[0, 0]

                if factor > 0.0:
                    break
        else:
            factor = None

        if not use_alternative_structure_generator_implementation:

            if factor is None:
                raise Exception(
                    "Conditional kde for denseness_factor not supported in this implementation mode."
                )

            if lattice_paras_density_per_lattice_type is not None:
                raise Exception(
                    "KDE sampling of lattice parameters not supported in this implementation mode."
                )

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
                    factor=factor,
                    do_distance_checks=do_distance_checks,
                    fixed_volume=fixed_volume,
                    do_merge_checks=do_merge_checks,
                    max_volume=max_volume,
                    max_count=5,
                )

                if not volume_ok:
                    tries_counter += 1

                    if verbosity != 2:
                        if (
                            not use_element_repetitions_instead_of_NO_wyckoffs
                            and kde_per_spg is None
                            and all_data_per_spg is None
                        ):
                            print(
                                f"Volume too high, regenerating. (NO_wyckoffs: {NO_elements})"
                            )
                        elif all_data_per_spg is None:
                            print(
                                f"Volume too high, regenerating. (Number of unique elements: {NO_unique_elements})"
                            )
                        else:
                            print(f"Volume too high, regenerating.")

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
                print(
                    "Generated a non-valid crystal. Something went wrong.", flush=True
                )
                print(group_object.number, flush=True)
                print(chosen_elements, flush=True)
                print(chosen_numbers, flush=True)
                print(flush=True)

                tries_counter += 1

                continue

            # This is by no means the fastest way to implement this.
            # But since this is just for understanding the problem / testing, it is fine.
            if use_coordinates_directly and all_data_per_spg is not None:

                # potentially useful in the future: chosen_entry["lattice_parameters"]

                occupations_copy = chosen_entry["occupations"].copy()

                for atom in my_crystal.atom_sites:

                    wyckoff_name_atom = str(atom.wp.multiplicity) + atom.wp.letter
                    el_atom = atom.specie

                    for i, item in enumerate(occupations_copy):
                        el = item[0]
                        wyckoff_name = item[1]

                        if el == el_atom and wyckoff_name == wyckoff_name_atom:

                            atom.position = item[2]
                            atom.update()  # important

                            # Not really needed:
                            # new_site.coords = filtered_coords(new_site.coords)

                            del occupations_copy[i]
                            break

                assert len(occupations_copy) == 0

            if use_lattice_paras_directly and all_data_per_spg is not None:

                my_crystal.lattice.set_para(
                    chosen_entry["lattice_parameters"], radians=True
                )

        else:

            if (use_coordinates_directly and all_data_per_spg is not None) or (
                use_lattice_paras_directly and all_data_per_spg is not None
            ):
                raise Exception("Mode not supported.")

            try:
                my_crystal = generate_pyxtal_object(
                    group_object=group_object,
                    factor=factor,
                    species=chosen_elements,
                    chosen_wyckoff_indices=chosen_wyckoff_indices,
                    multiplicities=chosen_numbers,
                    max_volume=max_volume,
                    scale_volume_min_density=True,  # TODO: Maybe change
                    denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                    lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                    fixed_volume=fixed_volume,
                )
            except Exception as ex:
                print(flush=True)
                print(ex, flush=True)
                print(group_object.number, flush=True)
                print(chosen_elements, flush=True)
                print(chosen_numbers, flush=True)
                print(flush=True)

                tries_counter += 1

                continue

            if not my_crystal:
                tries_counter += 1

                if verbosity != 2:
                    if (
                        not use_element_repetitions_instead_of_NO_wyckoffs
                        and kde_per_spg is None
                        and all_data_per_spg is None
                    ):
                        print(
                            f"Volume too high, regenerating. (NO_wyckoffs: {NO_elements})"
                        )
                    elif all_data_per_spg is None:
                        print(
                            f"Volume too high, regenerating. (Number of unique elements: {NO_unique_elements})"
                        )
                    else:
                        print(f"Volume too high, regenerating.")

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

                    if verbosity != 2:
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
    probability_per_spg_per_element_per_wyckoff=None,
    max_volume=None,
    return_original_pyxtal_object=False,
    NO_wyckoffs_prob_per_spg=None,
    do_symmetry_checks=True,
    set_NO_elements_to_max=False,
    force_wyckoff_indices=True,
    use_element_repetitions_instead_of_NO_wyckoffs=False,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg_per_element=None,
    verbose=False,
    denseness_factors_density_per_spg=None,
    kde_per_spg=None,
    all_data_per_spg=None,
    use_coordinates_directly=False,
    use_lattice_paras_directly=False,
    group_object=None,  # for speedup
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    per_element=False,
    verbosity=2,
):

    if group_object is None:
        group = Group(spacegroup_number, dim=3)
    else:
        group = group_object

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
            probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
            max_volume=max_volume,
            return_original_pyxtal_object=return_original_pyxtal_object,
            NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
            do_symmetry_checks=do_symmetry_checks,
            set_NO_elements_to_max=set_NO_elements_to_max,
            force_wyckoff_indices=force_wyckoff_indices,
            use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions_instead_of_NO_wyckoffs,
            NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
            verbose=verbose,
            denseness_factors_density_per_spg=denseness_factors_density_per_spg,
            kde_per_spg=kde_per_spg,
            all_data_per_spg=all_data_per_spg,
            use_coordinates_directly=use_coordinates_directly,
            use_lattice_paras_directly=use_lattice_paras_directly,
            denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
            lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
            per_element=per_element,
            verbosity=verbosity,
        )
        for i in range(0, N)
    ]

    return result


def convert_to_new_format(input_file="prepared_training_old_format"):

    with open(
        input_file,
        "rb",
    ) as file:
        data = pickle.load(file)
        per_element = data[7]
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
        all_data_per_spg = data[6]
        (
            statistics_metas,
            statistics_crystals,
            statistics_match_metas,
            test_metas,
            test_labels,
            test_crystals,
            corrected_labels,
            test_match_metas,
            test_match_pure_metas,
        ) = data[8]

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
                statistics_match_metas,
                test_metas,
                test_labels,
                corrected_labels,
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


def convert_add_statistics_labels():

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
        statistics_match_metas = data[8]
        test_metas = data[9]
        test_labels = data[10]
        corrected_labels = list(reversed(data[11]))
        test_match_metas = data[12]
        test_match_pure_metas = data[13]

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

    statistics_labels = []
    statistics_match_labels = []

    sim_metas_flat = [item[0] for item in sim.sim_metas]

    for meta in statistics_metas:
        statistics_labels.append(sim.sim_labels[sim_metas_flat.index(meta[0])])
    for meta in statistics_match_metas:
        statistics_match_labels.append(sim.sim_labels[sim_metas_flat.index(meta[0])])

    with open("prepared_training/meta_new", "wb") as file:
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
                corrected_labels,
                test_match_metas,
                test_match_pure_metas,
            ),
            file,
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

    if False:

        (
            probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff,
            NO_wyckoffs_prob_per_spg,
            corrected_labels,
            statistics_metas,
            test_metas,
            represented_spgs,  # spgs represented in the statistics dataset (70%)
            NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element,
            denseness_factors_density_per_spg,
            kde_per_spg,
            all_data_per_spg,
            denseness_factors_conditional_sampler_seeds_per_spg,
            lattice_paras_density_per_lattice_type,
        ) = load_dataset_info()

        timings = []
        for i in range(0, 10):
            # for spg in represented_spgs:

            start = time.time()

            for j in range(0, 2):
                for spg in range(1, 231):

                    print(spg)
                    if denseness_factors_density_per_spg[spg] is None:
                        continue

                    structure = generate_structures(
                        spg,
                        1,
                        100,
                        -1,
                        False,
                        None,
                        False,
                        True,
                        probability_per_spg_per_element,
                        probability_per_spg_per_element_per_wyckoff,
                        7000,
                        False,
                        NO_wyckoffs_prob_per_spg,
                        True,
                        False,
                        True,
                        True,
                        NO_unique_elements_prob_per_spg,
                        NO_repetitions_prob_per_spg_per_element,
                        False,
                        denseness_factors_density_per_spg,
                        None,
                        None,
                        False,
                        False,
                        None,
                        denseness_factors_conditional_sampler_seeds_per_spg,
                        lattice_paras_density_per_lattice_type,
                    )[0]
            timings.append(time.time() - start)

        print(np.average(timings))

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
        prepare_training(per_element=False)

    if False:
        data = load_dataset_info(check_for_sum_formula_overlap=False)

    if False:
        show_dataset_statistics()

    if False:
        convert_add_statistics_labels()

    if False:

        convert_to_new_format()

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
