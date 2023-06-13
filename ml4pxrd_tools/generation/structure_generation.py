"""
Contains code to generate synthetic crystal structures based
on the symmetry operations of the specified space group.
"""

import numpy as np
from pyxtal.database.element import Element
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.crystal import atom_site
from pyxtal import pyxtal
import numpy as np
import random
from pyxtal.symmetry import Group
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ml4pxrd_tools.manage_dataset import load_dataset_info


def rejection_sampler(p, xbounds, pmax):
    """Sample from probability density function p(x).

    Args:
        p (function): Probability density function
        xbounds (tuple): (x_min, x_max)
        pmax (float): Maximum probability in probability density function p(x).

    Returns:
        float: x-value sampled from p(x). Returns None if not converged after 10000 iterations.
    """

    counter = 0

    while True:
        x = (np.random.rand(1) * (xbounds[1] - xbounds[0]) + xbounds[0])[0]
        y = (np.random.rand(1) * pmax)[0]
        if y <= p(x):
            return x

        if counter > 10000:
            return None

        counter += 1


def sample_denseness_factor(volume, sampler):
    """Sample a denseness factor conditioned on the sum of atomic volumes `volume`.

    Args:
        volume (float): Sum of atomic volumes, see publication.
        sampler (tuple): (conditional probability density of type statsmodels.api.nonparametric.KDEMultivariateConditional, minimum denseness factor, maximum denseness factor)

    Returns:
        float: Sampled denseness factor
    """

    conditional_density = sampler[0]
    min_denseness_factors = sampler[1]
    max_denseness_factors = sampler[2]

    return rejection_sampler(
        lambda factor: conditional_density.pdf([factor], [volume]),
        (0, max_denseness_factors + 2),
        np.max(
            [
                conditional_density.pdf(
                    [factor_lin],
                    [volume],
                )
                for factor_lin in np.linspace(
                    min_denseness_factors,
                    max_denseness_factors,
                    100,
                )
            ]
        ),
    )


def sample_lattice_paras(volume, lattice_type, lattice_paras_density_per_lattice_type):
    """Sample lattice parameters using the given lattice type and corresponding KDE.

    Args:
        volume (float): Target volume of lattice.
        lattice_type (str): Lattice type to generate: cubic, hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, or triclinic
        lattice_paras_density_per_lattice_type (dict of scipy.stats.kde.gaussian_kde): Dictionary yielding the KDE for each lattice type.

    Returns:
        tuple: (a,b,c,alpha,beta,gamma)
    """

    if lattice_type not in ["cubic", "Cubic"]:
        density = lattice_paras_density_per_lattice_type[lattice_type]
        paras_constrained = density.resample(1).T[0]

    if lattice_type in ["cubic", "Cubic"]:
        paras = [1, 1, 1, np.pi / 2, np.pi / 2, np.pi / 2]

    elif lattice_type in ["hexagonal", "trigonal", "Hexagonal", "Trigonal"]:
        paras = [
            paras_constrained[0],
            paras_constrained[0],
            paras_constrained[1],
            np.pi / 2,
            np.pi / 2,
            np.pi * 2 / 3,
        ]

    elif lattice_type in ["tetragonal", "Tetragonal"]:
        paras = [
            paras_constrained[0],
            paras_constrained[0],
            paras_constrained[1],
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

    elif lattice_type in ["orthorhombic", "Orthorhombic"]:
        paras = [
            paras_constrained[0],
            paras_constrained[1],
            paras_constrained[2],
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

    elif lattice_type in ["monoclinic", "Monoclinic"]:
        paras = [
            paras_constrained[0],
            paras_constrained[1],
            paras_constrained[2],
            np.pi / 2,
            paras_constrained[3],
            np.pi / 2,
        ]

    elif lattice_type == "triclinic":
        paras = paras_constrained

    else:
        raise Exception(f"Invalid lattice type {lattice_type}")

    cbrt_volume = np.cbrt(volume)

    return [
        cbrt_volume * paras[0],
        cbrt_volume * paras[1],
        cbrt_volume * paras[2],
        paras[3],
        paras[4],
        paras[5],
    ]


def create_pyxtal_object(
    group_object,
    factor,
    species,
    chosen_wyckoff_indices,
    multiplicities,
    max_volume,
    scale_volume_min_density=True,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    fixed_volume=None,
):
    """Used to generate a pyxtal object of a synthetic crystal using the given parameters.

    Args:
        group_object (pyxtal.symmetry.Group): The symmetry operations of this Group object will be used to generate the crystal.
        factor (float|None): The denseness factor to multiply the sum of atomic volumes with, yielding the unit cell volume.
        species (list of str): Species to set on wyckoff positions.
        chosen_wyckoff_indices (list of int): Wyckoff position (index) to use for each of the species.
        multiplicities (list[int]): Multiplicity of each of the chosen Wyckoff sites.
        max_volume (volume): Maximum value of lattice volume. Returns False if volume > max_volume.
        scale_volume_min_density (bool, optional): Whether or not to scale volume so it matches the minimum density. Defaults to True.
        denseness_factors_conditional_sampler_seeds_per_spg (dict of tuple, optional): dictionary containing tuple for each spg:
            (conditional probability density of denseness factor conditioned on sum of atomic volumes of type statsmodels.api.nonparametric.KDEMultivariateConditional,
            minimum denseness factor, maximum denseness factor). Only used if factor is None. Defaults to None.
        lattice_paras_density_per_lattice_type (dict of scipy.stats.kde.gaussian_kde | None): Dictionary yielding the KDE to sample lattice parameters for each lattice type.
            If this is None, the functions of pyxtal are used to generate lattice parameters.
        fixed_volume (float|None, optional): Feed in a fixed volume, so the volume is not generated using the denseness factor. Defaults to None.

    Returns:
        pyxtal.pyxtal: pyxtal crystal object
    """

    # Steps to complete by this function:
    # 1) calculate the sum of covalent volumes for the given species
    # 2) calculate actual volume of crystal (by multiplying by denseness factor)
    # 3) generate a lattice with the given volume
    # 4) create the pyxtal object with uniform random coordinates

    ##### 1)

    if fixed_volume is None:
        volume = 0
        for numIon, specie in zip(multiplicities, species):
            r = (Element(specie).covalent_radius + Element(specie).vdw_radius) / 2
            volume += numIon * 4 / 3 * np.pi * r**3

        if factor is not None:
            volume *= factor
        else:
            # Sample the denseness factor

            max_sum_cov_volumes = denseness_factors_conditional_sampler_seeds_per_spg[
                group_object.number
            ][3]

            if volume > max_sum_cov_volumes:
                return False

            factor = sample_denseness_factor(
                volume,
                denseness_factors_conditional_sampler_seeds_per_spg[
                    group_object.number
                ],
            )

            if factor is None:  # rejection sampler didn't converge
                return False

            volume *= factor

        if scale_volume_min_density:
            min_density = 0.75
            # make sure the volume is not too small
            if volume / sum(multiplicities) < min_density:
                volume = sum(multiplicities) * min_density
                print("Volume has been scaled to match minimum density.")

        if volume > max_volume:
            return False

    else:
        volume = fixed_volume

    pyxtal_object = pyxtal(molecular=False)

    pyxtal_object.lattice = Lattice(group_object.lattice_type, volume)

    # If a KDE for the lattice parameters is given, sample them.
    # Otherwise, the lattice parameters as generated by pyxtal are used.
    if lattice_paras_density_per_lattice_type is not None:
        paras = sample_lattice_paras(
            volume,
            group_object.lattice_type,
            lattice_paras_density_per_lattice_type,
        )
        pyxtal_object.lattice.set_para(paras, radians=True)

    # Place the given species on the given wyckoff indices
    for specie, wyckoff_index in zip(species, chosen_wyckoff_indices):
        wyckoff = group_object.get_wyckoff_position(wyckoff_index)

        # Generate a uniform coordinate
        random_coord = pyxtal_object.lattice.generate_point()
        projected_coord = wyckoff.project(random_coord, pyxtal_object.lattice.matrix)

        new_atom_site = atom_site(wp=wyckoff, coordinate=projected_coord, specie=specie)
        pyxtal_object.atom_sites.append(new_atom_site)

    pyxtal_object.valid = True

    return pyxtal_object


def randomize(
    crystals,
    randomize_coordinates=True,
    randomize_lattice=False,
    lattice_paras_density_per_lattice_type=None,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    spgs=None,
):
    """This function can be used to randomize parts of a given (ICSD) crystal.
    If randomize_coordinates is True, then the coordinates are replaced with uniformly
    sampled coordinates. If randomize_lattice is True, the lattice parameters are
    resampled using the given KDE in lattice_paras_density_per_lattice_type.

    Args:
        crystals (list of pymatgen.core.structure): Crystals to operate on.
        randomize_coordinates (bool, optional): Whether or not to randomize the coordinates. Defaults to True.
        randomize_lattice (bool, optional): Whether or not to randomize the lattice parameters. Defaults to False.
        lattice_paras_density_per_lattice_type (dict of scipy.stats.kde.gaussian_kde|None): Dictionary yielding the KDE for each lattice type.
            If randomize_lattice is False, this is not used and can be None.
        denseness_factors_conditional_sampler_seeds_per_spg (dict of tuple, optional): dictionary containing tuple for each spg:
            (conditional probability density of denseness factor conditioned on sum of atomic volumes of type statsmodels.api.nonparametric.KDEMultivariateConditional,
            minimum denseness factor, maximum denseness factor). Defaults to None.
        spgs (list of int, optional): List of space group numbers to operate on. Defaults to None.
    Returns:
        tuple: (list of randomized crystals, list of reference crystals, spg as output by pyxtal)
            The reference crystals are rebuilt using the pyxtal object, yielding essentially the same crystals as in `crystals`.
            However, since pyxtal does not handle partial occupancies, the reference crystals are slightly different.
    """

    reference_crystals = []
    randomized_crystals = []
    labels = []

    for crystal in crystals:
        pyxtal_object = pyxtal()

        try:
            pyxtal_object.from_seed(crystal)

            if pyxtal_object.group.number not in spgs:
                labels.append(None)
                reference_crystals.append(None)
                randomized_crystals.append(None)

                continue

            # Determine the atomic volume V_atomic of the structure
            volume = 0
            for site in pyxtal_object.atom_sites:
                specie_str = str(site.specie)
                r = (
                    Element(specie_str).covalent_radius + Element(specie_str).vdw_radius
                ) / 2
                volume += site.wp.multiplicity * 4 / 3 * np.pi * r**3

        except Exception as ex:
            print(ex)

            labels.append(None)
            reference_crystals.append(None)
            randomized_crystals.append(None)

            continue

        max_sum_cov_volumes = denseness_factors_conditional_sampler_seeds_per_spg[
            pyxtal_object.group.number
        ][3]

        if volume > max_sum_cov_volumes:
            labels.append(None)
            reference_crystals.append(None)
            randomized_crystals.append(None)

            continue

        factor = sample_denseness_factor(
            volume,
            denseness_factors_conditional_sampler_seeds_per_spg[
                pyxtal_object.group.number
            ],
        )

        if factor is None:  # rejection sampler didn't converge
            labels.append(None)
            reference_crystals.append(None)
            randomized_crystals.append(None)

            continue

        volume *= factor

        if volume > 7000:
            labels.append(None)
            reference_crystals.append(None)
            randomized_crystals.append(None)

            continue

        labels.append(pyxtal_object.group.number)

        reference_crystal = pyxtal_object.to_pymatgen()
        reference_crystals.append(reference_crystal)

        if randomize_lattice:  # regenerate the lattice
            pyxtal_object.lattice = Lattice(pyxtal_object.group.lattice_type, volume)

            if lattice_paras_density_per_lattice_type is not None:
                paras = sample_lattice_paras(
                    volume,
                    pyxtal_object.group.lattice_type,
                    lattice_paras_density_per_lattice_type,
                )
                pyxtal_object.lattice.set_para(paras, radians=True)

        if randomize_coordinates:  # regenerate coordinates
            for site in pyxtal_object.atom_sites:
                wyckoff = site.wp

                random_coord = pyxtal_object.lattice.generate_point()
                projected_coord = wyckoff.project(
                    random_coord, pyxtal_object.lattice.matrix
                )

                site.update(pos=projected_coord)

        randomized_crystal = pyxtal_object.to_pymatgen()
        randomized_crystals.append(randomized_crystal)

    return randomized_crystals, reference_crystals, labels


def __generate_structure(
    _,
    group_object,
    multiplicities,
    names,
    letters,
    dofs,
    max_NO_atoms_asymmetric_unit,
    max_volume,
    fixed_volume,
    probability_per_spg_per_element,
    probability_per_spg_per_element_per_wyckoff,
    NO_unique_elements_prob_per_spg,
    NO_repetitions_prob_per_spg_per_element,
    per_element,
    return_original_pyxtal_object,
    do_symmetry_checks,
    denseness_factors_density_per_spg,
    denseness_factors_conditional_sampler_seeds_per_spg,
    lattice_paras_density_per_lattice_type,
    seed,
    is_verbose,
):
    if (
        probability_per_spg_per_element is None
        or probability_per_spg_per_element_per_wyckoff is None
        or NO_unique_elements_prob_per_spg is None
        or NO_repetitions_prob_per_spg_per_element is None
    ):
        raise Exception("Statistics data missing.")

    if seed != -1:
        np.random.seed(seed)
        random.seed(seed)

    # Pick number of unique elements to sample:
    NO_unique_elements = np.random.choice(
        range(1, len(NO_unique_elements_prob_per_spg[group_object.number]) + 1),
        size=1,
        p=NO_unique_elements_prob_per_spg[group_object.number],
    )[0]
    if is_verbose:
        print(f"NO_unique_elements: {NO_unique_elements}")

    tries_counter = 0

    while True:
        # If trying 20 times to generate a crystal with the given NO_unique_elements fails, then pick a new
        # NO_unique_elements and call this function recursively. This should always return at some point.
        if tries_counter > 20:
            print(
                f"Failed generating crystal of spg {group_object.number} with {NO_unique_elements} unique elements 10 times. Choosing new NO_unique_elements now."
            )

            return __generate_structure(
                _,
                group_object,
                multiplicities,
                names,
                letters,
                dofs,
                max_NO_atoms_asymmetric_unit,
                max_volume,
                fixed_volume,
                probability_per_spg_per_element,
                probability_per_spg_per_element_per_wyckoff,
                NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element,
                per_element,
                return_original_pyxtal_object,
                do_symmetry_checks,
                denseness_factors_density_per_spg,
                denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type,
                seed,
                is_verbose,
            )

        # Number of atoms placed on each wyckoff site:
        number_of_atoms_per_site = np.zeros(len(names))

        chosen_elements = []
        chosen_numbers = []  # list of multiplicities of the chosen wychoff sites
        chosen_wyckoff_indices = []

        unique_elements_counter = 0  # Number of already processed unique elements
        current_element = (
            None  # Current element that is repeated X times on wyckoff positions
        )
        current_picked_repetition = None  # How often to repeat the current element?
        current_repetition_counter = 0  # How often has the element already been placed?

        while True:
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

                if (
                    current_repetition_counter == current_picked_repetition
                    or current_picked_repetition is None
                ):
                    # Done setting the previous unique element, continue with the next

                    # Pick a new element which has not been chosen before
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

                    # Decide how often to repeat the element
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

                probability_per_wyckoff = (
                    probability_per_spg_per_element_per_wyckoff[group_object.number][
                        current_element
                    ]
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

                # If the wyckoff site does not have a DOF and is already occupied:
                if (
                    dofs[chosen_index] == 0
                    and int(number_of_atoms_per_site[chosen_index]) == 1
                ):
                    counter_collisions += 1

                    # need to reset the changes that were made above, since this didn't work out:
                    chosen_elements.pop()
                    current_repetition_counter -= 1

                    continue  # try placement again

                number_of_atoms_per_site[chosen_index] += 1

                chosen_numbers.append(multiplicities[chosen_index])
                chosen_wyckoff_indices.append(chosen_index)

                break

        if len(chosen_numbers) > max_NO_atoms_asymmetric_unit:
            if is_verbose:
                print(
                    f"Too many total number of set wyckoff sites for spg {group_object.number}, regenerating..."
                )
            continue  # but do not increase tries_counter, this is totally fine and expected!

        if is_verbose:
            print(f"Number of chosen wyckoff sites: {len(chosen_numbers)}")

        if (
            denseness_factors_conditional_sampler_seeds_per_spg is None
            and denseness_factors_density_per_spg is None
        ):
            factor = np.random.uniform(0.7, 2.13)
        elif denseness_factors_conditional_sampler_seeds_per_spg is None:
            while True:
                factor = denseness_factors_density_per_spg[
                    group_object.number
                ].resample(1)[0, 0]

                if factor > 0.0:
                    break
        else:
            factor = None  # Will pass factor=None to create_pyxtal_object function, which will result in the
            # sampling of the denseness factor using denseness_factors_conditional_sampler_seeds_per_spg

        try:
            my_crystal = create_pyxtal_object(
                group_object=group_object,
                factor=factor,
                species=chosen_elements,
                chosen_wyckoff_indices=chosen_wyckoff_indices,
                multiplicities=chosen_numbers,
                max_volume=max_volume,
                scale_volume_min_density=True,
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
            if is_verbose:
                print(
                    f"Volume too high, regenerating. (Number of unique elements: {NO_unique_elements})"
                )

            continue

        try:
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
                    if is_verbose:
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

        if not return_original_pyxtal_object:
            return crystal
        else:
            return crystal, my_crystal


def generate_structures(
    spacegroup_number,
    group_object=None,
    N=1,
    max_NO_atoms_asymmetric_unit=100,
    max_volume=7000,
    fixed_volume=None,
    probability_per_spg_per_element=None,
    probability_per_spg_per_element_per_wyckoff=None,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg_per_element=None,
    per_element=False,  # applies to NO_repetitions_prob_per_spg_per_element and probability_per_spg_per_element_per_wyckoff
    return_original_pyxtal_object=False,
    do_symmetry_checks=True,
    denseness_factors_density_per_spg=None,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    seed=-1,
    is_verbose=False,
):
    """Generate a crystal of the given space group `N` times.

    Args:
        spacegroup_number (int): Space group number to use for generation, from 1 to 230.
        group_object (pyxtal.symmetry.Group, optional): Pass in a group object of the given spg to speed up the generation. Defaults to None.
        N (int, optional): How many crystals to generate. Defaults to 1.
        max_NO_atoms_asymmetric_unit (int, optional): Maximum number of atoms in the asymmetric unit of the crystal. Defaults to 100.
        max_volume (float, optional): Maximum volume of the conventional unit cell of the crystal. Defaults to 7000.
        fixed_volume (float, optional): If not None, the volume of all crystals will be set to `fixed_volume`. Defaults to None.
        probability_per_spg_per_element (optional): Indexed using [spg][element]. Returns the probability that an element of the given space group
            occurrs in a crystal. Defaults to None.
        probability_per_spg_per_element_per_wyckoff (optional): Indexed using [spg][element][wyckoff_index]. Returns the probability
            for the given spg that the given element is placed on the given wyckoff position. Defaults to None.
        NO_unique_elements_prob_per_spg (optional): Indexed using [spg]. For the given spg, return a list of probabilities
            that the number of unique elements in a crystal is equal to the list index + 1. Defaults to None.
        NO_repetitions_prob_per_spg_per_element (optional): Indexed using [spg][element]. For the given spg and element, return a list of probabilities
            that the number of appearances of the element on wyckoff sites in a crystal is equal to the list index + 1. Defaults to None.
        per_element (bool, optional): If this setting is True, NO_repetitions_prob_per_spg_per_element and probability_per_spg_per_element_per_wyckoff
            are indexed using the element. If False, they are independent of the element. Defaults to False.
        return_original_pyxtal_object (bool, optional): Whether or not to additionally to the pymatgen crystal, also return the pyxtal structure. Defaults to False.
        do_symmetry_checks (bool, optional): Whether or not to check the spg of the resulting crystals using `spglib`. Defaults to True.
        denseness_factors_density_per_spg (dict of scipy.stats.kde.gaussian_kde, optional): Dictionary of KDEs to generate the denseness factor for each spg. Defaults to None.
        denseness_factors_conditional_sampler_seeds_per_spg (dict of tuple, optional): dictionary containing tuple for each spg:
        lattice_paras_density_per_lattice_type (dict of scipy.stats.kde.gaussian_kde): Dictionary yielding the KDE for each lattice type.
        seed (int, optional): Seed to initialize the random generators. If -1, no seed is used. Defaults to -1.
        is_verbose (bool, optional): Whether or not to print additional info. Defaults to False.

    Returns:
        list of pymatgen.core.structure: Generated structures. If return_original_pyxtal_object is True, this is of type list of (pymatgen.core.structure, pyxtal.pyxtal).
    """

    if group_object is None:
        group = Group(spacegroup_number, dim=3)
    else:
        group = group_object

    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)
    letters = [x.letter for x in group]

    result = [
        __generate_structure(
            None,
            group_object=group,
            multiplicities=multiplicities,
            names=names,
            letters=letters,
            dofs=dofs,
            max_NO_atoms_asymmetric_unit=max_NO_atoms_asymmetric_unit,
            max_volume=max_volume,
            fixed_volume=fixed_volume,
            probability_per_spg_per_element=probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
            NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
            per_element=per_element,
            return_original_pyxtal_object=return_original_pyxtal_object,
            do_symmetry_checks=do_symmetry_checks,
            denseness_factors_density_per_spg=denseness_factors_density_per_spg,
            denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
            lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
            seed=seed,
            is_verbose=is_verbose,
        )
        for i in range(0, N)
    ]

    return result


if __name__ == "__main__":
    (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg,
        denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type,
        per_element,
        represented_spgs,
        probability_per_spg,
    ) = load_dataset_info(load_public_statistics_only=True)

    structures = generate_structures(
        125,
        N=1,
        probability_per_spg_per_element=probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
        NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
        denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
    )
