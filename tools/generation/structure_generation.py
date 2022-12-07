import numpy as np
from pyxtal.database.element import Element
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.crystal import atom_site
import time


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


def generate_pyxtal_object(
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
        except Exception as ex:
            print(ex)

            labels.append(None)
            reference_crystals.append(None)
            randomized_crystals.append(None)

            continue

        labels.append(pyxtal_object.group.number)

        reference_crystal = pyxtal_object.to_pymatgen()
        reference_crystals.append(reference_crystal)

        if randomize_lattice:  # regenerate the lattice
            pyxtal_object.lattice = Lattice(
                pyxtal_object.group.lattice_type, pyxtal_object.lattice.volume
            )

            if lattice_paras_density_per_lattice_type is not None:
                paras = sample_lattice_paras(
                    pyxtal_object.lattice.volume,
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
