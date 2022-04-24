import numpy as np
from pyxtal.database.element import Element
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.crystal import atom_site


def rejection_sampler(p, xbounds, pmax):
    while True:
        x = (np.random.rand(1) * (xbounds[1] - xbounds[0]) + xbounds[0])[0]
        y = (np.random.rand(1) * pmax)[0]
        if y <= p(x):
            return x


def sample_denseness_factor(volume, seed):

    conditional_density = seed[0]
    min_denseness_factors = seed[1]
    max_denseness_factors = seed[2]

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


def generate_pyxtal_object(
    group_object,
    factor,
    species,
    chosen_wyckoff_indices,
    multiplicities,
    max_volume,
    scale_volume_min_density=True,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
):
    """Used to generate a pyxtal object using the given parameters.

    Parameters
    ----------
    group_object: Group
        Symmetry operations of this group will be used.
    factor: float
        Factor to scale the volume with.
    species: list[str]
        Species to set on wyckoff sites.
    chosen_wyckoff_indices: list[int]
        Wyckoff sites to use for each specie.
    multiplicities: list[int]
        Multiplicity of each of the chosen Wyckoff sites.
    max_volume: float
        Maximum value of lattice volume. Returns False if volume > max_volumne.
    scale_volume_min_density: True
        Whether or not to scale volume so it matches the minimum density.

    Returns
    -------
    False
        If the volume was too high.
    pyxtal
        Pyxtal object.

    """

    # 1) calculate the sum of covalent volumes
    # 2) calculate actual volume of crystal (multiply by factor)
    # 3) generate a lattice with the given volume
    # 4) create the pyxtal object with random coordinates

    ### 1)

    volume = 0
    for numIon, specie in zip(multiplicities, species):
        # r = random.uniform(
        #    Element(specie).covalent_radius, Element(specie).vdw_radius
        # )
        r = (Element(specie).covalent_radius + Element(specie).vdw_radius) / 2
        volume += numIon * 4 / 3 * np.pi * r**3

    if factor is not None:
        volume *= factor
    else:
        factor = sample_denseness_factor(
            volume,
            denseness_factors_conditional_sampler_seeds_per_spg[group_object.number],
        )
        volume *= factor

    if scale_volume_min_density:
        min_density = 0.75
        # make sure the volume is not too small
        if volume / sum(multiplicities) < min_density:
            volume = sum(multiplicities) * min_density
            print("Volume has been scaled to match minimum density.")

    if volume > max_volume:
        return False

    pyxtal_object = pyxtal(molecular=False)

    pyxtal_object.lattice = Lattice(group_object.lattice_type, volume)

    for specie, wyckoff_index in zip(species, chosen_wyckoff_indices):

        wyckoff = group_object.get_wyckoff_position(wyckoff_index)

        random_coord = pyxtal_object.lattice.generate_point()
        projected_coord = wyckoff.project(random_coord, pyxtal_object.lattice.matrix)

        new_atom_site = atom_site(wp=wyckoff, coordinate=projected_coord, specie=specie)
        pyxtal_object.atom_sites.append(new_atom_site)

    pyxtal_object.valid = True

    return pyxtal_object


def randomize(crystals, randomize_coordinates=True, randomize_lattice=False):

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
