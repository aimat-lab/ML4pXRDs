import numpy as np
from pyxtal.database.element import Element
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.crystal import atom_site


def generate_pyxtal_object(
    group_object,
    factor,
    species,
    chosen_wyckoff_indices,
    multiplicities,
    max_volume,
    scale_volume_min_density=True,
):
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

        # TODO: Check if index is correct
        wyckoff = group_object.get_wyckoff_position(wyckoff_index)

        random_coord = pyxtal_object.lattice.generate_point()
        projected_coord = wyckoff.project(random_coord, pyxtal_object.lattice.matrix)

        new_atom_site = atom_site(wp=wyckoff, coordinate=projected_coord, specie=specie)
        pyxtal_object.atom_sites.append(new_atom_site)

    pyxtal_object.valid = True

    return pyxtal_object
