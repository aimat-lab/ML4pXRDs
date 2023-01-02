import re
from pyxtal.database.element import Element
import numpy as np


def get_denseness_factor(structure):
    """Obtain the denseness factor (unit cell volume divided by sum of atomic volumes)
    of the given pymatgen structure.

    Args:
        structure (pymatgen.core.structure): Structure object

    Returns:
        float|None: Denseness factor. Returns None if an error occurs.
    """

    try:

        actual_volume = structure.volume

        calculated_volume = 0
        for atom in structure:

            splitted_sites = [item.strip() for item in atom.species_string.split(",")]

            for splitted_site in splitted_sites:

                splitted = splitted_site.split(":")

                specie = re.sub(r"\d*[,.]?\d*\+?$", "", splitted[0])
                specie = re.sub(r"\d*[,.]?\d*\-?$", "", specie)

                if (
                    "-" in specie
                    or "+" in specie
                    or ":" in specie
                    or "," in specie
                    or "." in specie
                ):
                    raise Exception(
                        "Something went wrong in get_denseness_factor_ran function."
                    )

                if len(splitted) > 1:
                    occupancy = float(splitted[1])
                else:
                    occupancy = 1.0

                r = (Element(specie).covalent_radius + Element(specie).vdw_radius) / 2

                calculated_volume += 4 / 3 * np.pi * r**3 * occupancy

        return actual_volume / calculated_volume, calculated_volume

    except Exception as ex:

        print("Not able to get denseness factor:")
        print(ex)

        # For D and Am exceptions are OK

        return None
