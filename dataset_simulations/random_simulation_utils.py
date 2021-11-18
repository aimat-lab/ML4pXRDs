from pyxtal import pyxtal
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np
import random
from pymatgen.core.periodic_table import Element
from pyxtal.symmetry import Group
import time
from ase.visualize import view
from pymatgen.vis.structure_vtk import StructureVis

max_NO_atoms = 10
# 10 atoms per unit cell should probably be already enough.

# max_volume = 30

# from pyxtal element.py:
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


def generate_structures(spacegroup_number, N):

    group = Group(spacegroup_number, dim=3)

    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)

    # all_elements = list(Element.__members__.keys())

    output_crystals = []

    for i in range(0, N):

        print(f"Crystal {i}")
        # TODO: maybe use slightly random volume factors later

        number_of_atoms = np.zeros(len(names))

        NO_atoms = random.randint(1, max_NO_atoms)

        chosen_elements = []
        chosen_numbers = []
        chosen_wyckoff_positions = []

        counter_collisions = 0

        # TODO: Always choose the general wyckoff position

        for i in range(0, NO_atoms):
            while True:

                if counter_collisions > 100:
                    print("More than 100 collisions.")
                    break

                chosen_index = random.randint(0, len(number_of_atoms) - 1)

                if dofs[chosen_index] == 0 and int(number_of_atoms[chosen_index]) == 1:
                    counter_collisions += 1
                    print(f"{counter_collisions} collisions.")
                    continue

                number_of_atoms[chosen_index] += 1

                chosen_elements.append(random.choice(all_elements))
                chosen_numbers.append(multiplicities[chosen_index])
                chosen_wyckoff_positions.append([names[chosen_index]])

                break

        # TODO: Maybe bring unique entries of chosen_elements together to form one?

        """
        chosen_elements = np.random.choice(elements, size=NO_atoms)

        chosen_elements_unique = np.unique(chosen_elements).tolist()
        chosen_elements_counts = []

        for element in chosen_elements_unique:
            chosen_elements_counts.append(np.sum(np.array(chosen_elements) == element))
        """

        # TODO: Do proper error handling

        my_crystal = pyxtal()

        my_crystal.from_random(
            dim=3,
            group=spacegroup_number,
            species=chosen_elements,
            numIons=chosen_numbers,
            # sites=chosen_wyckoff_positions,
        )

        if not my_crystal.valid:
            continue

        # TODO: If it fails, retry with next, so N is actually correct

        plt.show()

        crystal = my_crystal.to_pymatgen()

        # vis = StructureVis()
        # vis.set_structure(crystal)
        # vis.show()

        output_crystals.append(crystal)

    print(f"Generated {len(output_crystals)} of {N} requested crystals")

    return output_crystals


if __name__ == "__main__":
    start = time.time()
    generate_structures(125, 20)
    stop = time.time()

    print(f"Took {stop-start} s")
