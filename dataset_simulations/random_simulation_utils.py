from pyxtal import pyxtal
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np
import random
from pymatgen.core.periodic_table import Element
from pyxtal.symmetry import Group

max_NO_atoms = 5
max_volume = 30


def generate_structures(spacegroup_number, N):

    group = Group(spacegroup_number, dim=3)

    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)

    all_elements = list(Element.__members__.keys())

    output_crystals = []

    for i in range(0, N):

        # TODO: maybe use slightly random volume factors later

        number_of_atoms = np.zeros(len(names))

        NO_atoms = random.randint(1, max_NO_atoms)

        chosen_elements = []
        chosen_numbers = []

        counter_collisions = 0

        for i in range(0, NO_atoms):
            while True:

                if counter_collisions > 100:
                    print("More than 100 collisions.")
                    break

                chosen_index = random.randint(0, len(number_of_atoms) - 1)

                # TODO: Does this even happen?
                if dofs[chosen_index] == 0 and int(number_of_atoms[chosen_index]) == 1:
                    counter_collisions += 1
                    print(f"{counter_collisions} collisions.")
                    continue

                number_of_atoms[chosen_index] += 1

                chosen_elements.append(random.choice(all_elements))
                chosen_numbers.append(multiplicities[chosen_index])

                break

        # TODO: Maybe bring unique entries of chosen_elements together to form one?

        """
        chosen_elements = np.random.choice(elements, size=NO_atoms)

        chosen_elements_unique = np.unique(chosen_elements).tolist()
        chosen_elements_counts = []

        for element in chosen_elements_unique:
            chosen_elements_counts.append(np.sum(np.array(chosen_elements) == element))
        """

        my_crystal = pyxtal()

        my_crystal.from_random(
            dim=3,
            group=spacegroup_number,
            species=chosen_elements,
            numIons=chosen_numbers,
        )

        if not my_crystal.valid:
            continue

        my_crystal.show()

        plt.show()

        crystal = my_crystal.to_pymatgen()

        output_crystals.append(crystal)

    return output_crystals


if __name__ == "__main__":
    generate_structures(125, 100)

"""
calculator = XRDCalculator()
pattern = calculator.get_pattern(crystal, scaled=True)

xs = np.linspace(0, 90, 1001)


def smeared_peaks(xs):

    output = np.zeros(len(xs))

    for x, y in zip(pattern.x, pattern.y):

        sigma = 0.1
        mean = x

        ys = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - mean) ** 2)
        )

        delta_x = xs[1] - xs[0]
        volume = delta_x * np.sum(ys)

        ys = y * ys / volume

        output += ys

    return output


ys = smeared_peaks(xs)

plt.plot(xs, ys)
plt.show()
"""
