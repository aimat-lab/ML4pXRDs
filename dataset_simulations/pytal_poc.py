from pyxtal import pyxtal
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np
import random
from pymatgen.core.periodic_table import Element

max_NO_atoms = 50
max_volume = 30


def generate_structures(spacegroup_number, N):

    elements = list(Element.__members__.keys())

    crystals = []

    for i in range(0, N):

        NO_atoms = random.randint(1, max_NO_atoms)

        chosen_elements = np.random.choice(elements, size=NO_atoms)

        chosen_elements_unique = np.unique(chosen_elements).tolist()
        chosen_elements_counts = []

        for element in chosen_elements_unique:
            chosen_elements_counts.append(np.sum(np.array(chosen_elements) == element))

        my_crystal = pyxtal()

        try:
            my_crystal.from_random(
                dim=3,
                group=spacegroup_number,
                species=chosen_elements_unique,
                numIons=chosen_elements_counts,
            )

            # my_crystal.show()
        except:
            pass
            continue

        if not my_crystal.valid:
            continue

        crystal = my_crystal.to_pymatgen()

        crystals.append(crystal)

    return crystals


if __name__ == "__main__":
    generate_structures(3, 100)

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
