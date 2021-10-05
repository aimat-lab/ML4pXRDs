from pyxtal import pyxtal
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np

my_crystal = pyxtal()
my_crystal.from_random(3, 99, ["Ba", "Ti", "O"], [1, 1, 3])

print(my_crystal)

crystal = my_crystal.to_pymatgen()

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
