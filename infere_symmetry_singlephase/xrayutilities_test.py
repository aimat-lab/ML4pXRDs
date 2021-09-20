import xrayutilities as xu
from xrayutilities.materials.cif import CIFFile
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator

crystal = xu.materials.Crystal.fromCIF("test.cif")

powder = xu.simpack.Powder(crystal, 1)

powder_model = xu.simpack.PowderModel(powder, I0=100)

xs = np.arange(5,120,0.01)
diffractogram = powder_model.simulate(xs)

# Default settings:
print(powder_model.pdiff[0].settings)
# Further information on the default settings: https://nvlpubs.nist.gov/nistpubs/jres/120/jres.120.014.c.py

powder_model.plot(xs)
plt.show()

exit()

# do the same with pymatgen:

parser = CifParser("test.cif")
structure = parser.get_structures()[0]

calculator = XRDCalculator()
pattern = calculator.get_pattern(structure, scaled=True)

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

plt.figure()
plt.plot(xs, ys)
plt.title("pymatgen")

plt.show()