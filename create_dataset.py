from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
import matplotlib.pyplot as plt

test_cif_file = r"C:\Users\hscho\BigFiles\OCD database\cif\1\00\00\1000041.cif"

parser = CifParser(test_cif_file)

structure = parser.get_structures()[0]

calculator = XRDCalculator()

pattern = calculator.get_pattern(structure, scaled=False)

#xs = np.linspace(11,89,157)
#ys = [pattern.get_interpolated_value(x) for x in xs]

#print(pattern.x)
#plt.plot(pattern.x, pattern.y)
#plt.show()

# this actually looks better:
calculator.show_plot(structure)

# TODO: what went wrong?