from pymatgen.io.cif import CifParser
from pyxtal import pyxtal

parser = CifParser("test.cif")
crystals = parser.get_structures()
crystal = crystals[0]

test = pyxtal()

test.from_seed(crystal)

back = test.to_pymatgen()

print()
