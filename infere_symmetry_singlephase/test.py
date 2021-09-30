from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def print_pattern(cif_file):

    parser = CifParser(cif_file)

    structures = parser.get_structures()
    structure = structures[0]
    calculator = XRDCalculator(wavelength=1.2)

    pattern = calculator.get_pattern(structure, scaled=True)

    for i in range(0, len(pattern.x)):
        print(
            "{} {} {}".format(
                pattern.hkls[i][0]["hkl"],
                pattern.hkls[i][0]["multiplicity"],
                pattern.x[i],
                pattern.y[i],
            )
        )


print_pattern("/home/henrik/Dokumente/ICSD_cleaned/ICSD_1529.cif")

print("######################################")

print_pattern("/home/henrik/Dokumente/ICSD_cleaned/ICSD_246372.cif")
