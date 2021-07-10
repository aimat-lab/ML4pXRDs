from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import multiprocessing

all_cifs = glob(r"C:\Users\hscho\BigFiles\OCD database\cif\**\*.cif", recursive=True)
print("{} cif files found".format(len(all_cifs)))

# way down: https://scicomp.stackexchange.com/questions/19586/parallelizing-a-for-loop-in-python

def process_cif(cif):

    # NaCl for testing:
    #test_cif_file = r"C:\Users\hscho\BigFiles\OCD database\cif\1\00\00\1000041.cif"

    print(cif)

    try:
        parser = CifParser(cif)
        structures = parser.get_structures()

    except Exception as error:
        print("Error encountered processing cif {}, skipping structure:".format(cif))
        print(error)
        return None

    if len(structures) != 1:
        print("Cif {}: Found more than one structure".format(cif))
        return None
    else:
        structure = structures[0]

    calculator = XRDCalculator()

    pattern = calculator.get_pattern(structure, scaled=False)

    xs = np.linspace(0,90,181)

    def smeared_peaks(xs):

        output = np.zeros(len(xs))

        for x, y in zip(pattern.x, pattern.y):

            sigma = 0.2
            mean = x
            
            ys = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/(2*sigma**2)*(xs-mean)**2)

            delta_x = xs[1] - xs[0]
            volume = delta_x*np.sum(ys)

            ys = y * ys / volume

            output += ys

        return output

    ys = smeared_peaks(xs)