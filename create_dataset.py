from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import multiprocessing
from itertools import repeat
import os
import time
import warnings
import json

def process_cif(input):

    #warnings.filterwarnings(action='error')
    warnings.filterwarnings(action='ignore')

    (cif, number_of_points) = input

    cif_number = int(os.path.basename(cif).split(".")[0])

    # NaCl for testing:
    #test_cif_file = r"C:\Users\hscho\BigFiles\OCD database\cif\1\00\00\1000041.cif"

    try:

        parser = CifParser(cif)
        structures = parser.get_structures()

    except Exception as error:

        print("##### Error encountered processing cif {}, skipping structure:".format(cif))
        print(error)
        return None

    if len(structures) != 1:

        print("##### Cif {}: Found no or more than one structure".format(cif))
        return None

    else:

        structure = structures[0]

    calculator = XRDCalculator()

    try:

        pattern = calculator.get_pattern(structure, scaled=True)
    
    except Exception as error:

        print("##### Error encountered calculating XRD of cif {}, skipping structure:".format(cif))
        print(error)
        return None

    xs = np.linspace(0, 90, number_of_points)

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

    # determine space group:

    try:
        analyzer = SpacegroupAnalyzer(structure)

        group_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()
        space_group_symbol = analyzer.get_space_group_symbol()[0]

    except Exception as error:

        print("##### Error encountered analyzing crystal structure, skipping structure:")
        print(error)
        return None

    crystal_system_letter = ""

    if crystal_system == "anortic" or crystal_system == "triclinic":
        crystal_system_letter = "a"
    elif crystal_system == "monoclinic":
        crystal_system_letter = "m"
    elif crystal_system == "orthorhombic":
        crystal_system_letter = "o"
    elif crystal_system == "tetragonal":
        crystal_system_letter = "t"
    elif crystal_system == "cubic":
        crystal_system_letter = "c"
    elif crystal_system == "hexagonal" or crystal_system == "trigonal":
        crystal_system_letter = "h"
    else:
        print("Crystal system {} not recognized. Skipping structure.".format(crystal_system))
        return None

    if space_group_symbol in "ABC":
        space_group_symbol = "S" # side centered

    bravais = crystal_system_letter + space_group_symbol

    if bravais not in ["aP", "mP", "mS", "oP", "oS", "oI", "oF", "tP", "tI", "cP", "cI", "cF", "hP", "hR"]:
        print("Bravais lattice {} not recognized. Skipping structure.".format(bravais))
        return None

    #plt.plot(xs, ys)
    #plt.show()

    return [cif_number, *ys, bravais, group_number]

def track_job(job, update_interval=5):
    while job._number_left > 0:
        print("Tasks remaining = {0}".format(
        job._number_left * job._chunksize))
        time.sleep(update_interval)

if __name__ == "__main__":

    all_cifs = glob(r"C:\Users\hscho\BigFiles\OCD database\cif\**\*.cif", recursive=True)
    print("{} cif files found".format(len(all_cifs)))

    #json.dump(all_cifs, open("filenames.json", "w"))
    all_cifs = json.load(open("filenames_bak.json","r"))

    # number of points to use for angle range:
    number_of_points = 181

    pool = multiprocessing.Pool(processes=8)

    start = time.time()

    handle = pool.map_async(process_cif, zip(all_cifs[400000:], repeat(number_of_points)))
    track_job(handle)

    result = handle.get()

    end = time.time()

    print("##### Result obtained after {} s".format(end-start))

    result = [x for x in result if x is not None]
    result = np.array(result)

    np.savetxt("dataset_8.csv", result, delimiter=" ", fmt="%s")