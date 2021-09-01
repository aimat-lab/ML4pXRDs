from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import multiprocessing
import os
import time
import warnings
import json
import math
import sys


def process_cif(cif):

    # warnings.filterwarnings(action='error')
    # warnings.filterwarnings(action='ignore')

    cif_number = int(os.path.basename(cif).split(".")[0].split("_")[-1])

    # NaCl for testing:
    # test_cif_file = r"C:\Users\hscho\BigFiles\OCD database\cif\1\00\00\1000041.cif"

    try:

        parser = CifParser(cif)
        structures = parser.get_structures()

    except Exception as error:

        print(
            "##### Error encountered processing cif {}, skipping structure:".format(cif)
        )
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

        print(
            "##### Error encountered calculating XRD of cif {}, skipping structure:".format(
                cif
            )
        )
        print(error)
        return None

    # xs = np.linspace(0, 90, number_of_points)
    xs = np.linspace(0, 90, 9001)  # use 0.01 steps

    # print(list(xs))

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

    # determine space group:

    try:

        analyzer = SpacegroupAnalyzer(structure)

        group_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()
        space_group_symbol = analyzer.get_space_group_symbol()[0]

    except Exception as error:

        print(
            "##### Error encountered analyzing crystal structure, skipping structure:"
        )
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
        print(
            "Crystal system {} not recognized. Skipping structure.".format(
                crystal_system
            )
        )
        return None

    if space_group_symbol in "ABC":
        space_group_symbol = "S"  # side centered

    bravais = crystal_system_letter + space_group_symbol

    if bravais not in [
        "aP",
        "mP",
        "mS",
        "oP",
        "oS",
        "oI",
        "oF",
        "tP",
        "tI",
        "cP",
        "cI",
        "cF",
        "hP",
        "hR",
    ]:
        print("Bravais lattice {} not recognized. Skipping structure.".format(bravais))
        return None

    # print(pattern.x[0:100])
    # print(pattern.y[0:100])

    # plt.plot(xs, ys)
    # plt.show()

    return [cif_number, *ys, bravais, group_number]


def track_job(job, update_interval=5):
    while job._number_left > 0:
        print(
            "Tasks remaining in this batch of 1000: {0}".format(
                job._number_left * job._chunksize
            )
        )
        time.sleep(update_interval)


if __name__ == "__main__":

    output_dir = "databases/icsd/"

    if not os.path.exists(os.path.join(output_dir, "filenames.json")):

        # all_cifs = glob(r"C:\Users\hscho\BigFiles\OCD database\cif\**\*.cif", recursive=True)
        all_cifs = glob(
            r"/mnt/c/Users/legion/BigFiles/ICSD_cleaned/*.cif", recursive=True
        )
        json.dump(all_cifs, open(os.path.join(output_dir, "filenames.json"), "w"))

    else:

        all_cifs = json.load(open(os.path.join(output_dir, "filenames.json"), "r"))

    # all_cifs = ["/mnt/c/Users/legion/BigFiles/ICSD_cleaned/ICSD_10924.cif"]

    print("{} cif files found".format(len(all_cifs)))

    # put 10000 entries into one file:
    batch_size = 1000

    for i in range(0, math.ceil(len(all_cifs) / batch_size)):

        if os.path.exists(os.path.join(output_dir, "dataset_" + str(i) + ".csv")):
            continue

        if ((i + 1) * batch_size) < len(all_cifs):
            end_index = (i + 1) * batch_size
            cifs = all_cifs[i * batch_size : end_index]
        else:
            cifs = all_cifs[i * batch_size :]
            end_index = len(all_cifs)

        pool = multiprocessing.Pool(processes=8)

        start = time.time()

        handle = pool.map_async(process_cif, cifs)

        track_job(handle)

        result = handle.get()

        end = time.time()

        result = [x for x in result if x is not None]
        result = np.array(result)

        np.savetxt(
            os.path.join(output_dir, "dataset_" + str(i) + ".csv"),
            result,
            delimiter=" ",
            fmt="%s",
        )

        print(
            "##### Calculated from cif {} to {} (total: {}) in {} s".format(
                i * batch_size, end_index, len(all_cifs), end - start
            )
        )

