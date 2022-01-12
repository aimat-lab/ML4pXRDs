import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os

raw_files = glob("RUFF_data/XY_RAW/*.txt")
processed_files = []

processed_xys = []
raw_xys = []

counter = 0

for raw_file in raw_files:

    raw_filename = os.path.basename(raw_file)

    processed_file = os.path.join("RUFF_data/XY_Processed/", "__".join(raw_filename.replace("RAW", "Processed").split("__")[:-1]) + "*.txt")
    processed_file = glob(processed_file)

    if not len(processed_file) > 0:
        continue

    counter += 1

    processed_file = processed_file[0]

    processed_files.append(processed_file)

    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")
    raw_xys.append(raw_xy)

    processed_xy = np.genfromtxt(processed_file, dtype=float, delimiter=",", comments="#")
    processed_xys.append(processed_xy)

    difference = raw_xy[:,1] - processed_xy[:,1]

    plt.plot(raw_xy[:,0], difference)
    plt.plot(raw_xy[:,0], processed_xy[:,1])
    plt.plot(raw_xy[:,0], raw_xy[:,1])

    plt.show()

print(f"{counter} matching files found.")