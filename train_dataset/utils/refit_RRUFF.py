import numpy as np
from glob import glob
import os
import random
from train_dataset.utils.rruff_helpers import *
import pickle

if True:
    with open("to_test_on.pickle", "rb") as file:
        raw_files = pickle.load(file)
else:
    raw_files = glob("../RRUFF_data/XY_RAW/*.txt")

processed_files = []
dif_files = []

processed_xys = []

raw_xys = []

angles = []
intensities = []

counter_processed = 0
counter_dif = 0

# random.shuffle(raw_files)

parameter_results = []

for i, raw_file in enumerate(raw_files):

    print(f"{(i+1)/len(raw_files)*100:.2f}% processed")

    raw_filename = os.path.basename(raw_file)
    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")
    raw_xys.append(raw_xy)

    processed_file = os.path.join(
        "../RRUFF_data/XY_Processed/",
        "__".join(raw_filename.replace("RAW", "Processed").split("__")[:-1]) + "*.txt",
    )
    processed_file = glob(processed_file)
    if len(processed_file) > 0:
        counter_processed += 1
        found_processed_file = True
        processed_file = processed_file[0]
        processed_files.append(processed_file)
    else:
        found_processed_file = False
        processed_files.append(None)
        pass

    dif_file = os.path.join(
        "../RRUFF_data/DIF/",
        "__".join(raw_filename.split("__")[:-2]) + "__DIF_File__*.txt",
    )
    dif_file = glob(dif_file)

    data = None
    if len(dif_file) > 0:
        counter_dif += 1
        dif_file = dif_file[0]
        dif_files.append(dif_file)

        data, wavelength, spg_number = dif_parser(dif_file)

    else:

        dif_files.append(None)

    if found_processed_file:

        processed_xy = np.genfromtxt(
            processed_file, dtype=float, delimiter=",", comments="#"
        )
        processed_xys.append(processed_xy)

    if True and data is not None:  # and found_processed_file:  # if nothing went wrong

        angles.append(data[:, 0])
        intensities.append(data[:, 1])

        # plt.plot(processed_xy[:,0][::10], processed_xy[:,1][::10], label="Processed")
        # plt.plot(raw_xy[:,0][::10], raw_xy[:,1][::10], label="Raw")
        # plt.plot(raw_xy[:,0][::10], raw_xy[:,1][::10] - processed_xy[:,1][::10], label="BG")
        # plt.legend()
        # plt.show()

        fit_parameters = fit_diffractogram(
            raw_xys[-1][:, 0],
            raw_xys[-1][:, 1] / np.max(raw_xys[-1][:, 1]),
            # processed_xy[:,0],
            # processed_xy[:,1] / np.max(processed_xy[:,1]),
            angles[-1],
            intensities[-1] / np.max(intensities[-1]),
        )

        parameter_results.append((raw_file, fit_parameters))

    else:

        angles.append(None)
        intensities.append(None)

    pass

    # This alone is not enough, unfortunately:
    # difference = raw_xy[:,1] - processed_xy[:,1]
    # plt.plot(raw_xy[:,0], difference)
    # plt.plot(raw_xy[:,0], processed_xy[:,1])
    # plt.plot(raw_xy[:,0], raw_xy[:,1])
    # plt.show()

print(f"{counter_processed} processed files found.")
print(f"{counter_dif} dif files found.")

assert len(dif_files) == len(processed_files)

counter_both = 0
for i, dif_file in enumerate(dif_files):
    if dif_file is not None and processed_files[i] is not None:
        counter_both += 1

print(f"{counter_both} files with dif and processed file found.")

with open("rruff_refits.pickle", "wb") as file:
    pickle.dump(parameter_results, file)
