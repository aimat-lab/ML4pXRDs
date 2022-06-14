import tensorflow.keras as keras
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import generate_background_noise_utils
from datetime import datetime
from glob import glob
import pickle

select_which_to_use_for_testing = False
use_only_selected = True

# to_test = "18-05-2022_09-45-42_UNetPP"
to_test = "06-06-2022_22-15-44_UNetPP"

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

model = keras.models.load_model("../unet/" + to_test + "/final")

if not use_only_selected:
    raw_files = glob("../RRUFF_data/XY_RAW/*.txt")
else:
    with open("to_test_on.pickle", "rb") as file:
        raw_files = pickle.load(file)

if select_which_to_use_for_testing:
    raw_files_keep = []

for i, raw_file in enumerate(raw_files):
    # for i in range(1606, len(raw_files)):
    #    raw_file = raw_files[i]

    if not select_which_to_use_for_testing:
        plt.figure()
    else:
        plt.clf()

    raw_filename = os.path.basename(raw_file)
    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")

    if len(raw_xy) == 0:
        print("Skipped empty pattern.")
        continue

    # Model:
    pattern_x = np.arange(0, 90.24, 0.02)
    start_x = pattern_x[0]
    end_x = pattern_x[-1]
    N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

    x_test = raw_xy[:, 0]
    y_test = raw_xy[:, 1]

    # Remove nans:
    not_nan_indices = np.where(~np.isnan(x_test))[0]
    x_test = x_test[not_nan_indices]
    y_test = y_test[not_nan_indices]
    not_nan_indices = np.where(~np.isnan(y_test))[0]
    x_test = x_test[not_nan_indices]
    y_test = y_test[not_nan_indices]

    if not min(abs((x_test[0] % 0.02) - 0.02), abs(x_test[0] % 0.02)) < 0.0000001:
        print(f"Skipping pattern due to different x-steps.")
        continue

    if not np.all(np.diff(x_test) >= 0):  # not ascending
        print("Skipped pattern, inconsistent x axis.")
        continue

    dx = x_test[1] - x_test[0]
    if abs(dx - 0.01) < 0.0000001:  # allow some tollerance
        x_test = x_test[::2]
        y_test = y_test[::2]
    elif abs(dx - 0.02) < 0.0000001:
        pass
    else:
        print(f"Skipping pattern with dx={dx}.")
        continue

    y_test = np.array(y_test)
    y_test -= min(y_test)
    y_test = y_test / np.max(y_test)

    if x_test[0] > 0.0:
        to_add = np.arange(0.0, x_test[0], 0.02)
        x_test = np.concatenate((to_add, x_test), axis=0)
        y_test = np.concatenate((np.repeat([y_test[0]], len(to_add)), y_test), axis=0)

    if x_test[-1] < 90.24:
        to_add = np.arange(x_test[-1] + 0.02, 90.24, 0.02)
        x_test = np.concatenate((x_test, to_add), axis=0)
        y_test = np.concatenate((y_test, np.repeat([y_test[-1]], len(to_add))), axis=0)

    # print(x_test[-300:])

    if len(x_test) != 4512:
        print("Skipping pattern due to wrong dimensions of xs.")
        continue

    if not select_which_to_use_for_testing:
        predictions = model.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

    plt.xlabel(r"$2 \theta$")
    plt.ylabel("Intensity")

    if not select_which_to_use_for_testing:
        plt.plot(pattern_x, predictions[0, :, 0], label="Prediction")

    plt.plot(
        pattern_x,
        y_test,
        label="Input pattern",
    )

    if not select_which_to_use_for_testing:
        plt.plot(
            pattern_x,
            y_test - predictions[0, :, 0],
            label="Prediced background and noise",
            linestyle="dotted",
        )

    plt.legend()

    if select_which_to_use_for_testing:

        plt.pause(0.1)

        result = input(f"{i+1}/{len(raw_files)} Keep? (y/n)")

        if result == "y" or result == "Y":
            raw_files_keep.append(raw_file)

    else:

        plt.show()

if select_which_to_use_for_testing:
    with open("to_test_on.pickle", "wb") as file:
        pickle.dump(raw_files_keep, file)
