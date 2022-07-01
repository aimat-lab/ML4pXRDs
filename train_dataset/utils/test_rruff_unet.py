from regex import W
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
from scipy.interpolate import CubicSpline

from pyxtal.symmetry import Group
import pickle

from scipy.signal import savgol_filter

from train_dataset.utils.rruff_helpers import *

select_which_to_use_for_testing = False
use_only_selected = True

do_plot = False

unet_model_path = "10-06-2022_13-12-26_UNetPP"
classification_model_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/24-06-2022_10-54-18/"
# classification_model_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/30-05-2022_13-43-21/" # no save file :(
classification_model_path = classification_model_base + "final"
do_unet_preprocessing = False

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

model_unet = keras.models.load_model("../unet/" + unet_model_path + "/final")
model_classification = keras.models.load_model(classification_model_path)

with open(classification_model_base + "spgs.pickle", "rb") as file:
    spgs = pickle.load(file)

if not use_only_selected:
    raw_files = glob("../RRUFF_data/XY_RAW/*.txt")
else:
    with open("to_test_on.pickle", "rb") as file:
        raw_files = pickle.load(file)

if select_which_to_use_for_testing:
    raw_files_keep = []

patterns_counter = 0
correct_counter = 0
correct_top5_counter = 0

for i, raw_file in enumerate(raw_files):

    print(f"{i} of {len(raw_files)}")

    if not select_which_to_use_for_testing and do_plot:
        plt.figure()
    else:
        plt.clf()

    raw_filename = os.path.basename(raw_file)
    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")

    dif_file = os.path.join(
        "../RRUFF_data/DIF/",
        "__".join(raw_filename.split("__")[:-2]) + "__DIF_File__*.txt",
    )
    dif_file = glob(dif_file)

    if len(dif_file) == 0:
        continue

    dif_file = dif_file[0]

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

    # nan_indices = np.where(np.isnan(x_test))[0] # This only happens for the first three indices for some of the patterns; so this is fine
    # if len(nan_indices > 0):
    #    print()

    x_test = x_test[not_nan_indices]
    y_test = y_test[not_nan_indices]

    not_nan_indices = np.where(~np.isnan(y_test))[0]

    # nan_indices = np.where(np.isnan(y_test))[0] # This actually doesn't happen at all
    # if len(nan_indices > 0):
    #    print()

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
        print("")
        pass
    else:
        print(f"Skipping pattern with dx={dx}.")
        continue

    y_test = np.array(y_test)
    y_test -= min(y_test)
    y_test = y_test / np.max(y_test)

    # For now don't use those:
    if x_test[0] > 5.0 or x_test[-1] < 90.0:
        continue

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

    if not select_which_to_use_for_testing and do_unet_preprocessing:
        predictions = model_unet.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

    if do_plot:
        plt.xlabel(r"$2 \theta$")
        plt.ylabel("Intensity")

    if not select_which_to_use_for_testing and do_unet_preprocessing and do_plot:
        plt.plot(pattern_x, predictions[0, :, 0], label="Prediction")

    if do_plot:
        plt.plot(
            pattern_x,
            y_test,
            label="UNet Input pattern",
        )

    if not select_which_to_use_for_testing and do_unet_preprocessing and do_plot:
        plt.plot(
            pattern_x,
            y_test - predictions[0, :, 0],
            label="Prediced background and noise",
            linestyle="dotted",
        )

    data, wavelength, spg_number = dif_parser(dif_file)

    if data is None or wavelength is None or spg_number is None:
        continue

    # if wavelength != 1.541838:
    #    print(wavelength)
    #    exit()

    if do_unet_preprocessing:
        corrected_pattern = predictions[0, :, 0]
    else:
        corrected_pattern = y_test
    # Dimensions of this: np.arange(0, 90.24, 0.02) (pattern_x)

    # Needed for classification:
    classification_pattern_x = np.linspace(5, 90, 8501)

    f = CubicSpline(pattern_x, corrected_pattern)

    y_scaled_up = f(classification_pattern_x)

    # y_scaled_up = savgol_filter(y_scaled_up, 19, 3)
    y_scaled_up -= np.min(y_scaled_up)
    y_scaled_up = y_scaled_up / np.max(y_scaled_up)

    # y_scaled_up[y_scaled_up < 0.002] = 0
    #
    # y_scaled_up = savgol_filter(y_scaled_up, 13, 3)

    if True and do_plot:
        plt.plot(classification_pattern_x, y_scaled_up, label="Scaled up")

    if do_plot:
        plt.plot(pattern_x, [0] * len(pattern_x))

    predictions = model_classification.predict(np.expand_dims([y_scaled_up], axis=2))[
        0, :
    ]

    top_5_predictions_indices = np.flip(np.argsort(predictions)[-5:])

    predicted_spgs = [spgs[i] for i in top_5_predictions_indices]

    if do_plot:
        plt.legend(title=f"Predicted: {predicted_spgs}, true: {spg_number}")

    patterns_counter += 1

    if predicted_spgs[0] == spg_number:
        correct_counter += 1

    if spg_number in predicted_spgs:
        correct_top5_counter += 1

    if select_which_to_use_for_testing:

        plt.pause(0.1)

        result = input(f"{i+1}/{len(raw_files)} Keep? (y/n)")

        if result == "y" or result == "Y":
            raw_files_keep.append(raw_file)

    else:

        if do_plot:
            plt.show()

if select_which_to_use_for_testing:
    with open("to_test_on.pickle", "wb") as file:
        pickle.dump(raw_files_keep, file)

print(f"Total number of patterns tested on: {patterns_counter}")
print(f"Correct: {correct_counter} ({correct_counter / patterns_counter * 100} %)")
print(
    f"Top-5: {correct_top5_counter} ({correct_top5_counter / patterns_counter * 100} %)"
)
