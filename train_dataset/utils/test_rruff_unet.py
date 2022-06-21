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

select_which_to_use_for_testing = False
use_only_selected = True

do_plot = True

# to_test = "18-05-2022_09-45-42_UNetPP"
unet_model_path = "10-06-2022_13-12-26_UNetPP"
classification_model_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/07-06-2022_09-43-41/"
classification_model_path = classification_model_base + "final"

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

model_unet = keras.models.load_model("../unet/" + unet_model_path + "/final")
model_classification = keras.models.load_model(classification_model_path)

with open(classification_model_base + "spgs.pickle", "rb") as file:
    spgs = pickle.load(file)


def dif_parser(path):

    try:

        with open(path, "r") as file:
            content = file.readlines()

        relevant_content = []
        is_reading = False
        wavelength = None

        for line in content:

            if "X-RAY WAVELENGTH" in line:
                wavelength = float(line.replace("X-RAY WAVELENGTH:", "").strip())

            if "SPACE GROUP" in line:
                spg_specifier = (
                    line.replace("SPACE GROUP:", "")
                    .replace("ALTERNATE SETTING FOR", "")
                    .strip()
                    .replace("_", "")
                )

                spg_number = None

                if spg_specifier == "Fm3m":
                    spg_specifier = "Fm-3m"
                elif spg_specifier == "Pncm":
                    spg_number = 53
                elif spg_specifier == "C-1":
                    spg_number = 1
                elif (
                    spg_specifier == "P21/n"
                    or spg_specifier == "P21/a"
                    or spg_specifier == "P21/b"
                ):
                    spg_number = 14
                elif (
                    spg_specifier == "Pbnm"
                    or spg_specifier == "Pcmn"
                    or spg_specifier == "Pnam"
                ):
                    spg_number = 62
                elif spg_specifier == "Amma":
                    spg_number = 63
                elif spg_specifier == "Fd2d":
                    spg_number = 43
                elif spg_specifier == "Fd3m":
                    spg_specifier = "Fd-3m"
                elif (
                    spg_specifier == "A2/a"
                    or spg_specifier == "I2/a"
                    or spg_specifier == "I2/c"
                ):
                    spg_number = 15
                elif spg_specifier == "P4/n":
                    spg_number = 85
                elif spg_specifier == "I41/acd":
                    spg_number = 142
                elif spg_specifier == "I41/amd":
                    spg_number = 141
                elif spg_specifier == "Pmcn":
                    spg_number = 62
                elif spg_specifier == "I41/a":
                    spg_number = 88
                elif spg_specifier == "Pbn21" or spg_specifier == "P21nb":
                    spg_number = 33
                elif spg_specifier == "P2cm":
                    spg_number = 28
                elif spg_specifier == "P4/nnc":
                    spg_number = 126
                elif spg_specifier == "Pn21m":
                    spg_number = 31
                elif spg_specifier == "B2/b":
                    spg_number = 15
                elif spg_specifier == "Cmca":
                    spg_number = 64
                elif spg_specifier == "I2/m" or spg_specifier == "A2/m":
                    spg_number = 12
                elif spg_specifier == "Pcan":
                    spg_number = 60
                elif spg_specifier == "Ia3d":
                    spg_specifier = "Ia-3d"
                elif spg_specifier == "P4/nmm":
                    spg_number = 129
                elif spg_specifier == "Pa3":
                    spg_specifier = "Pa-3"
                elif spg_specifier == "P4/ncc":
                    spg_number = 130
                elif spg_specifier == "Imam":
                    spg_number = 74
                elif spg_specifier == "Pmmn":
                    spg_number = 59
                elif spg_specifier == "Pncn" or spg_specifier == "Pbnn":
                    spg_number = 52
                elif spg_specifier == "Bba2":
                    spg_number = 41
                elif spg_specifier == "C1":
                    spg_number = 1
                elif spg_specifier == "Pn3":
                    spg_specifier = "Pn-3"
                elif spg_specifier == "Fddd":
                    spg_number = 70
                elif spg_specifier == "Pcab":
                    spg_number = 61
                elif spg_specifier == "P2/a":
                    spg_number = 13
                elif spg_specifier == "Pmnb":
                    spg_number = 62
                elif spg_specifier == "I-1":
                    spg_number = 2
                elif spg_specifier == "Pmnb":
                    spg_number = 154
                elif spg_specifier == "B2mb":
                    spg_number = 40
                elif spg_specifier == "Im3":
                    spg_specifier = "Im-3"
                elif spg_specifier == "Pn21a":
                    spg_number = 33
                elif spg_specifier == "Pm2m":
                    spg_number = 25
                elif spg_specifier == "Fd3":
                    spg_specifier = "Fd-3"
                elif spg_specifier == "Im3m":
                    spg_specifier = "Im-3m"
                elif spg_specifier == "Cmma":
                    spg_number = 67
                elif spg_specifier == "Pn3m":
                    spg_specifier = "Pn-3m"
                elif spg_specifier == "F2/m":
                    spg_number = 12
                elif spg_specifier == "Pnm21":
                    spg_number = 31

                if spg_number is None:
                    spg_object = Group(spg_specifier)
                    spg_number = spg_object.number

            if (
                "==========" in line
                or "XPOW Copyright" in line
                or "For reference, see Downs" in line
            ) and is_reading:
                break

            if is_reading:
                relevant_content.append(line)

            if "2-THETA" in line and "INTENSITY" in line and "D-SPACING" in line:
                is_reading = True
            elif "2-THETA" in line and "D-SPACING" in line and not "INTENSITY" in line:
                print(f"Error processing file {path}:")
                print("No intensity data found.")
                return None, None, None

        data = np.genfromtxt(relevant_content)[:, 0:2]

        if wavelength is None:
            print(f"Error for file {path}:")
            print("No wavelength information found.")
            return None, None, None

        return data, wavelength, spg_number

    except Exception as ex:
        print(f"Error processing file {path}:")
        print(ex)
        return None, None, None


if not use_only_selected:
    raw_files = glob("../RRUFF_data/XY_RAW/*.txt")
else:
    with open("to_test_on.pickle", "rb") as file:
        raw_files = pickle.load(file)

if select_which_to_use_for_testing:
    raw_files_keep = []

patterns_counter = 0
correct_counter = 0

for i, raw_file in enumerate(raw_files):

    print(f"{i} of {len(raw_files)}")

    if not select_which_to_use_for_testing:
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

    if not select_which_to_use_for_testing:
        predictions = model_unet.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

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

    data, wavelength, spg_number = dif_parser(dif_file)

    if data is None or wavelength is None or spg_number is None:
        continue

    corrected_pattern = predictions[0, :, 0]
    # Dimensions of this: np.arange(0, 90.24, 0.02) (pattern_x)

    # Needed for classification:
    classification_pattern_x = np.linspace(5, 90, 8501)

    f = CubicSpline(pattern_x, corrected_pattern)

    y_scaled_up = f(classification_pattern_x)

    y_scaled_up = savgol_filter(y_scaled_up, 19, 3)
    y_scaled_up = y_scaled_up / np.max(y_scaled_up)

    if True:
        plt.plot(classification_pattern_x, y_scaled_up, label="Scaled up")

    plt.plot(pattern_x, [0] * len(pattern_x))

    predictions = model_classification.predict(np.expand_dims([y_scaled_up], axis=2))[
        0, :
    ]

    top_5_predictions_indices = np.flip(np.argsort(predictions)[-5:])

    predicted_spgs = [spgs[i] for i in top_5_predictions_indices]

    plt.legend(title=f"Predicted: {predicted_spgs}, true: {spg_number}")

    patterns_counter += 1

    if predicted_spgs[0] == spg_number:
        correct_counter += 1

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
