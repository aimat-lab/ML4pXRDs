import tensorflow.keras as keras
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from glob import glob
import pickle
from scipy.interpolate import CubicSpline
import pickle
from train_dataset.utils.test_unet.rruff_helpers import *
from train_dataset.utils.test_unet.heuristic_bg_utils import *

mode = "compare_UNet_heuristic"  # possible: "select_which_to_use_for_testing", "select_heuristic_parameters", "test_classification_accuracy", and "compare_UNet_heuristic"
# only for select_heuristic_parameters mode:
select_heuristic_parameters_by_best_fit = True
NO_samples_for_fit = 10

use_only_selected = True

compare_UNet_heuristic = True
test_classification_accuracy = not compare_UNet_heuristic
do_unet_preprocessing = True  # only relevant when testing classification accuracy

do_plot = False

unet_model_path = "10-06-2022_13-12-26_UNetPP"
classification_model_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/29-05-2022_22-12-56/"  # direct run
classification_model_path = classification_model_base + "final"

pattern_x_unet = np.arange(0, 90.24, 0.02)
start_x_unet = pattern_x_unet[0]
end_x_unet = pattern_x_unet[-1]
N_unet = len(pattern_x_unet)  # UNet works without error for N ~ 2^model_depth

print(pattern_x_unet)

model_unet = keras.models.load_model("../unet/" + unet_model_path + "/final")
model_classification = keras.models.load_model(classification_model_path)

with open(classification_model_base + "spgs.pickle", "rb") as file:
    spgs = pickle.load(file)

if mode == "select_which_to_use_for_testing":
    raw_files = glob("../RRUFF_data/XY_RAW/*.txt")
    raw_files_keep = []
else:
    with open("to_test_on.pickle", "rb") as file:
        raw_files = pickle.load(file)

with open("rruff_refits.pickle", "rb") as file:
    parameter_results = pickle.load(file)

patterns_counter = 0
correct_counter = 0
correct_top5_counter = 0

current_parameters = None

if select_heuristic_parameters_by_best_fit:
    raw_files = raw_files[0:NO_samples_for_fit]

for i, raw_file in enumerate(raw_files):

    print(f"{i} of {len(raw_files)}")

    if mode != "select_which_to_use_for_testing" and do_plot:
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
    pattern_x_unet = np.arange(0, 90.24, 0.02)
    start_x_unet = pattern_x_unet[0]
    end_x_unet = pattern_x_unet[-1]
    N_unet = len(pattern_x_unet)  # UNet works without error for N ~ 2^model_depth

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
        if do_unet_preprocessing or compare_UNet_heuristic:
            x_test = x_test[::2]
            y_test = y_test[::2]
    elif abs(dx - 0.02) < 0.0000001:
        print("")
        pass
    else:
        print(f"Skipping pattern with dx={dx}.")
        continue

    if (
        not do_unet_preprocessing and not compare_UNet_heuristic
    ):  # both need fitting dimensions for UNet
        dx = 0.01
        start_angle = 5
        end_angle = 90
    else:
        dx = 0.02
        start_angle = 0
        end_angle = 90.24

    y_test = np.array(y_test)
    y_test -= min(y_test)
    y_test = y_test / np.max(y_test)

    # For now don't use those:
    if x_test[0] > 5.0 or x_test[-1] < 90.0:
        continue

    if x_test[0] > start_angle:
        to_add = np.arange(0.0, x_test[0], dx)
        x_test = np.concatenate((to_add, x_test), axis=0)
        y_test = np.concatenate((np.repeat([y_test[0]], len(to_add)), y_test), axis=0)

    if x_test[-1] < end_angle:
        to_add = np.arange(x_test[-1] + dx, end_angle, dx)
        x_test = np.concatenate((x_test, to_add), axis=0)
        y_test = np.concatenate((y_test, np.repeat([y_test[-1]], len(to_add))), axis=0)

    # print(x_test[-300:])

    if ((do_unet_preprocessing or compare_UNet_heuristic) and len(x_test) != 4512) or (
        (not do_unet_preprocessing and not compare_UNet_heuristic)
        and len(x_test) != 8501
    ):
        print("Skipping pattern due to wrong dimensions of xs.")
        continue

    if mode == "select_heuristic_parameters":

        if select_heuristic_parameters_by_best_fit:

            plt.plot(x_test, y_test)
            current_parameters = plot_heuristic_fit(x_test, y_test)

            # construct the fit function:

        continue

    # TODO: Keep this here
    if not select_which_to_use_for_testing and (
        do_unet_preprocessing or compare_UNet_heuristic
    ):
        predictions = model_unet.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

    ##### TODO: Move this stuff down
    # TODO: Continue downwards from here

    if do_plot:
        plt.xlabel(r"$2 \theta$")
        plt.ylabel("Intensity")

    if (
        not select_which_to_use_for_testing
        and (do_unet_preprocessing or compare_UNet_heuristic)
        and do_plot
    ):
        plt.plot(pattern_x_unet, predictions[0, :, 0], label="Prediction")

    if do_plot and (do_unet_preprocessing or compare_UNet_heuristic):
        plt.plot(
            pattern_x_unet,
            y_test,
            label="UNet Input pattern",
        )

    if (
        not select_which_to_use_for_testing
        and (do_unet_preprocessing or compare_UNet_heuristic)
        and do_plot
    ):
        plt.plot(
            pattern_x_unet,
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

    if do_unet_preprocessing or compare_UNet_heuristic:
        corrected_pattern = predictions[0, :, 0]
    # Dimensions of this: np.arange(0, 90.24, 0.02) (pattern_x)

    ##########

    if test_classification_accuracy:

        classification_pattern_x = np.linspace(5, 90, 8501)
        if do_unet_preprocessing:
            # Needed for classification:

            f = CubicSpline(pattern_x_unet, corrected_pattern)

            y_scaled_up = f(classification_pattern_x)

        else:

            y_scaled_up = y_test

        # y_scaled_up = savgol_filter(y_scaled_up, 19, 3)
        y_scaled_up -= np.min(y_scaled_up)
        y_scaled_up = y_scaled_up / np.max(y_scaled_up)

        # y_scaled_up[y_scaled_up < 0.002] = 0
        #
        # y_scaled_up = savgol_filter(y_scaled_up, 13, 3)

        if True and do_plot:
            plt.plot(classification_pattern_x, y_scaled_up, label="Scaled up")

        if do_plot:
            plt.plot(pattern_x_unet, [0] * len(pattern_x_unet))

        predictions = model_classification.predict(
            np.expand_dims([y_scaled_up], axis=2)
        )[0, :]

        top_5_predictions_indices = np.flip(np.argsort(predictions)[-5:])

        predicted_spgs = [spgs[i] for i in top_5_predictions_indices]

        if do_plot:
            plt.legend(title=f"Predicted: {predicted_spgs}, true: {spg_number}")

        if predicted_spgs[0] == spg_number:
            correct_counter += 1

        if spg_number in predicted_spgs:
            correct_top5_counter += 1

    else:  # compare heuristic and UNet

        pass

        # TODO: Run heuristic algos on this (using the parameters)
        # TODO: Create the real background function from the parameters (loaded from file previously)
        # TODO: Calculate mean absolute error between UNet output and ground truth vs. heuristic output vs. UNet output

    if select_which_to_use_for_testing:

        plt.plot(x_test, y_test)

        plt.pause(0.1)

        result = input(f"{i+1}/{len(raw_files)} Keep? (y/n)")

        if result == "y" or result == "Y":
            raw_files_keep.append(raw_file)

    else:

        if do_plot:
            plt.show()

    patterns_counter += 1

if select_which_to_use_for_testing:
    with open("to_test_on.pickle", "wb") as file:
        pickle.dump(raw_files_keep, file)

if select_heuristic_parameters:
    print("Final heuristic parameters: ", current_parameters)
    with open("final_heuristic_parameters.pickle", "wb") as file:
        pickle.dump(current_parameters, file)

print(f"Total number of patterns tested on: {patterns_counter}")
print(f"Correct: {correct_counter} ({correct_counter / patterns_counter * 100} %)")
print(
    f"Top-5: {correct_top5_counter} ({correct_top5_counter / patterns_counter * 100} %)"
)
