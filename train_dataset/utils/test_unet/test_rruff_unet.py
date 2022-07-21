import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from scipy.interpolate import CubicSpline
import pickle
from train_dataset.utils.test_unet.rruff_helpers import *
from train_dataset.utils.test_unet.heuristic_bg_utils import *

# This script is used to apply UNet and classification in sequence OR for selecting which patterns to test on
# Direct prediction of spg is done inside the training script

select_which_to_use_for_testing = False
use_only_selected = True
use_only_refitted = False
do_plot = False

unet_model_path = "10-06-2022_13-12-26_UNetPP"
classification_model_base = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/29-05-2022_22-12-56/"  # direct run
classification_model_path = classification_model_base + "final"

pattern_x_unet = np.arange(0, 90.24, 0.02)
start_x_unet = pattern_x_unet[0]
end_x_unet = pattern_x_unet[-1]
N_unet = len(pattern_x_unet)  # UNet works without error for N ~ 2^model_depth

model_unet = keras.models.load_model("../unet/" + unet_model_path + "/final")
model_classification = keras.models.load_model(classification_model_path)

with open(classification_model_base + "spgs.pickle", "rb") as file:
    spgs = pickle.load(file)

xs, ys, difs, raw_files, parameters = get_rruff_patterns(
    only_refitted_patterns=use_only_refitted,
    only_if_dif_exists=True,
    start_angle=0.0,
    end_angle=90.24,
    reduced_resolution=True,  # to be able to use UNet on this
    return_refitted_parameters=True,
)

patterns_counter = 0
correct_counter = 0
correct_top5_counter = 0
raw_files_keep = []

for i, ys in enumerate(ys):

    print(f"{i} of {len(raw_files)}")

    if select_which_to_use_for_testing and do_plot:
        plt.figure()
    else:
        plt.clf()

    # Model:
    pattern_x_unet = np.arange(0, 90.24, 0.02)
    start_x_unet = pattern_x_unet[0]
    end_x_unet = pattern_x_unet[-1]
    N_unet = len(pattern_x_unet)  # UNet works without error for N ~ 2^model_depth

    x_test = xs[i]
    y_test = ys[i]

    if not select_which_to_use_for_testing:

        predictions = model_unet.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

        if do_plot:
            plt.xlabel(r"$2 \theta$")
            plt.ylabel("Intensity")

        if do_plot:
            plt.plot(pattern_x_unet, predictions[0, :, 0], label="Prediction")

        if do_plot:
            plt.plot(
                pattern_x_unet,
                y_test,
                label="UNet Input pattern",
            )

        if do_plot:
            plt.plot(
                pattern_x_unet,
                y_test - predictions[0, :, 0],
                label="Prediced background and noise",
                linestyle="dotted",
            )

        data, wavelength, spg_number = dif_parser(difs[i])

        if data is None or wavelength is None or spg_number is None:
            continue

        if wavelength != 1.541838:
            raise Exception("Wrong wavelength found in pattern!")

        corrected_pattern = predictions[0, :, 0]
        # Dimensions of this: np.arange(0, 90.24, 0.02) (pattern_x)

        ##########

        classification_pattern_x = np.linspace(5, 90, 8501)

        # Scale up to dimensions needed for classification:
        f = CubicSpline(pattern_x_unet, corrected_pattern)

        y_scaled_up = f(classification_pattern_x)

        # y_scaled_up = savgol_filter(y_scaled_up, 19, 3)
        y_scaled_up -= np.min(y_scaled_up)
        y_scaled_up = y_scaled_up / np.max(y_scaled_up)

        # y_scaled_up[y_scaled_up < 0.002] = 0
        #
        # y_scaled_up = savgol_filter(y_scaled_up, 13, 3)

        if do_plot:
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

        if do_plot:
            plt.show()

    else:  # select_which_to_use_for_testing

        plt.plot(x_test, y_test)

        plt.pause(0.1)

        result = input(f"{i+1}/{len(raw_files)} Keep? (y/n)")

        if result == "y" or result == "Y":
            raw_files_keep.append(raw_files[i])

    patterns_counter += 1

if select_which_to_use_for_testing:
    with open("to_test_on.pickle", "wb") as file:
        pickle.dump(raw_files_keep, file)

print(f"Total number of patterns tested on: {patterns_counter}")
print(f"Correct: {correct_counter} ({correct_counter / patterns_counter * 100} %)")
print(
    f"Top-5: {correct_top5_counter} ({correct_top5_counter / patterns_counter * 100} %)"
)
