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

to_test = "09-05-2022_12-24-34_UNetPP"

pattern_x = np.arange(0, 90.24, 0.02)
start_x = pattern_x[0]
end_x = pattern_x[-1]
N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

print(pattern_x)

model = keras.models.load_model("../unet/" + to_test + "/final")

raw_files = glob("../RRUFF_data/XY_RAW/*.txt")

for i, raw_file in enumerate(raw_files):

    raw_filename = os.path.basename(raw_file)
    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")

    # Model:
    pattern_x = np.arange(0, 90.24, 0.02)
    start_x = pattern_x[0]
    end_x = pattern_x[-1]
    N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

    x_test = raw_xy[:, 0]
    y_test = raw_xy[:, 1]

    x_test = x_test[::2]
    y_test = y_test[::2]

    y_test = np.array(y_test) / np.max(y_test)

    if x_test[0] > 0.0:
        to_add = np.arange(0.0, x_test[0] - 0.02, 0.02)
        x_test = np.concatenate((to_add, x_test), axis=0)
        y_test = np.concatenate((np.repeat([y_test[0]], len(to_add)), y_test), axis=0)

    if x_test[-1] < 90.24:
        to_add = np.arange(x_test[-1] + 0.02, 90.24, 0.02)
        x_test = np.concatenate((to_add, x_test), axis=0)
        y_test = np.concatenate((y_test, np.repeat([y_test[-1]], len(to_add))), axis=0)

    predictions = model.predict(np.expand_dims(np.expand_dims(y_test, 0), -1))

    plt.xlabel(r"$2 \theta$")
    plt.ylabel("Intensity")

    plt.plot(pattern_x, predictions[0, :, 0], label="Prediction")

    plt.plot(
        pattern_x,
        y_test,
        label="Input pattern",
    )

    plt.plot(
        pattern_x,
        y_test - predictions[0, :, 0],
        label="Prediced background and noise",
        linestyle="dotted",
    )

    plt.legend()
    plt.show()
    plt.figure()
