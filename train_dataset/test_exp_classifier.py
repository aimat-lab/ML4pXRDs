import numpy as np
import matplotlib.pyplot as plt
import sys
from UNet_1DCNN import UNet

sys.path.append("../")
from train_dataset.utils import load_experimental_data
from scipy import interpolate as ip
import tensorflow.keras as keras

remove_background = True

unet_model = ""
classifier_model = ""

if __name__ == "__main__":

    xs, ys = load_experimental_data("exp_data/XRDdata.csv")

    N = 9018
    pattern_x = np.linspace(0, 90, N)

    start_x = 10
    end_x = 50
    start_index = np.argwhere(pattern_x >= start_x)[0][0]
    end_index = np.argwhere(pattern_x <= end_x)[-1][0]
    pattern_x = pattern_x[start_index : end_index + 1]
    N = len(pattern_x)

    classifier_model_name = ""
    unet_model_name = ""

    classifier_model = keras.models.load_model(
        "classifier/" + classifier_model_name + "/final"
    )
    unet_model = keras.models.load_model("unet/" + +unet_model_name + "/final")

    for i in range(0, xs.shape[1]):

        current_xs = xs[:, i]
        current_ys = ys[:, i]

        if remove_background:

            # Scale experimental pattern to the right dimension
            f = ip.CubicSpline(current_xs, current_ys, bc_type="natural")
            ys = f(pattern_x)
            ys -= np.min(ys)
            ys = ys / np.max(ys)

            plt.plot(pattern_x, ys, label="Experimental rescaled")

            ys = np.expand_dims([ys], axis=2)

            corrected = unet_model.predict(ys)

            probability_model = keras.Sequential(
                [classifier_model, keras.layers.Activation("sigmoid")]
            )
            label = probability_model.predict(ys)

            print(f"Predicted label: {label}")

            plt.plot(
                pattern_x,
                corrected[0, :, 0],
                label=f"Corrected via U-Net; predicted label: {label}",
            )

            plt.plot(pattern_x, np.zeros(len(pattern_x)))

            plt.plot(pattern_x, ys - corrected[0, :, 0], label="Background and noise")

