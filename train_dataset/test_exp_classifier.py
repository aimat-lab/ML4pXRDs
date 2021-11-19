import numpy as np
import matplotlib.pyplot as plt
import sys
from UNet_1DCNN import UNet

sys.path.append("../")
from train_dataset.utils import load_experimental_data
from scipy import interpolate as ip
import tensorflow.keras as keras
import pickle
import pandas as pd

remove_background = True

if __name__ == "__main__":

    xs_exp, ys_exp = load_experimental_data("exp_data/XRDdata_classification.csv")

    N = 9018
    pattern_x = np.linspace(0, 90, N)

    start_x = 10
    end_x = 50
    start_index = np.argwhere(pattern_x >= start_x)[0][0]
    end_index = np.argwhere(pattern_x <= end_x)[-1][0]
    pattern_x = pattern_x[start_index : end_index + 1]
    N = len(pattern_x)

    classifier_model_name = "narrow_19-11-2021_08:18:39_test"
    unet_model_name = "removal_17-11-2021_16-03-57_variance_30"

    classifier_model = keras.models.load_model(
        "classifier/" + classifier_model_name + "/final"
    )
    unet_model = keras.models.load_model("unet/" + unet_model_name + "/final")
    probability_model = keras.Sequential(
        [classifier_model, keras.layers.Activation("sigmoid")]
    )

    data_true_labels = pd.read_csv(
        "exp_data/experimental_phases.txt", delimiter=" ", skiprows=0
    )
    labels = np.array(data_true_labels.iloc[:, 2])

    for i in range(0, xs_exp.shape[1]):

        current_xs = xs_exp[:, i]
        current_ys = ys_exp[:, i]

        if remove_background:

            # Scale experimental pattern to the right dimension
            f = ip.CubicSpline(current_xs, current_ys, bc_type="natural")
            ys = f(pattern_x)
            ys -= np.min(ys)
            ys = ys / np.max(ys)

            plt.plot(pattern_x, ys, label="Experimental rescaled")

            ys_to_be_corrected = np.expand_dims([ys], axis=2)
            corrected = unet_model.predict(ys_to_be_corrected)

            with open("classifier/scaler", "rb") as file:
                sc = pickle.load(file)
                ys_to_be_classified = sc.transform(
                    [corrected[0, :, 0] / np.max(corrected[0, :, 0])]
                )
                ys_to_be_classified = np.expand_dims(ys_to_be_classified, axis=2)

            label = probability_model.predict(ys_to_be_classified)

            print(f"Output of classification: {label}")

            narrow_phases = ["Fm-3m", "Ia-3", "P63/m"]

            plt.plot(
                pattern_x,
                corrected[0, :, 0],
                label=f"Corrected via U-Net\n\nPredicted label: {narrow_phases[np.argmax(label[0])]}\nTrue label: {labels[i]}",
            )

            plt.plot(pattern_x, np.zeros(len(pattern_x)))

            plt.plot(pattern_x, ys - corrected[0, :, 0], label="Background and noise")

            plt.legend()

            plt.show()
