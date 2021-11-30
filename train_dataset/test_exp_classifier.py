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

    classifier_model_name = "narrow_30-11-2021_09:42:59_test"
    unet_model_name = "removal_17-11-2021_16-03-57_variance_30"

    classifier_model = keras.models.load_model(
        "classifier/" + classifier_model_name + "/final", compile=False
    )
    unet_model = keras.models.load_model("unet/" + unet_model_name + "/final")

    data_true_labels = pd.read_csv(
        "exp_data/experimental_phases.txt", delimiter=" ", skiprows=0, header=None
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

            softmax_activation = keras.layers.Activation("softmax")(
                classifier_model.get_layer("outputs_softmax").output
            )
            prob_model_softmax = keras.Model(
                inputs=classifier_model.layers[0].output, outputs=softmax_activation
            )
            prediction_softmax = prob_model_softmax.predict(ys_to_be_classified)
            prediction_softmax = np.argmax(prediction_softmax, axis=1)

            sigmoid_activation = keras.layers.Activation("sigmoid")(
                classifier_model.get_layer("output_sigmoid").output
            )
            prob_model_sigmoid = keras.Model(
                inputs=classifier_model.layers[0].output, outputs=sigmoid_activation
            )
            prediction_sigmoid = prob_model_sigmoid.predict(ys_to_be_classified)
            prediction_sigmoid = prediction_sigmoid[:, 0]
            prediction_sigmoid = np.where(prediction_sigmoid > 0.5, 1, 0)

            narrow_phases = ["Fm-3m", "Ia-3", "P63/m"]
            purities = ["non-pure", "pure"]
            print(
                f"Output of phase classification: {narrow_phases[prediction_softmax[0]]}"
            )
            print(f"Output of pure classification: {purities[prediction_sigmoid[0]]}")

            plt.plot(
                pattern_x,
                corrected[0, :, 0],
                label=f"Corrected via U-Net\n\nPredicted labels: {narrow_phases[prediction_softmax[0]]}, {purities[prediction_sigmoid[0]]}\nTrue label: {labels[i]}",
            )

            plt.plot(pattern_x, np.zeros(len(pattern_x)))

            plt.plot(pattern_x, ys - corrected[0, :, 0], label="Background and noise")

            plt.legend()

            plt.show()
