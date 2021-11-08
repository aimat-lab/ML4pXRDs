import numpy as np
import matplotlib.pyplot as plt
import sys
from UNet_1DCNN import UNet

sys.path.append("../")
from train_dataset.utils import load_experimental_data
from scipy import interpolate as ip
import tensorflow.keras as keras

remove_background = True
unet_model_path = "unet/removal_cps/weights50"
model_path = "unet/removal_cps/weights50"  # classifier model

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

    my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
    unet_model = my_unet.UNet()
    unet_model.load_weights()

    classifier_N = 4000  # TODO: Update this
    classifier_model = keras.models.load_model(model_path)

    for i in range(0, xs.shape[1]):

        current_xs = xs[:, i]
        current_ys = ys[:, i]

        if remove_background:

            f = ip.CubicSpline(current_xs, current_ys, bc_type="natural")
            ys = f(pattern_x)
            ys -= np.min(ys)
            ys = ys / np.max(ys)

            plt.plot(pattern_x, ys, label="Experimental rescaled")

            ys = np.expand_dims([ys], axis=2)

            corrected = unet_model.predict(ys)

            plt.plot(
                pattern_x, corrected[0, :, 0], label="Corrected via U-Net",
            )

            plt.plot(pattern_x, np.zeros(len(pattern_x)))

            plt.plot(pattern_x, ys - corrected[0, :, 0], label="Background and noise")

            label = classifier_model.predict(ys)

            print(f"Predicted label: {label}")
