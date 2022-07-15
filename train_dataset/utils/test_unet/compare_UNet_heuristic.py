from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pickle
from scipy.optimize import curve_fit

skip_first_N = 5

unet_model_path = "10-06-2022_13-12-26_UNetPP"
model_unet = keras.models.load_model("../../unet/" + unet_model_path + "/final")

xs, ys, difs, raw_files, parameters = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=0.0,
    end_angle=90.24,
    reduced_resolution=True,  # to be able to use UNet on this
    return_refitted_parameters=True,
)

x_range = np.linspace(5, 90, 8501)

with open("bestfit_heuristic.pickle", "rb") as file:
    bestfit_parameters = pickle.load(file)


def background_fit(xs, a0, a1, a2, a3, a4, a5):
    return a0 + a1 * xs + a2 * xs**2 + a3 * xs**3 + a4 * xs**4 + a5 * xs**5


for method in ["arPLS", "rb"]:

    for i in range(len(xs)):

        predictions = model_unet.predict(np.expand_dims(np.expand_dims(ys[i], 0), -1))[
            0, :, 0
        ]

        if method == "rb":
            result_heuristic = rolling_ball(
                xs[i],
                ys[i],
                bestfit_parameters[method][0],
                bestfit_parameters[method][1],
            )
        else:
            result_heuristic = baseline_arPLS(
                ys[i],
                ratio=10 ** bestfit_parameters[method][0],
                lam=10 ** bestfit_parameters[method][1],
            )

        target_y = (
            parameters[i][0]
            + parameters[i][1] * xs[i]
            + parameters[i][2] * xs[i] ** 2
            + parameters[i][3] * xs[i] ** 3
            + parameters[i][4] * xs[i] ** 4
            + parameters[i][5] * xs[i] ** 5
        )

        plt.plot(xs[i], ys[i], label="Input")
        plt.plot(xs[i], ys[i] - predictions, label="UNet")
        plt.plot(xs[i], result_heuristic, label=method)
        plt.plot(xs[i], target_y, label="Target")

        xs_to_fit = xs[i][250:]
        ys_to_fit = ys[i][250:]
        predictions = predictions[250:]
        result_heuristic = result_heuristic[250:]
        target_y = target_y[250:]

        result_unet_fit = curve_fit(background_fit, xs_to_fit, ys_to_fit - predictions)[
            0
        ]
        ys_unet_fit = background_fit(xs_to_fit, *result_unet_fit)
        plt.plot(xs_to_fit, ys_unet_fit, label="UNet Fit")

        result_rb_fit = curve_fit(background_fit, xs_to_fit, result_heuristic)[0]
        ys_rb_fit = background_fit(xs_to_fit, *result_rb_fit)
        plt.plot(xs_to_fit, ys_rb_fit, label=method + " Fit")

        print("Difference UNet:", np.sum(np.square(ys_unet_fit - target_y)))
        print(f"Difference {method}:", np.sum(np.square(ys_rb_fit - target_y)))

        plt.legend()

        plt.show()
