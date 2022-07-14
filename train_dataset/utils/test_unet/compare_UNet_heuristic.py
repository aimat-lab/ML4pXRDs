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
    heuristic_parameters = pickle.load(file)


def background_fit(xs, a0, a1, a2, a3, a4, a5):
    return a0 + a1 * xs + a2 * xs**2 + a3 * xs**3 + a4 * xs**4 + a5 * xs**5


for i in range(len(xs)):

    predictions = model_unet.predict(np.expand_dims(np.expand_dims(ys[i], 0), -1))[
        0, :, 0
    ]

    result_rb = rolling_ball(
        xs[i], ys[i], heuristic_parameters[0], heuristic_parameters[1]
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
    plt.plot(xs[i], result_rb, label="Rolling ball")
    plt.plot(xs[i], target_y, label="Target")

    result_unet_fit = curve_fit(background_fit, xs[i], ys[i] - predictions)[0]
    ys_unet_fit = background_fit(xs[i], *result_unet_fit)
    plt.plot(xs[i], ys_unet_fit, label="UNet Fit")

    result_rb_fit = curve_fit(background_fit, xs[i], result_rb)[0]
    ys_rb_fit = background_fit(xs[i], *result_rb_fit)
    plt.plot(xs[i], ys_rb_fit, label="RB Fit")

    print("Difference UNet:", np.sum(np.square(ys_unet_fit - target_y)))
    print("Difference RB:", np.sum(np.square(ys_rb_fit - target_y)))

    plt.legend()

    plt.show()

    # TODO: Output MSE for rb and UNet
