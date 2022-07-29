from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pickle
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity

skip_first_N = 20
N_to_process = None  # None possible => use all

N_polynomial_coefficients = 20
R2_score_threshold = 0.95  # minimum 0.9

unet_model_path = "10-06-2022_13-12-26_UNetPP"
model_unet = keras.models.load_model("../../unet/" + unet_model_path + "/final")

do_plot = False
print_singular = False

xs, ys, difs, raw_files, parameters, scores = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=0.0,
    end_angle=90.24,
    reduced_resolution=True,  # to be able to use UNet on this
    return_refitted_parameters=True,
)

for i in reversed(range(len(scores))):
    if scores[i] < R2_score_threshold:
        del scores[i]
        del parameters[i]
        del raw_files[i]
        del difs[i]
        del ys[i]
        del xs[i]

xs = xs[skip_first_N:]
ys = ys[skip_first_N:]
difs = difs[skip_first_N:]
raw_files = raw_files[skip_first_N:]
parameters = parameters[skip_first_N:]

if N_to_process is not None:
    xs = xs[:N_to_process]
    ys = ys[:N_to_process]
    difs = difs[:N_to_process]
    raw_files = raw_files[:N_to_process]
    parameters = parameters[:N_to_process]

x_range = np.linspace(5, 90, 8501)

with open("bestfit_heuristic.pickle", "rb") as file:
    bestfit_parameters = pickle.load(file)


def background_fit(xs, *params):
    result = np.zeros(len(xs))
    for j in range(N_polynomial_coefficients):
        result += params[j] * xs**j
    return result


for method in ["arPLS", "rb"]:

    diffs_method = []
    diffs_unet = []

    ssims_method = []
    ssims_unet = []

    for i in range(len(xs)):
        print(f"{i} of {len(xs)}, score {scores[i]}")

        if print_singular:
            print(raw_files[i])

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

        target_y = np.zeros(len(xs[i]))
        for j in range(N_polynomial_coefficients):
            target_y += parameters[i][j] * xs[i] ** j

        if do_plot:
            plt.plot(xs[i], ys[i], label="Input")
            # plt.plot(xs[i], ys[i] - predictions, label="UNet")
            plt.plot(xs[i], result_heuristic, label=method)

        xs_to_fit = xs[i][500:-500]
        ys_to_fit = ys[i][500:-500]
        predictions = predictions[500:-500]
        result_heuristic = result_heuristic[500:-500]
        target_y = target_y[500:-500]

        if do_plot:
            plt.plot(xs_to_fit, target_y, label="Target")

        result_unet_fit = curve_fit(
            background_fit,
            xs_to_fit,
            ys_to_fit - predictions,
            p0=[0.0] * N_polynomial_coefficients,
        )[0]
        ys_unet_fit = background_fit(xs_to_fit, *result_unet_fit)

        if do_plot:
            plt.plot(xs_to_fit, ys_unet_fit, label="UNet Fit")

        result_rb_fit = curve_fit(
            background_fit,
            xs_to_fit,
            result_heuristic,
            p0=[0.0] * N_polynomial_coefficients,
            maxfev=20000,
        )[0]
        ys_rb_fit = background_fit(xs_to_fit, *result_rb_fit)

        if do_plot:
            plt.plot(xs_to_fit, ys_rb_fit, label=method + " Fit")

        diff_unet = np.sum(np.square(ys_unet_fit - target_y))
        diffs_unet.append(diff_unet)
        diff_method = np.sum(np.square(ys_rb_fit - target_y))
        diffs_method.append(diff_method)

        ssims_unet.append(structural_similarity(ys_unet_fit, target_y))
        ssims_method.append(structural_similarity(ys_rb_fit, target_y))

        if print_singular:
            print("Difference / score UNet (mse, ssim):", diff_unet, ssims_unet[-1])
            print(
                f"Difference / score {method} (mse, ssim):",
                diff_method,
                ssims_method[-1],
            )

        if do_plot:
            plt.legend()
            plt.show()

    print(
        "Average difference / score UNet (mse, ssim):",
        np.average(diffs_unet),
        np.average(ssims_unet),
    )
    print(
        f"Average difference / score {method} (mse, ssim):",
        np.average(diffs_method),
        np.average(ssims_method),
    )
