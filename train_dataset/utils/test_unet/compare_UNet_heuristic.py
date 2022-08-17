from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pickle
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import matplotlib_defaults

figure_double_width_pub = matplotlib_defaults.pub_width

skip_first_N = 10
N_to_process = None  # None possible => use all

N_polynomial_coefficients = 12
R2_score_threshold = 0.97  # minimum 0.9

ratio_fixed = 10 ** (-5)

unet_model_path = "10-06-2022_13-12-26_UNetPP"  # Previous model, this had superior accuracy to the heuristic algorithms.
# unet_model_path = "31-07-2022_12-40-47"  # with caglioti + different bg parameters etc.
# unet_model_path = "05-08-2022_07-59-47"  # Latest model with caglioti # continuation!
# These two latest runs had faulty caglioti peak profiles! Therefore, the run from 10-06 is used in the thesis!

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

print(len(xs))

xs = xs[skip_first_N:]
ys = ys[skip_first_N:]
difs = difs[skip_first_N:]
raw_files = raw_files[skip_first_N:]
parameters = parameters[skip_first_N:]
scores = scores[skip_first_N:]

if N_to_process is not None:
    xs = xs[:N_to_process]
    ys = ys[:N_to_process]
    difs = difs[:N_to_process]
    raw_files = raw_files[:N_to_process]
    parameters = parameters[:N_to_process]
    scores = scores[:N_to_process]

x_range = np.linspace(5, 90, 8501)

with open("bestfit_heuristic.pickle", "rb") as file:
    bestfit_parameters = pickle.load(file)


def background_fit(xs, *params):
    result = np.zeros(len(xs))
    for j in range(N_polynomial_coefficients):
        result += params[j] * xs**j
    return result


diffs_arPLS = []
diffs_rb = []
diffs_unet = []

ssims_arPLS = []
ssims_rb = []
ssims_unet = []

for i in range(len(xs)):

    # print(raw_files[i])

    if do_plot:
        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95,
                figure_double_width_pub * 0.7,
            )
        )

    print(f"{i} of {len(xs)}, score {scores[i]}")

    if do_plot:
        plt.plot(xs[i], ys[i], label="Input")
        # plt.plot(xs[i], ys[i] - predictions, label="UNet")

    if print_singular:
        print(raw_files[i])

    predictions = model_unet.predict(np.expand_dims(np.expand_dims(ys[i], 0), -1))[
        0, :, 0
    ]

    result_unet_fit = curve_fit(
        background_fit,
        xs[i],
        ys[i] - predictions,
        p0=[0.0] * N_polynomial_coefficients,
    )[0]
    ys_unet_fit = background_fit(xs[i], *result_unet_fit)

    if do_plot:
        plt.plot(xs[i], ys_unet_fit, label="U-Net polynomial fit")

    target_y = np.zeros(len(xs[i]))
    for j in range(N_polynomial_coefficients):
        target_y += parameters[i][j] * xs[i] ** j

    if do_plot:
        plt.plot(xs[i], target_y, label="Rietveld (target)")

    target_y = target_y[500:-500]
    ys_unet_fit = ys_unet_fit[500:-500]

    diff_unet = np.sum(np.square(ys_unet_fit - target_y))
    diffs_unet.append(diff_unet)
    ssims_unet.append(structural_similarity(ys_unet_fit, target_y))

    if print_singular:
        print("Difference / score UNet (mse, ssim):", diff_unet, ssims_unet[-1])

    for method in ["arPLS", "rb"]:

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
                ratio=ratio_fixed,
                # ratio=10 ** (-5),
                lam=10 ** bestfit_parameters[method][0],
                # lam=10 ** (9.15943541e00),
            )

        if do_plot:
            plt.plot(xs[i], result_heuristic, label=method)

        result_heuristic = result_heuristic[500:-500]

        diff_method = np.sum(np.square(result_heuristic - target_y))

        if method == "rb":
            diffs_rb.append(diff_method)
        else:
            diffs_arPLS.append(diff_method)

        if method == "rb":
            ssims_rb.append(structural_similarity(result_heuristic, target_y))
        else:
            ssims_arPLS.append(structural_similarity(result_heuristic, target_y))

        if print_singular:
            print(
                f"Difference / score {method} (mse, ssim):",
                diff_method,
                ssims_rb[-1] if method == "rb" else ssims_arPLS[-1],
            )

    if do_plot:
        plt.legend()
        plt.ylim((0, 0.1))
        plt.xlim((10, 80))
        plt.xlabel(r"$2 \theta$")
        plt.ylabel(r"Intensity / rel.")
        plt.savefig(f"comparison_{i}.pdf")
        plt.show()

    if do_plot:
        plt.figure(
            figsize=(
                figure_double_width_pub * 0.95,
                figure_double_width_pub * 0.7,
            )
        )
        plt.plot(xs[i], ys[i], label="Input")
        plt.plot(xs[i], predictions, label="U-Net result")
        # plt.ylim((0, 0.4))

        plt.legend()
        plt.xlabel(r"$2 \theta$")
        plt.ylabel(r"Intensity / rel.")
        plt.savefig(f"unet_{i}.pdf")
        plt.show()

print(
    "Average difference / score UNet (mse, ssim):",
    np.average(diffs_unet),
    np.average(ssims_unet),
)
print(
    f"Average difference / score arPLS (mse, ssim):",
    np.average(diffs_arPLS),
    np.average(ssims_arPLS),
)
print(
    f"Average difference / score rb (mse, ssim):",
    np.average(diffs_rb),
    np.average(ssims_rb),
)

counter_unet_better_than_arPLS = 0
for i in range(len(diffs_unet)):
    if diffs_unet[i] < diffs_arPLS[i]:
        counter_unet_better_than_arPLS += 1
print(f"{counter_unet_better_than_arPLS} of {len(diffs_unet)} UNet better than arPLS")

counter_unet_better_than_rb = 0
for i in range(len(diffs_unet)):
    if diffs_unet[i] < diffs_rb[i]:
        counter_unet_better_than_rb += 1
print(f"{counter_unet_better_than_rb} of {len(diffs_unet)} UNet better than rb")

if False:
    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.hist(diffs_unet, alpha=0.5, label="U-Net squared error", bins=15)
    plt.hist(diffs_method, alpha=0.5, label=method + " squared error", bins=15)
    plt.legend()
    plt.savefig(f"diffs_{method}.pdf")
