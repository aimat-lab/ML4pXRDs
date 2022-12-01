from glob import glob
from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pickle
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import utils.matplotlib_defaults as matplotlib_defaults

figure_double_width_pub = matplotlib_defaults.pub_width

N_polynomial_coefficients = 12
R2_score_threshold = 0.97  # minimum 0.9

ratio_fixed = 10 ** (-5)

unet_model_path = "10-06-2022_13-12-26_UNetPP"  # Previous model, this had superior accuracy to the heuristic algorithms.
# unet_model_path = "31-07-2022_12-40-47"  # with caglioti + different bg parameters etc.
# unet_model_path = "05-08-2022_07-59-47"  # Latest model with caglioti # continuation!
# These two latest runs had faulty caglioti peak profiles! Therefore, the run from 10-06 is used in the thesis!

model_unet = keras.models.load_model("../../unet/" + unet_model_path + "/final")

do_plot = False
print_singular = True
dont_use_excluded = True

xs, ys, difs, raw_files, parameters, scores = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=0.0,
    end_angle=90.24,
    reduced_resolution=True,  # to be able to use UNet on this
    return_refitted_parameters=True,
)

if dont_use_excluded:
    with open("to_exclude.pickle", "rb") as file:
        to_exclude = pickle.load(file)

for i in reversed(range(len(scores))):
    if scores[i] < R2_score_threshold or (
        dont_use_excluded and raw_files[i] in to_exclude
    ):
        del scores[i]
        del parameters[i]
        del raw_files[i]
        del difs[i]
        del ys[i]
        del xs[i]

x_range = np.linspace(5, 90, 8501)


def background_fit(xs, *params):
    result = np.zeros(len(xs))
    for j in range(N_polynomial_coefficients):
        result += params[j] * xs ** j
    return result


average_diffs_arPLS = []
average_diffs_rb = []
average_diffs_unet = []
average_ssim_arPLS = []
average_ssim_rb = []
average_ssim_unet = []

bestfit_filenames = glob("bestfit_heuristic_*.pickle")

for bestfit_filename in bestfit_filenames:

    with open(bestfit_filename, "rb") as file:
        bestfit_parameters, raw_files_used_for_fitting = pickle.load(file)

    indices_used_for_fitting = [
        raw_files.index(raw_file) for raw_file in raw_files_used_for_fitting
    ]

    xs_current = [xs[l] for l in range(0, len(xs)) if l not in indices_used_for_fitting]
    ys_current = [ys[l] for l in range(0, len(xs)) if l not in indices_used_for_fitting]
    parameters_current = [
        parameters[l] for l in range(0, len(xs)) if l not in indices_used_for_fitting
    ]
    scores_current = [
        scores[l] for l in range(0, len(xs)) if l not in indices_used_for_fitting
    ]
    raw_files_current = [
        raw_files[l] for l in range(0, len(xs)) if l not in indices_used_for_fitting
    ]

    diffs_arPLS = []
    diffs_rb = []
    diffs_unet = []

    ssims_arPLS = []
    ssims_rb = []
    ssims_unet = []

    for i in range(len(xs_current)):

        print(raw_files_current[i])

        if do_plot:
            plt.figure(
                figsize=(figure_double_width_pub * 0.95, figure_double_width_pub * 0.7,)
            )

        print(f"{i} of {len(xs_current)}, score {scores_current[i]}")

        if do_plot:
            plt.plot(xs_current[i], ys_current[i], label="Input")
            # plt.plot(xs_current[i], ys_current[i] - predictions, label="UNet")

        predictions = model_unet.predict(
            np.expand_dims(np.expand_dims(ys_current[i], 0), -1)
        )[0, :, 0]

        result_unet_fit = curve_fit(
            background_fit,
            xs_current[i],
            ys_current[i] - predictions,
            p0=[0.0] * N_polynomial_coefficients,
        )[0]
        ys_unet_fit = background_fit(xs_current[i], *result_unet_fit)

        if do_plot:
            plt.plot(xs_current[i], ys_unet_fit, label="U-Net polynomial fit")

        target_y = np.zeros(len(xs_current[i]))
        for j in range(N_polynomial_coefficients):
            target_y += parameters_current[i][j] * xs_current[i] ** j

        if do_plot:
            plt.plot(xs_current[i], target_y, label="Rietveld (target)")

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
                    xs_current[i],
                    ys_current[i],
                    bestfit_parameters[method][0],
                    bestfit_parameters[method][1],
                    # 24.43999445,
                    # 0.18473314,
                )
            else:
                result_heuristic = baseline_arPLS(
                    ys_current[i],
                    ratio=ratio_fixed,
                    # ratio=10 ** (-5),
                    lam=10 ** bestfit_parameters[method][0],
                    # lam=10 ** (7.5),
                )

            if do_plot:
                plt.plot(xs_current[i], result_heuristic, label=method)

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
                figsize=(figure_double_width_pub * 0.95, figure_double_width_pub * 0.7,)
            )
            plt.plot(xs_current[i], ys_current[i], label="Input")
            plt.plot(xs_current[i], predictions, label="U-Net result")
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

    if False:
        print(
            "Median difference / score UNet (mse, ssim):",
            np.median(diffs_unet),
            np.median(ssims_unet),
        )
        print(
            f"Median difference / score arPLS (mse, ssim):",
            np.median(diffs_arPLS),
            np.median(ssims_arPLS),
        )
        print(
            f"Median difference / score rb (mse, ssim):",
            np.median(diffs_rb),
            np.median(ssims_rb),
        )

    average_diffs_unet.append(np.average(diffs_unet))
    average_diffs_arPLS.append(np.average(diffs_arPLS))
    average_diffs_rb.append(np.average(diffs_rb))

    average_ssim_unet.append(np.average(ssims_unet))
    average_ssim_arPLS.append(np.average(ssims_arPLS))
    average_ssim_rb.append(np.average(ssims_rb))

    counter_unet_better_than_arPLS = 0
    for i in range(len(diffs_unet)):
        if diffs_unet[i] < diffs_arPLS[i]:
            counter_unet_better_than_arPLS += 1
    print(
        f"{counter_unet_better_than_arPLS} of {len(diffs_unet)} UNet better than arPLS"
    )

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

print("Final (averaged over N runs) results:")

print(
    "Average difference / score UNet (mse, ssim):",
    np.average(average_diffs_unet),
    f"(+- {np.std(average_diffs_unet)})",
    np.average(average_ssim_unet),
    f"(+- {np.std(average_ssim_unet)})",
)
print(
    f"Average difference / score arPLS (mse, ssim):",
    np.average(average_diffs_arPLS),
    f"(+- {np.std(average_diffs_arPLS)})",
    np.average(average_ssim_arPLS),
    f"(+- {np.std(average_ssim_arPLS)})",
)
print(
    f"Average difference / score rb (mse, ssim):",
    np.average(average_diffs_rb),
    f"(+- {np.std(average_diffs_rb)})",
    np.average(average_ssim_rb),
    f"(+- {np.std(average_ssim_rb)})",
)
