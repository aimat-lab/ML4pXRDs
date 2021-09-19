import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import csv
from scipy.signal import savgol_filter
import os

# both of the following functions are taken from https://stackoverflow.com/questions/29156532/python-baseline-correction-library
# TODO: Maybe use this library for baseline removal in the future: https://github.com/StatguyUser/BaselineRemoval (algorithm should be the same, though)
def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print("Maximum number of iterations exceeded")
            break

    if full_output:
        info = {"num_iter": count, "stop_criterion": crit}
        return z, d, info
    else:
        return z


def baseline_als(y, lam=3 * 10 ** 3, p=0.3, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def fit_baseline(xs, ys):

    # TODO: Maybe find even better parameters for this
    # ys_baseline = baseline_als(ys, 3 * 10 ** 3, 0.3)
    ys_baseline = baseline_arPLS(ys, 0.1, lam=10 ** 7)

    plt.subplot(221)
    plt.plot(xs, ys, rasterized=True)
    plt.plot(xs, ys_baseline, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")

    plt.title("Raw data with baseline fit")

    return ys_baseline


def gaussian(x, a, x0, sigma):
    return (
        a
        * 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    )


# TODO: Maybe at some point fit a mixture of Lorentzian and Gaussian instead of only Gaussian
def fit_gaussian(peak_index, xs, ys, ys_smoothed):

    # find first index where function increases, again (to the left and to the right)

    last_entry = ys_smoothed[peak_index]
    for i in range(peak_index + 1, len(ys_smoothed)):
        if ys_smoothed[i] > last_entry:
            if xs[i] - xs[peak_index] > 0.5:
                break
        else:
            last_entry = ys_smoothed[i]
    right = i

    last_entry = ys_smoothed[peak_index]
    for i in reversed(range(0, peak_index)):
        if ys_smoothed[i] > last_entry:
            if xs[peak_index] - xs[i] > 0.5:
                break
        else:
            last_entry = ys_smoothed[i]
    left = i

    plt.subplot(223)
    plt.scatter([xs[left], xs[right]], [ys_smoothed[left], ys_smoothed[right]], c="r", rasterized=True)
    plt.subplot(222)

    fit = curve_fit(
        gaussian,
        xs[left : right + 1],
        ys[left : right + 1],
        p0=[100, xs[peak_index], 0.3],
    )

    plt.plot(xs, gaussian(xs, *fit[0]), rasterized=True)

    return fit[0]  # return the fit parameters


def peak_to_str(fit_results, height_raw, height_after_removal):
    return "Area: {}\nMean: {}\nSigma: {}\nFWHM: {}\nHeight raw: {}\nHeight after removal: {}".format(
        str(fit_results[0]),
        str(fit_results[1]),
        str(fit_results[2]),
        str(2 * np.sqrt(2 * np.log(2)) * fit_results[2]),
        height_raw,
        height_after_removal,
    )


def get_real_maximum(xs, ys, index, tollerance_left_right):

    step = xs[1] - xs[0]
    no_steps = int(tollerance_left_right / step)

    ys_current = ys[index - no_steps : index + no_steps + 1]

    true_maximum_index = np.argmax(ys_current)

    return (true_maximum_index + index - no_steps, ys_current[true_maximum_index])


data_file_path = "data/XRD_6_component_systems.csv"
df = pd.read_csv(data_file_path, sep=";")

xs = np.array(df.iloc[:, list(range(0, len(df.columns.values), 2))])
ys = np.array(df.iloc[:, list(range(1, len(df.columns.values), 2))])

names = df.columns.values[1::2]

ratios = []
properties_peak_0 = []  # biggest peak
properties_peak_1 = []  # second biggest peak

for i in range(0, xs.shape[1]):
    # for i in range(0, 10):

    fig, ax = plt.subplots(2, 2)

    fig.canvas.manager.set_window_title("Plot {} of {}".format(i + 1, xs.shape[1]))
    plt.suptitle("Sample " + names[i])    

    ax[1, 1].set_axis_off()
    fig.set_size_inches(18.5, 10.5)

    print("Processing {} of {}".format(i + 1, xs.shape[1]))

    xs_current = xs[:, i]
    ys_current = ys[:, i]

    ys_baseline = fit_baseline(xs_current, ys_current)

    ys_baseline_removed = ys_current - ys_baseline

    plt.subplot(222)
    plt.plot(xs, ys_baseline_removed, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")
    plt.title("Raw data with baseline removed and gauß fits")

    # use moving average smoothing
    # window_width = int(0.5 / (xs_current[1] - xs_current[0]))
    window_width = 50
    # print("Window width: {}".format(window_width))
    ys_smoothed = np.convolve(
        ys_baseline_removed, np.ones(window_width) / window_width, mode="same"
    )

    (
        peaks,
        props,
    ) = find_peaks(  # TODO: Change these parameters when doing more sophisticated analysis
        ys_smoothed,
        distance=0.5 / (xs_current[1] - xs_current[0]),
        prominence=20,
        height=10,
    )

    plt.subplot(223)
    plt.plot(xs_current, ys_smoothed, rasterized=True)
    plt.scatter(xs_current[peaks], ys_smoothed[peaks], c="r", rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")
    plt.title("Smoothed, baseline removed, with marked peaks")

    plt.subplot(222)
    parameters = []
    for peak in peaks:
        para = fit_gaussian(peak, xs_current, ys_baseline_removed, ys_smoothed)
        parameters.append(para)
    zipped_sorted = sorted(
        zip(parameters, peaks), key=lambda x: x[0][0]
    )  # sort by area

    text = "\nBiggest peak:\n"
    if len(peaks) > 0:

        real_maximum_raw = get_real_maximum(
            xs_current, ys_current, zipped_sorted[-1][1], 0.3
        )
        real_maximum_baseline_removed = get_real_maximum(
            xs_current, ys_baseline_removed, zipped_sorted[-1][1], 0.3
        )
        # real_maximum_smoothed = get_real_maximum(
        #    xs_current, ys_smoothed, zipped_sorted[-1][1], 0.3
        # )

        text += (
            peak_to_str(
                zipped_sorted[-1][0],
                real_maximum_raw[1],
                real_maximum_baseline_removed[1],
                # real_maximum_smoothed[1],
            )
            + "\n"
        )
        properties_peak_0.append(
            (
                zipped_sorted[-1][0][0],
                zipped_sorted[-1][0][1],
                zipped_sorted[-1][0][2],
                2 * np.sqrt(2 * np.log(2)) * zipped_sorted[-1][0][2],
                real_maximum_raw[1],
                real_maximum_baseline_removed[1],
                # real_maximum_smoothed[1],
            )
        )
    else:
        text += "Not found\n"
        properties_peak_0.append(("None", "None", "None", "None", "None", "None"))

    text += "\n\nSecond biggest peak:\n"
    if len(peaks) > 1:

        real_maximum_raw = get_real_maximum(
            xs_current, ys_current, zipped_sorted[-2][1], 0.3
        )
        real_maximum_baseline_removed = get_real_maximum(
            xs_current, ys_baseline_removed, zipped_sorted[-2][1], 0.3
        )
        # real_maximum_smoothed = get_real_maximum(
        #    xs_current, ys_smoothed, zipped_sorted[-2][1], 0.3
        # )

        text += (
            peak_to_str(
                zipped_sorted[-2][0],
                real_maximum_raw[1],
                real_maximum_baseline_removed[1],
                # real_maximum_smoothed[1],
            )
            + "\n"
        )
        properties_peak_1.append(
            (
                zipped_sorted[-2][0][0],
                zipped_sorted[-2][0][1],
                zipped_sorted[-2][0][2],
                2 * np.sqrt(2 * np.log(2)) * zipped_sorted[-2][0][2],
                real_maximum_raw[1],
                real_maximum_baseline_removed[1],
                # real_maximum_smoothed[1],
            )
        )
    else:
        text += "Not found\n"
        properties_peak_1.append(("None", "None", "None", "None", "None", "None"))

    text += "\nRatio of the two biggest peaks:\n"
    if len(peaks) > 1:

        ratio = zipped_sorted[-1][0][0] / zipped_sorted[-2][0][0]
        text += str(ratio) + "\n"
        ratios.append(ratio)

    else:

        text += "Found less than two peaks.\n"
        ratios.append("None")

    ax[1, 1].text(
        0.5, 0.5, text, horizontalalignment="center", verticalalignment="center"
    )

    plt.savefig("plots/" + names[i] + ".pdf", dpi=300)
    # plt.show()

with open("ratios.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=";")

    # transpose the lists:
    properties_peak_0 = list(map(list, zip(*properties_peak_0)))
    properties_peak_1 = list(map(list, zip(*properties_peak_1)))

    header = [
        "Sample name",
        "Ratio",
        "Peak_0 area",
        "Peak_0 mean",
        "Peak_0 sigma",
        "Peak_0 FWHM",
        "Peak_0 height raw",
        "Peak_0 height after removal",
        "Peak_1 area",
        "Peak_1 mean",
        "Peak_1 sigma",
        "Peak_1 FWHM",
        "Peak_1 height raw",
        "Peak_1 height after removal",
    ]

    # print(names)
    # print(ratios)
    # print(*properties_peak_0)
    # print(*properties_peak_1)

    data = zip(names, ratios, *properties_peak_0, *properties_peak_1)

    csv_writer.writerow(header)
    csv_writer.writerows(data)

os.system("pdfunite plots/*.pdf plots/all.pdf")