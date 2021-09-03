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


test_file = "data/XRD_6_component_systems.csv"
df = pd.read_csv(test_file, sep=";")

xs = np.array(df.iloc[:, list(range(0, len(df.columns.values), 2))])
ys = np.array(df.iloc[:, list(range(1, len(df.columns.values), 2))])

names = df.columns.values[1::2]


def fit_baseline(xs, ys, num_of_samples):

    # ys_baseline = baseline_als(ys, 3 * 10 ** 3, 0.3)
    ys_baseline = baseline_arPLS(ys, 0.1, lam=10 ** 7)

    plt.figure("Plot {} of {}".format(i + 1, num_of_samples), figsize=(16, 16))
    plt.subplot(221)
    plt.plot(xs, ys)
    plt.plot(xs, ys_baseline)
    plt.title("Raw data with baseline fit")

    return ys_baseline


for i in range(0, xs.shape[1]):

    # use this for testing
    xs_current = xs[:, i]
    ys_current = ys[:, i]

    ys_baseline = fit_baseline(xs_current, ys_current, xs.shape[1])

    ys_baseline_removed = ys_current - ys_baseline

    plt.subplot(222)
    plt.plot(xs, ys_baseline_removed)
    plt.title("Raw data with baseline removed")

    # use moving average smoothing
    window_width = 50
    cumsum_vec = np.cumsum(np.insert(ys_baseline_removed, 0, 0))
    ys_smoothed = (
        cumsum_vec[window_width:] - cumsum_vec[:-window_width]
    ) / window_width

    # TODO: Don't do this
    cumsum_vec = np.cumsum(np.insert(xs_current, 0, 0))
    xs_smoothed = (
        cumsum_vec[window_width:] - cumsum_vec[:-window_width]
    ) / window_width

    peaks, props = find_peaks(
        ys_smoothed,
        distance=0.5 / (xs_smoothed[1] - xs_smoothed[0]),
        prominence=20,
        height=10,
    )

    plt.subplot(223)
    plt.plot(xs_smoothed, ys_smoothed)
    plt.scatter(xs_smoothed[peaks], ys_smoothed[peaks], color="r")
    plt.title("Smoothed, baseline removed, with marked peaks and gauss fit")

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    def fit_gaussian(peak_index, xs, ys, min_distance=0.5):

        # find first index where function increases, again (to the left and to the right)

        last_entry = ys[peak_index]
        for i in range(peak_index + 1, len(ys)):
            if ys[i] > last_entry:
                if xs[i] - xs[peak_index] > 0.5:
                    break
            else:
                last_entry = ys[i]
        right = i

        last_entry = ys[peak_index]
        for i in reversed(range(0, peak_index)):
            if ys[i] > last_entry:
                if xs[peak_index] - xs[i] > 0.5:
                    break
            else:
                last_entry = ys[i]
        left = i

        plt.scatter([xs[left], xs[right]], [ys[left], ys[right]])

        fit = curve_fit(
            gaussian,
            xs[left : right + 1],
            ys[left : right + 1],
            p0=[100, xs[peak_index], 0.3],
        )

        plt.plot(xs, gaussian(xs, *fit[0]))

    for peak in peaks:
        fit_gaussian(peak, xs_smoothed, ys_smoothed)

    plt.show()

