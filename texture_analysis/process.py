import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter


test_file = "data/XRD-8 component systems.csv"
df = pd.read_csv(test_file, sep=";")

xs = np.array(df.iloc[:, list(range(0, len(df.columns.values), 2))])
ys = np.array(df.iloc[:, list(range(1, len(df.columns.values), 2))])

names = df.columns.values[1::2]

xs_test = xs[:, 0]
ys_test = ys[:, 0]


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def fit_baseline(xs, ys):

    ys_baseline = baseline_als(ys, 5 * 10 ** 6, 0.01)

    plt.figure()
    plt.plot(xs, ys)
    plt.plot(xs, ys_baseline)
    plt.title("Raw data with baseline fit")

    return ys_baseline


ys_baseline = fit_baseline(xs_test, ys_test)

ys_baseline_removed = ys_test - ys_baseline

plt.show()

# TODO: smooth after baseline removal

"""
plt.figure()
plt.plot(xs, ys_baseline_removed)

ys_smoothed = savgol_filter(ys_baseline_removed, 101, 2)

plt.figure()
plt.plot(xs_test, ys_smoothed)
plt.title("Smoothed with baseline fit")
plt.show()
"""
