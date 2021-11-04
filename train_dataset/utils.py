import numpy as np
import pandas as pd
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import interpolate as ip
from scipy.signal import filtfilt
from skimage import restoration

# baseline_arPLS parameters:
current_ratio = -2.37287
current_lambda = 7.311915
arPLS_ratio = 10 ** current_ratio
arPLS_lam = 10 ** current_lambda
arPLS_niter = 100

rolling_ball_sphere_x = 6.619
rolling_ball_sphere_y = 0.3

# the following function is taken from https://stackoverflow.com/questions/29156532/python-baseline-correction-library
def baseline_arPLS(y, ratio=None, lam=None, niter=None, full_output=False):

    ratio = arPLS_ratio if not ratio else ratio
    lam = arPLS_lam if not lam else lam
    niter = arPLS_niter if not niter else niter

    print(f"Ratio {ratio:.5E} lam {lam:.5E}")

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


def rolling_ball(
    x, y, sphere_x=15, sphere_y=0.3, min_x=10, max_x=50, n_xs=5000, ax=None
):

    if ax == None:
        ax = plt.gca()

    y = y / np.max(y)

    # ax.plot(x, y + 3, label="Raw")

    f = ip.CubicSpline(x, y, bc_type="natural")

    xs = np.linspace(min_x, max_x, n_xs)
    ys = f(xs)

    # ax.plot(xs, ys + 2, label="Cubic spline")
    # ax.plot(xs, ys + 1, label="Cubic spline")

    ## Smooth out noise
    # Smoothing parameters defined by n
    n = 25
    b = [1.0 / n] * n
    a = 1

    # Filter noise
    ys = filtfilt(b, a, ys, padtype="constant")

    ax.plot(xs, np.array(ys) + 1, label="Noise filtered")

    width = sphere_x / (xs[1] - xs[0])
    # ax.add_patch(Ellipse(xy=(10, 0.6), width=sphere_x, height=sphere_y))
    yb = restoration.rolling_ball(
        ys, kernel=restoration.ellipsoid_kernel((width,), sphere_y)
    )

    ax.plot(xs, yb + 1, label="Baseline")

    ys = ys - yb
    # ys[ys < 0.009] = 0  # thresholding
    ax.plot(xs, ys, label="Baseline removed")

    ax.plot(xs, [0] * len(xs))

    ax.legend()

    return ys - yb


def load_experimental_data(path, mode="HEO"):
    if mode == "HEO":

        data = pd.read_csv(path, delimiter=",", skiprows=1)
        xs = np.array(data.iloc[:, list(range(0, len(data.columns.values), 2))])
        ys = np.array(data.iloc[:, list(range(1, len(data.columns.values), 2))])

        return (xs, ys)

    else:

        raise Exception("Mode for loading experimental data not supported.")
