import numpy as np
import pandas as pd
from scipy.sparse import linalg
from numpy.linalg import norm

remove_background = True
removal_strategie = "arPLS"  # currently only arPLS is supported
remove_noise = False  # currently noise removal is not yet supported
model_path = ""  # path to the model to test on
experimental_file = ""

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


def fit_baseline(xs, ys):

    # TODO: Maybe find even better parameters for this
    ys_baseline = baseline_arPLS(
        ys, ratio=arPLS_ratio, lam=arPLS_lam, niter=arPLS_niter
    )

    plt.subplot(221)
    plt.plot(xs, ys, rasterized=True)
    plt.plot(xs, ys_baseline, rasterized=True)
    plt.xlabel(r"$ 2 \theta \, / \, ° $")
    plt.ylabel("Intensity")

    plt.title("Raw data with baseline fit")

    return ys_baseline
