# Source: https://stackoverflow.com/questions/67466241/full-algorithm-math-of-natural-cubic-splines-computation-in-python

import numpy as np
import numba

# Solves linear system given by Tridiagonal Matrix
# Helper for calculating cubic splines
@numba.njit
def tri_diag_solve(A, B, C, F):
    n = B.size
    assert A.ndim == B.ndim == C.ndim == F.ndim == 1 and (
        A.size == B.size == C.size == F.size == n
    )  # , (A.shape, B.shape, C.shape, F.shape)
    Bs, Fs = np.zeros_like(B), np.zeros_like(F)
    Bs[0], Fs[0] = B[0], F[0]
    for i in range(1, n):
        Bs[i] = B[i] - A[i] / Bs[i - 1] * C[i - 1]
        Fs[i] = F[i] - A[i] / Bs[i - 1] * Fs[i - 1]
    x = np.zeros_like(B)
    x[-1] = Fs[-1] / Bs[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (Fs[i] - C[i] * x[i + 1]) / Bs[i]
    return x


# Calculate cubic spline params
@numba.njit
def calc_spline_params(x, y):
    a = y
    h = np.diff(x)
    c = np.concatenate(
        (
            np.zeros((1,), dtype=y.dtype),
            np.append(
                tri_diag_solve(
                    h[:-1],
                    (h[:-1] + h[1:]) * 2,
                    h[1:],
                    ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1]) * 3,
                ),
                0,
            ),
        )
    )
    d = np.diff(c) / (3 * h)
    b = (a[1:] - a[:-1]) / h + (2 * c[1:] + c[:-1]) / 3 * h
    return a[1:], b, c[1:], d


# Spline value calculating function, given params and "x"
@numba.njit
def func_spline(x, ix, x0, a, b, c, d):
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx


@numba.njit
def searchsorted_merge(a, b, sort_b):
    ix = np.zeros((len(b),), dtype=np.int64)
    if sort_b:
        ib = np.argsort(b)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] < (b[ib[pb]] if sort_b else b[pb]):
            pa += 1
        else:
            ix[pb] = pa
            pb += 1
    return ix


# Compute piece-wise spline function for "x" out of sorted "x0" points
@numba.njit
def piece_wise_spline(x, x0, a, b, c, d):
    xsh = x.shape
    x = x.ravel()
    # ix = np.searchsorted(x0[1 : -1], x)
    ix = searchsorted_merge(x0[1:-1], x, False)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = y.reshape(xsh)
    return y


@numba.njit
def spline_numba(x0, y0, xs):
    a, b, c, d = calc_spline_params(x0, y0)
    return piece_wise_spline(xs, x0, a, b, c, d)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    samples_xs = np.array([1, 2, 3, 4, 5], dtype=float)
    samples_ys = np.array([1, 4, 9, 22, 4], dtype=float)

    xs = np.linspace(1, 5, 100)

    values = spline_numba(samples_xs, samples_ys)(xs)

    plt.plot(xs, values)
    plt.plot(samples_xs, samples_ys)
    plt.show()
