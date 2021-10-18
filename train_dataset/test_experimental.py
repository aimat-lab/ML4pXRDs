import numpy as np
import pandas as pd
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from cv2_rolling_ball import subtract_background_rolling_ball
from scipy import interpolate as ip
from scipy.signal import find_peaks, filtfilt

remove_background = True
removal_strategie = "arPLS"  # currently only arPLS is supported
remove_noise = False  # currently noise removal is not yet supported
model_path = ""  # path to the model to test on
experimental_file = ""

# baseline_arPLS parameters:
current_ratio = -2.37287
current_lambda = 7.311915

arPLS_ratio = 10 ** current_ratio
arPLS_lam = 10 ** current_lambda
arPLS_niter = 100

bg_subtraction_algorithm = "rollingball"  # arPLS, rollingball
tune_arPLS_parameters = True

# both of the following functions are taken from https://stackoverflow.com/questions/29156532/python-baseline-correction-library
# TODO: Maybe use this library for baseline removal in the future: https://github.com/StatguyUser/BaselineRemoval (algorithm should be the same, though)
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


def process_spectrum(x, y):

    ## Fit to 4,501 values as to be compatible with CNN
    f = ip.CubicSpline(x, y)
    xs = np.linspace(10, 50, 2000)
    ys = f(xs)

    ## Smooth out noise
    # Smoothing parameters defined by n
    n = 20
    b = [1.0 / n] * n
    a = 1

    # Filter noise
    ys = filtfilt(b, a, ys)

    ## Map to integers in range 0 to 255 so cv2 can handle
    ys = [val - min(ys) for val in ys]
    ys = [255 * (val / max(ys)) for val in ys]
    ys = [int(val) for val in ys]

    ## Perform baseline correction with cv2
    pixels = []
    for q in range(10):
        pixels.append(ys)
    pixels = np.array(pixels)
    img, background = subtract_background_rolling_ball(
        pixels, 800, light_background=False, use_paraboloid=True, do_presmooth=False
    )
    yb = np.array(background[0])
    ys = np.array(ys) - yb

    ## Normalize from 0 to 100
    ys = np.array(ys) - min(ys)
    ys = list(100 * np.array(ys) / max(ys))

    return ys


def load_experimental_data(path, mode="HEO"):
    if mode == "HEO":

        data = pd.read_csv(path, delimiter=",", skiprows=1)
        xs = np.array(data.iloc[:, list(range(0, len(data.columns.values), 2))])
        ys = np.array(data.iloc[:, list(range(1, len(data.columns.values), 2))])

        return (xs, ys)

    else:

        raise Exception("Mode for loading experimental data not supported.")


if __name__ == "__main__":

    xs, ys = load_experimental_data("exp_data/XRDdata.csv")

    print(xs[0])
    print(ys[0])
    for i in range(0, xs.shape[1]):

        if bg_subtraction_algorithm == "arPLS":
            if not tune_arPLS_parameters:

                baseline = baseline_arPLS(ys[:, i])

                plt.plot(xs[:, i], ys[:, i])
                plt.plot(xs[:, i], baseline)
                plt.plot(xs[:, i], ys[:, i] - baseline)
                plt.plot(xs[:, i], [0] * xs.shape[0])
                plt.show()

            else:

                fig, ax = plt.subplots()

                axwave1 = plt.axes([0.17, 0.06, 0.65, 0.03])  # slider dimensions
                axwave2 = plt.axes([0.17, 0, 0.65, 0.03])  # slider dimensions

                slider_ratio = Slider(
                    axwave1, "Event No. 1", -3, -1, valinit=current_ratio, valfmt="%E"
                )  # 1
                slider_lam = Slider(
                    axwave2, "Event No. 2", 2, 9, valinit=current_lambda, valfmt="%E"
                )  # 2

                def update_wave(val):
                    value1 = 10 ** slider_ratio.val
                    slider_ratio.valtext.set_text(f"{value1:.5E} {slider_ratio.val}")
                    value2 = 10 ** slider_lam.val
                    slider_lam.valtext.set_text(f"{value2:.5E} {slider_lam.val}")

                    ax.cla()
                    baseline = baseline_arPLS(ys[:, i], value1, value2)
                    ax.plot(xs[:, i], 0.4 + ys[:, i])
                    ax.plot(xs[:, i], 0.4 + baseline)
                    ax.plot(xs[:, i], ys[:, i] - baseline)
                    ax.plot(xs[:, i], [0] * len(xs[:, i]))
                    fig.canvas.draw_idle()

                baseline = baseline_arPLS(
                    ys[:, i], 10 ** current_ratio, 10 ** current_lambda
                )
                ax.plot(xs[:, i], 0.4 + ys[:, i])
                ax.plot(
                    xs[:, i], 0.4 + baseline,
                )
                ax.plot(xs[:, i], ys[:, i] - baseline)
                ax.plot(xs[:, i], [0] * len(xs[:, i]))

                slider_ratio.on_changed(update_wave)
                slider_lam.on_changed(update_wave)

                plt.show()

                current_lambda = slider_lam.val
                current_ratio = slider_ratio.val

        elif bg_subtraction_algorithm == "rollingball":

            formatted = process_spectrum(xs[:, i], ys[:, i])

            plt.plot(xs[:, i], 30 + ys[:, i] * 100)
            plt.plot(np.linspace(10, 50, 2000), formatted)
            plt.plot(xs[:, i], [0] * len(xs[:, i]))
            plt.show()

