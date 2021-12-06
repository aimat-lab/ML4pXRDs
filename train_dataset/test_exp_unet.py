import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate as ip
from UNet_1DCNN import UNet
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import interpolate as ip
from scipy.signal import find_peaks, filtfilt
from skimage import data, restoration, util
from matplotlib.patches import Ellipse
from functools import partial
import pybaselines

mode = "removal"  # possible: info and removal (currently only removal)
to_test = "removal_03-12-2021_16-48-30_UNetPP"

show_comparison = False


def load_experimental_data(loading_mode="classification"):
    if loading_mode == "classification":

        path = "exp_data/XRDdata_classification.csv"
        data = pd.read_csv(path, delimiter=",", skiprows=1)
        xs = np.array(data.iloc[:, list(range(0, len(data.columns.values), 2))])
        ys = np.array(data.iloc[:, list(range(1, len(data.columns.values), 2))])

        return (xs, ys)

    elif loading_mode == "texture":  # TODO: watch out: these have a different range

        data_file_path = "exp_data/XRD_6_component_systems_repeat.csv"
        df = pd.read_csv(data_file_path, sep=";")

        x = np.array(df.iloc[:, 0])
        xs = np.repeat(x[:, np.newaxis], len(df.columns.values) - 1, axis=1)
        ys = np.array(df.iloc[:, list(range(1, len(df.columns.values)))])

        return (xs, ys)

    else:

        raise Exception("Mode for loading experimental data not supported.")


# baseline_arPLS parameters:
current_ratio_exponent = -2.37287
current_lambda_exponent = 7.311915
arPLS_ratio = 10 ** current_ratio_exponent
arPLS_lam = 10 ** current_lambda_exponent
arPLS_niter = 100

# rolling ball parameters:
rolling_ball_sphere_x = 6.619
rolling_ball_sphere_y = 0.3

# wavelet parameters
# num_std=0..5, min_length=2..100
wavelet_num_std = 1.0
wavelet_min_length = 2.0

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


def rolling_ball(xs, ys, sphere_x=15, sphere_y=0.3, ax=None):

    if ax == None:
        ax = plt.gca()

    ys = ys - np.min(ys)
    ys = ys / np.max(ys)

    # ax.plot(x, y + 3, label="Raw")
    # ax.plot(xs, ys + 2, label="Cubic spline")
    # ax.plot(xs, ys + 1, label="Cubic spline")

    ## Smooth out noise
    # Smoothing parameters defined by n
    n = 25
    b = [1.0 / n] * n
    a = 1

    # Filter noise
    ys = filtfilt(b, a, ys, padtype="constant")

    # ax.plot(xs, np.array(ys) + 1, label="Noise filtered")

    width = sphere_x / (xs[1] - xs[0])
    # height = sphere_y / (ys[1] - ys[0])
    height = sphere_y
    # ax.add_patch(Ellipse(xy=(10, 0.6), width=sphere_x, height=sphere_y))
    yb = restoration.rolling_ball(
        ys,
        kernel=restoration.ellipsoid_kernel((width,), height),
    )

    # ax.plot(xs, yb + 1, label="Baseline")

    # ys = ys - yb
    # ys[ys < 0.009] = 0  # thresholding
    # ax.plot(xs, ys, label="Baseline removed")

    # ax.plot(xs, [0] * len(xs))

    # ax.legend()

    return yb


def update_wave(
    val,
    slider_1_rb,
    slider_2_rb,
    slider_1_arPLS,
    slider_2_arPLS,
    slider_1_wavelet,
    slider_2_wavelet,
    xs,
    ys,
    ax,
    fig,
    do_remove,
):

    global current_lambda_exponent
    global current_ratio_exponent
    global rolling_ball_sphere_x
    global rolling_ball_sphere_y
    global wavelet_num_std
    global wavelet_min_length

    current_ratio_exponent = slider_1_arPLS.val
    current_lambda_exponent = slider_2_arPLS.val
    rolling_ball_sphere_x = slider_1_rb.val
    rolling_ball_sphere_y = slider_2_rb.val
    current_wavelet_num_std = slider_1_wavelet.val
    current_wavelet_min_length = slider_2_wavelet.val

    if do_remove:
        ax.lines[-1].remove()
        ax.lines[-1].remove()
        ax.lines[-1].remove()

    value1 = 10 ** slider_1_arPLS.val
    slider_1_arPLS.valtext.set_text(f"{value1:.3E} {slider_1_arPLS.val:.3f}")
    value2 = 10 ** slider_2_arPLS.val
    slider_2_arPLS.valtext.set_text(f"{value2:.3E} {slider_2_arPLS.val:.3f}")

    baseline = baseline_arPLS(ys, value1, value2)
    ax.plot(xs, baseline, label="arPLS", c="b")

    background = rolling_ball(
        xs,
        ys,
        sphere_x=slider_1_rb.val,
        sphere_y=slider_2_rb.val,
        ax=ax,
    )
    ax.plot(xs, background, label="Rolling ball", c="k")

    baseline_wavelet = pybaselines.classification.cwt_br(
        ys, num_std=current_wavelet_num_std, min_length=int(current_wavelet_min_length)
    )
    ax.plot(xs, baseline_wavelet[0], label="Wavelet transform", c="m")

    ax.legend()

    fig.canvas.draw_idle()


def plot_heuristic_fit(xs, ys):

    fig = plt.gcf()
    ax = plt.gca()

    sliders = []
    for method in ["rolling_ball", "arPLS", "wavelet"]:

        fig.subplots_adjust(
            left=0.05, bottom=0.5, right=0.95, top=0.98, wspace=0.05, hspace=0.05
        )

        if method == "rolling_ball":
            bottom_1 = 0.30
            min_1 = 0
            max_1 = 100
            valinit_1 = rolling_ball_sphere_x

            bottom_2 = 0.24
            min_2 = 0
            max_2 = 3
            valinit_2 = rolling_ball_sphere_y

            valfmt = "%1.3f"

        elif method == "arPLS":
            bottom_1 = 0.18
            min_1 = -3
            max_1 = -1
            valinit_1 = current_ratio_exponent

            bottom_2 = 0.12
            min_2 = 2
            max_2 = 9
            valinit_2 = current_lambda_exponent

            valfmt = "%E"

        elif method == "wavelet":

            # num_std=0..5, min_length=2..100
            bottom_1 = 0.06
            min_1 = 0
            max_1 = 5
            valinit_1 = wavelet_num_std

            bottom_2 = 0.00
            min_2 = 2
            max_2 = 100
            valinit_2 = wavelet_min_length

            valfmt = "%1.3f"

        else:
            raise Exception("Baseline detection method not recognized.")

        axwave1 = plt.axes([0.17, bottom_1, 0.65, 0.03])  # slider dimensions
        axwave2 = plt.axes(
            [0.17, bottom_2, 0.65, 0.03]
        )  # slider dimensions # left, bottom, width, height

        slider_1 = Slider(
            axwave1,
            "Event No. 1",
            min_1,
            max_1,
            valinit=valinit_1,
            valfmt=valfmt,
        )  # 1
        slider_2 = Slider(
            axwave2,
            "Event No. 2",
            min_2,
            max_2,
            valinit=valinit_2,
            valfmt=valfmt,
        )  # 2

        sliders.append(slider_1)
        sliders.append(slider_2)

    update_wave(
        None,
        sliders[0],
        sliders[1],
        sliders[2],
        sliders[3],
        sliders[4],
        sliders[5],
        xs,
        ys,
        ax,
        fig,
        do_remove=False,
    )

    for slider in sliders:
        slider.on_changed(
            partial(
                update_wave,
                slider_1_rb=sliders[0],
                slider_2_rb=sliders[1],
                slider_1_arPLS=sliders[2],
                slider_2_arPLS=sliders[3],
                slider_1_wavelet=sliders[4],
                slider_2_wavelet=sliders[5],
                xs=xs,
                ys=ys,
                ax=ax,
                fig=fig,
                do_remove=True,
            )
        )

    plt.show()


if __name__ == "__main__":

    xs_exp, ys_exp = load_experimental_data(loading_mode="classification")

    model_pre = keras.models.load_model("unet/" + to_test + "/final")

    if mode == "removal":
        model = model_pre
    else:
        model = keras.Sequential([model_pre, keras.layers.Activation("sigmoid")])

    """
    with open("unet/removal_cps/scaler", "rb") as file:
        sc = pickle.load(file)
    """

    for i in range(0, xs_exp.shape[1]):

        current_xs = xs_exp[:, i][0:2672]
        current_ys = ys_exp[:, i][0:2672]

        # plt.plot(current_xs, current_ys)
        # plt.plot(current_xs[0:2672], current_ys[0:2672])
        # plt.show()
        # continue

        # plt.plot(current_xs, current_ys)
        # plt.show()

        # f = ip.CubicSpline(current_xs, current_ys, bc_type="natural")

        ys = current_ys
        ys -= np.min(ys)
        ys = ys / np.max(ys)

        plt.plot(current_xs, np.zeros(len(current_xs)))

        plt.plot(current_xs, ys, label="Experimental rescaled", zorder=2)

        # with open("unet/scaler", "rb") as file:
        #    scaler = pickle.load(file)
        # ys = scaler.transform(ys)

        # ys = np.expand_dims(sc.transform([ys]), axis=2)
        ys = np.expand_dims([ys], axis=2)

        if mode == "removal":

            corrected = model.predict(ys)

            plt.plot(
                current_xs,
                ys[0, :, 0] - corrected[0, :, 0],
                label="Background and noise",
                zorder=1,
                linestyle="dotted",
            )
            plt.plot(
                current_xs, corrected[0, :, 0], label="Corrected via U-Net", zorder=3
            )

        else:

            corrected = model.predict(ys)

            plt.scatter(
                current_xs,
                corrected[0, :, 0],
                label="Peak detection",
                s=3,
            )

        if show_comparison:
            plot_heuristic_fit(current_xs, ys[0, :, 0])
        else:
            plt.show()
