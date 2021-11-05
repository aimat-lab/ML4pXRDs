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
from skimage import data, restoration, util
from matplotlib.patches import Ellipse
import sys

sys.path.append("../")

import train_dataset.utils
from train_dataset.utils import load_experimental_data, baseline_arPLS, rolling_ball

remove_background = True
model_path = ""  # path to the model to test on
experimental_file = ""
bg_subtraction_algorithm = (
    "rollingball"  # arPLS (without noise filtering), rollingball (with noise filtering)
)
tune_parameters = True


if __name__ == "__main__":

    xs, ys = load_experimental_data("exp_data/XRDdata.csv")

    for i in range(0, xs.shape[1]):

        current_xs = xs[:, i]
        current_ys = ys[:, i]

        if remove_background:

            if bg_subtraction_algorithm == "arPLS":

                if not tune_parameters:

                    baseline = baseline_arPLS(current_ys)

                    plt.plot(current_xs, current_ys)
                    plt.plot(current_xs, baseline)
                    plt.plot(current_xs, current_ys - baseline)
                    plt.plot(current_xs, [0] * xs.shape[0])
                    plt.show()

                else:

                    fig, ax = plt.subplots()

                    axwave1 = plt.axes([0.17, 0.06, 0.65, 0.03])  # slider dimensions
                    axwave2 = plt.axes([0.17, 0, 0.65, 0.03])  # slider dimensions

                    slider_ratio = Slider(
                        axwave1,
                        "Event No. 1",
                        -3,
                        -1,
                        valinit=current_ratio,
                        valfmt="%E",
                    )  # 1
                    slider_lambda = Slider(
                        axwave2,
                        "Event No. 2",
                        2,
                        9,
                        valinit=current_lambda,
                        valfmt="%E",
                    )  # 2

                    def update_wave(val):
                        value1 = 10 ** slider_ratio.val
                        slider_ratio.valtext.set_text(
                            f"{value1:.5E} {slider_ratio.val}"
                        )
                        value2 = 10 ** slider_lambda.val
                        slider_lambda.valtext.set_text(
                            f"{value2:.5E} {slider_lambda.val}"
                        )

                        ax.cla()
                        baseline = baseline_arPLS(current_ys, value1, value2)
                        ax.plot(current_xs, 0.4 + current_ys)
                        ax.plot(current_xs, 0.4 + baseline)
                        ax.plot(current_xs, current_ys - baseline)
                        ax.plot(current_xs, [0] * len(current_xs))
                        fig.canvas.draw_idle()

                    baseline = baseline_arPLS(
                        current_ys, 10 ** current_ratio, 10 ** current_lambda
                    )
                    ax.plot(current_xs, 0.4 + current_ys)
                    ax.plot(
                        current_xs, 0.4 + baseline,
                    )
                    ax.plot(current_xs, current_ys - baseline)
                    ax.plot(current_xs, [0] * len(current_xs))

                    slider_ratio.on_changed(update_wave)
                    slider_lambda.on_changed(update_wave)

                    plt.show()

                    current_lambda = slider_ratio.val
                    current_ratio = slider_lambda.val

                current_ys = current_ys - baseline

            elif bg_subtraction_algorithm == "rollingball":

                if not tune_parameters:

                    current_ys = rolling_ball(
                        xs[:, i],
                        ys[:, i],
                        sphere_x=rolling_ball_sphere_x,
                        sphere_y=rolling_ball_sphere_y,
                        min_x=10,
                        max_x=50,
                        n_xs=5000,
                    )

                else:

                    fig, ax = plt.subplots()

                    axwave1 = plt.axes([0.17, 0.06, 0.65, 0.03])  # slider dimensions
                    axwave2 = plt.axes([0.17, 0, 0.65, 0.03])  # slider dimensions

                    slider_sphere_x = Slider(
                        axwave1,
                        "sphere x",
                        0,
                        100,
                        valinit=rolling_ball_sphere_x,
                        valfmt="%1.3f",
                    )  # 1
                    slider_sphere_y = Slider(
                        axwave2,
                        "sphere y",
                        0,
                        3,
                        valinit=rolling_ball_sphere_y,
                        valfmt="%1.3f",
                    )  # 2

                    def update_wave(val):

                        ax.cla()

                        removed = rolling_ball(
                            xs[:, i],
                            ys[:, i],
                            sphere_x=slider_sphere_x.val,
                            sphere_y=slider_sphere_y.val,
                            min_x=10,
                            max_x=50,
                            ax=ax,
                            n_xs=5000,
                        )

                        fig.canvas.draw_idle()

                    update_wave(None)

                    slider_sphere_x.on_changed(update_wave)
                    slider_sphere_y.on_changed(update_wave)

                    plt.show()

                    rolling_ball_sphere_x = slider_sphere_x.val
                    rolling_ball_sphere_y = slider_sphere_y.val

    if tune_parameters:
        exit()  # do not continue, since the same parameters should be used for all samples. Restart!


# TODO: Don't forget to transform first
