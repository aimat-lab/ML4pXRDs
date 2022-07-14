from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pybaselines
from lmfit import minimize
from scipy.optimize import minimize
import pickle

use_first_N = 5

xs, ys, difs, raw_files, parameters = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=5,
    end_angle=90,
    reduced_resolution=False,
    return_refitted_parameters=True,
)

x_range = np.linspace(5, 90, 8501)


def loss_function(inputs, method="rb"):  # possible methods: "rb", "wavelet", "arPLS"

    parameter_0 = inputs[0]
    parameter_1 = inputs[1]

    loss = 0

    for i, pattern in enumerate(ys[0:use_first_N]):

        target_y = (
            parameters[i][0]
            + parameters[i][1] * xs[i]
            + parameters[i][2] * xs[i] ** 2
            + parameters[i][3] * xs[i] ** 3
            + parameters[i][4] * xs[i] ** 4
            + parameters[i][5] * xs[i] ** 5
        )

        if method == "rb":
            result = rolling_ball(xs[i], ys[i], parameter_0, parameter_1)
        elif method == "wavelet":
            result = pybaselines.classification.cwt_br(
                ys[i],
                num_std=parameter_0,
                min_length=int(parameter_1),
            )[0]
        elif method == "arPLS":
            result = baseline_arPLS(ys[i], ratio=parameter_0, lam=parameter_1)

        loss += np.sum((target_y - result) ** 2)

    return loss


if __name__ == "__main__":

    for method in ["rb"]:

        def wrapper(inputs):
            return loss_function(inputs, method)

        result = minimize(wrapper, [6.619, 0.3], bounds=[(0, np.inf), (0, np.inf)])

        print("Best fit heuristic parameters:")
        print(result.x)

        # TODO: Do this for all methods
        with open("bestfit_heuristic.pickle", "wb") as file:
            pickle.dump(result.x, file)

        for i in range(len(xs)):

            target_y = (
                parameters[i][0]
                + parameters[i][1] * xs[i]
                + parameters[i][2] * xs[i] ** 2
                + parameters[i][3] * xs[i] ** 3
                + parameters[i][4] * xs[i] ** 4
                + parameters[i][5] * xs[i] ** 5
            )

            plt.plot(xs[i], target_y)
            plt.plot(xs[i], ys[i])
            plt.plot(xs[i], rolling_ball(xs[i], ys[i], result.x[0], result.x[1]))
            plt.show()
