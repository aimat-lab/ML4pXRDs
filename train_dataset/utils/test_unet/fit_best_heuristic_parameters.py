from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import pickle
import numpy as np
import pybaselines
from lmfit import minimize
from lmfit import Minimizer
from lmfit import Parameters
from scipy.optimize import minimize

use_first_N = 10
show_results = True

xs, ys, difs = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=5,
    end_angle=90,
    reduced_resolution=False,
)

with open("rruff_refits.pickle", "rb") as file:
    parameter_results = pickle.load(file)

    parameters = [item[1] for item in parameter_results]
    for i in range(len(parameters)):
        parameters[i] = [item.value for item in parameters[i].values()]

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

        """
        params = Parameters()

        if method == "rb":
            params.add("parameter_0", 6.619, min=0, max=np.inf)
            params.add("parameter_1", 0.3, min=0, max=np.inf)
        elif method == "wavelet":
            # TODO: Fix these ranges
            params.add("parameter_0", 6.619, min=0, max=np.inf)
            params.add("parameter_1", 0.3, min=0, max=np.inf)
        elif method == "arPLS":
            pass  # TODO: Add ranges

        minimizer = Minimizer(wrapper, params)

        # print(minimize(wrapper, params))
        minimizer.minimize()
        """

        result = minimize(wrapper, [6.619, 0.3], bounds=[(0, np.inf), (0, np.inf)])

        print(result.x)

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

            # TODO: Something is still wrong with the y-shift!
