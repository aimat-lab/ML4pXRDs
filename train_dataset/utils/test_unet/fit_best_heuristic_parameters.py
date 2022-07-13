from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import pickle
import numpy as np
import pybaselines

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


def loss_function(
    parameter_0, parameter_1, method="rb"
):  # possible methods: "rb", "wavelet", "arPLS"

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
    default_ratio_exponent = -2.37287
    default_lambda_exponent = 7.311915
    default_arPLS_ratio = 10**default_ratio_exponent
    default_arPLS_lam = 10**default_lambda_exponent
    print(loss_function(default_arPLS_ratio, default_arPLS_lam, method="arPLS"))
    print(loss_function(default_arPLS_ratio, default_arPLS_lam, method="arPLS"))
