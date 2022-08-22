from typing import final
from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns
from train_dataset.utils.test_unet.heuristic_bg_utils import *
import numpy as np
import pybaselines
from lmfit import minimize
from scipy.optimize import minimize
import pickle

use_N = 10

N_polynomial_coefficients = 12
R2_score_threshold = 0.97

# ratio_initial = 10 ** (-2.37287)
ratio_fixed = 10 ** (-5)

lambda_initial = 7.311915
sphere_x_initial = 6.619
sphere_y_initial = 0.3

do_plot = True

xs, ys, difs, raw_files, parameters, scores = get_rruff_patterns(
    only_refitted_patterns=True,
    only_if_dif_exists=True,
    start_angle=5,
    end_angle=90,
    reduced_resolution=False,
    return_refitted_parameters=True,
)

for i in reversed(range(len(scores))):
    if scores[i] < R2_score_threshold:
        del scores[i]
        del parameters[i]
        del raw_files[i]
        del difs[i]
        del ys[i]
        del xs[i]

x_range = np.linspace(5, 90, 8501)


def loss_function(inputs, indices, method="rb"):  # possible methods: "rb", "arPLS"

    loss = 0

    for i in indices:

        target_y = np.zeros(len(xs[i]))
        for j in range(N_polynomial_coefficients):
            target_y += parameters[i][j] * xs[i] ** j

        if method == "rb":
            result = rolling_ball(xs[i], ys[i], inputs[0], inputs[1])
        elif method == "arPLS":
            result = baseline_arPLS(ys[i], ratio=ratio_fixed, lam=10 ** inputs[0])

        loss += np.sum((target_y - result) ** 2)

    return loss


if __name__ == "__main__":

    # repeat this procedure 5 times
    for k in range(0, 5):

        # randomly pick indices to perform the fitting on:
        indices = np.random.choice(list(range(len(xs))), size=use_N)

        final_parameters_per_method = {}

        for method in ["arPLS", "rb"]:

            print(f"Showing results of method {method}:")

            def wrapper(inputs):
                return loss_function(inputs, indices, method)

            if method == "rb":
                result = minimize(
                    wrapper,
                    [sphere_x_initial, sphere_y_initial],
                    bounds=[(0, np.inf), (0, np.inf)],
                    method="Nelder-Mead",
                    options={"maxiter": 1000000},
                )
            elif method == "arPLS":
                result = minimize(
                    wrapper,
                    [lambda_initial],
                    bounds=[(2, 11)],
                    method="Nelder-Mead",
                    options={"maxiter": 1000000},
                )

            if not result.success:
                raise ValueError(result.message)
            else:
                print("Fit successful.")

            print(f"Best fit heuristic parameters for {method}:")
            print(result.x)

            final_parameters_per_method[method] = result.x

        if do_plot:

            for i in indices:

                target_y = np.zeros(len(xs[i]))
                for j in range(N_polynomial_coefficients):
                    target_y += parameters[i][j] * xs[i] ** j

                plt.plot(
                    xs[i],
                    baseline_arPLS(ys[i], ratio=ratio_fixed, lam=10**lambda_initial),
                    label="Initial",
                )
                plt.plot(xs[i], target_y, label="Target")
                plt.plot(xs[i], ys[i], label="Pattern")

                plt.plot(
                    xs[i],
                    rolling_ball(
                        xs[i],
                        ys[i],
                        final_parameters_per_method["rb"][0],
                        final_parameters_per_method["rb"][1],
                    ),
                    label="RB",
                )
                plt.plot(
                    xs[i],
                    baseline_arPLS(
                        ys[i],
                        ratio=ratio_fixed,
                        lam=10 ** final_parameters_per_method["arPLS"][0],
                    ),
                    label="arPLS",
                )
                plt.legend()
                plt.show()

        # For testing by hand
        # if True:
        #    plt.plot(xs[0], ys[0])
        #    plot_heuristic_fit(
        #        xs[0],
        #        ys[0],
        #        [
        #            final_parameters_per_method["rb"][0],
        #            final_parameters_per_method["rb"][1],
        #            5,
        #            final_parameters_per_method["arPLS"][0],
        #            1.0,
        #            2.0,
        #        ],
        #    )

        raw_files_used_for_fitting = [raw_files[l] for l in indices]
        with open(f"bestfit_heuristic_{k}.pickle", "wb") as file:
            pickle.dump((final_parameters_per_method, raw_files_used_for_fitting), file)
