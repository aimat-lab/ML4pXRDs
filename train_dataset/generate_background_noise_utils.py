import numpy as np

import random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy import interpolate as ip

import time
import pandas as pd

from scipy.stats import truncnorm
import scipy.stats as st


N = 9018
n_angles_gp = 100
start_x = 0
end_x = 90
pattern_x = np.linspace(start_x, end_x, N)

max_peaks_per_sample = 40  # max number of peaks per sample
min_peak_height = 0.02

# GP parameters:
scaling = 1.0
# variance = 30.0 # this was my original estimate
# variance = 100.0
# variance = 50

variance = 10.0

# for background to peaks ratio:
scaling_max = 4.0

base_noise_level_max = 0.017
base_noise_level_min = 0.0015


def convert_to_discrete(peak_positions, peak_sizes, N=N, do_print=False):
    peak_info_disc = np.zeros(N)
    peak_size_disc = np.zeros(N)

    for i, peak_pos in enumerate(peak_positions):
        index = np.argwhere(pattern_x < peak_pos)[-1][0]

        while True:

            if peak_info_disc[index] == 0:

                peak_info_disc[index] += 1
                peak_size_disc[index] += peak_sizes[i]
                break

            else:

                if do_print:
                    print("Two or more peaks in the same spot!")
                index += 1  # move peak to the right

    return peak_info_disc, peak_size_disc


def generate_samples_gp(
    n_samples,
    n_angles_gp=n_angles_gp,
    n_angles_output=N,
    scaling=scaling,
    variance=variance,
    random_seed=None,
    mode="removal",
    do_plot=False,
    start_index=None,  # for proper scaling to 1.0
    end_index=None,  # inclusive
    compare_to_exp=False,
    plot_whole_range=False,
):

    # start = time.time()

    # first, generate enough random functions using a gaussian process
    xs_pattern = np.linspace(0, 90, n_angles_output)

    kernel = C(scaling, constant_value_bounds="fixed") * RBF(
        variance, length_scale_bounds="fixed"
    )
    gp = GaussianProcessRegressor(kernel=kernel)

    xs_gp = np.atleast_2d(np.linspace(0, 90, n_angles_gp)).T

    ys_gp = gp.sample_y(xs_gp, random_state=random_seed, n_samples=n_samples,)

    # stop = time.time()
    # print(f"GP took {stop-start} s")

    xs_all = []
    ys_all = []

    """
    class my_pdf(st.rv_continuous):
        def _pdf(self, x):
            if x < 0 or x > 1:
                return 0
            else:
                # return ((x - 0.5) ** 2 + 0.1) * 60 / 11
                # return (x - 1.0) ** 2 * 3
                return 1 / x ** 0.9 / 10

    my_cv = my_pdf(a=0, b=1, name="my_pdf")
    """

    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    trunc = get_truncated_normal(0, sd=0.3, low=0, upp=1)

    for i in range(0, n_samples):

        # scale the output of the gp to the desired length
        if n_angles_output != n_angles_gp:
            f = ip.CubicSpline(xs_gp[:, 0], ys_gp[:, i], bc_type="natural")
            background = f(xs_pattern)
        else:
            background = ys_gp[:, i]

        background = background - np.min(background)

        weight_background = np.sum(background)

        sigma_peaks = random.uniform(0.1, 0.5)
        peak_positions = []
        peak_sizes = []

        ys_unaltered = np.zeros(n_angles_output)

        for j in range(0, random.randint(0, max_peaks_per_sample)):

            mean = random.uniform(0, 90)

            # change
            # if j == 0:
            #    mean = 45
            # if j == 1:
            #    mean = 40

            peak_positions.append(mean)

            if j == 0:
                peak_size = 1.0
            elif random.random() < 0.3:
                peak_size = random.uniform(min_peak_height, 1)
                # loc = 0.1
                # scale = 0.3
                # peak_size = truncnorm.rvs(
                #    (min_peak_height - loc) / scale, (1 - loc) / scale, loc, scale
                # )

                ## change
                # if j == 0:
                #    peak_size = min_peak_height
                # elif j == 1:
                #    peak_size = 1.0
            else:

                peak_size = trunc.rvs()

            peak = (
                1
                / (sigma_peaks * np.sqrt(2 * np.pi))
                * np.exp(-1 / (2 * sigma_peaks ** 2) * (pattern_x - mean) ** 2)
            ) * peak_size

            peak_sizes.append(peak_size)

            ys_unaltered += peak

        print(peak_sizes)
        weight_peaks = np.sum(ys_unaltered)

        scaling = random.uniform(0, scaling_max)

        ys_altered = (
            background
            / weight_background
            * (weight_peaks if weight_peaks != 0 else 1)
            * scaling
            + ys_unaltered
        )

        noise_level = np.random.uniform(
            low=base_noise_level_min, high=base_noise_level_max
        )
        ys_altered += np.random.normal(size=N, scale=noise_level) * np.max(
            ys_altered[start_index : end_index + 1]
            if start_index is not None
            else ys_altered
        )
        ys_altered -= np.min(ys_altered)

        if start_index is None or end_index is None:
            normalizer = np.max(ys_altered)
        else:
            normalizer = np.max(ys_altered[start_index : end_index + 1])
        ys_altered /= normalizer
        ys_unaltered /= normalizer

        if do_plot:
            if not plot_whole_range:
                plt.plot(
                    pattern_x[start_index : end_index + 1],
                    ys_altered[start_index : end_index + 1],
                )
                plt.plot(
                    pattern_x[start_index : end_index + 1],
                    ys_unaltered[start_index : end_index + 1],
                )
            else:
                plt.plot(pattern_x, ys_altered)
                plt.plot(pattern_x, ys_unaltered)

            if compare_to_exp:

                for i in range(6, 7):
                    index = i
                    path = "exp_data/XRDdata_classification.csv"
                    data = pd.read_csv(path, delimiter=",", skiprows=1)
                    xs = np.array(
                        data.iloc[:, list(range(0, len(data.columns.values), 2))]
                    )
                    ys = np.array(
                        data.iloc[:, list(range(1, len(data.columns.values), 2))]
                    )
                    ys[:, index] = ys[:, index] - np.min(ys[:, index])
                    ys[:, index] = ys[:, index] / np.max(ys[:, index])
                    plt.plot(xs[:, index], ys[:, index], label=str(index))
                plt.legend()

            # plt.xlim(10, 50)
            plt.show()

        if mode == "removal":

            xs_all.append(ys_altered)
            ys_all.append(ys_unaltered)
            # ys_all.append(background_noise / scaler)

        elif mode == "info":

            peak_info_disc, peak_size_disc = convert_to_discrete(
                peak_positions, peak_sizes, do_print=False
            )

            xs_all.append(ys_altered)
            ys_all.append(peak_info_disc)

    return np.array(xs_all), np.array(ys_all)


if __name__ == "__main__":

    generate_samples_gp(
        100,
        n_angles_gp=100,
        do_plot=True,
        start_index=1002,
        end_index=5009,
        compare_to_exp=True,
        plot_whole_range=False,
    )
    # generate_samples_gp(1, n_angles_gp=1000, random_seed=1234, do_plot=False)
    # generate_samples_gp(1, n_angles_gp=2000, random_seed=1234, do_plot=False)
    # plt.show()
    exit()

    start = time.time()

    test = generate_samples_gp(128, do_plot=False)

    end = time.time()
    print(f"Took {end-start} s")

    """
    for i in range(0, test[0].shape[0]):
        plt.plot(pattern_x, test[0][i, :])
        plt.plot(pattern_x, test[1][i, :])
        plt.xlim((10, 50))
        plt.show()
    """
