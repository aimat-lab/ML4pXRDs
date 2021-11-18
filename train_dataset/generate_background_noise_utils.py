import numpy as np

import random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy import interpolate as ip

import time


N = 9018
n_angles_gp = 100
start_x = 0
end_x = 90
pattern_x = np.linspace(start_x, end_x, N)

max_peaks_per_sample = 50  # max number of peaks per sample
min_peak_height = 0.01

# GP parameters:
scaling = 1.0
# variance = 30.0 # this was my original estimate
# variance = 100.0
# variance = 50Thursday:

variance = 30.0

# for background to peaks ratio:
scaling_max = 2.0


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

    for i in range(0, n_samples):

        # plt.scatter(xs_gp[:, 0], ys_gp[:, i], s=1)

        if n_angles_output != n_angles_gp:
            f = ip.CubicSpline(xs_gp[:, 0], ys_gp[:, i], bc_type="natural")
            background = f(xs_pattern)
        else:
            background = ys_gp[:, i]

        # plt.plot(xs_pattern, background)

        background = background - np.min(background)
        background = background / np.max(background)

        weight_background = np.sum(background)

        base_noise_level_max = 0.05
        base_noise_level_min = 0.01
        noise_level = np.random.uniform(
            low=base_noise_level_min, high=base_noise_level_max
        )
        noise_scale = 1.0

        background += np.random.normal(size=N, scale=noise_scale) * noise_level
        background -= np.min(background)

        sigma_peaks = random.uniform(0.1, 0.5)
        peak_positions = []
        peak_sizes = []

        ys_unaltered = np.zeros(n_angles_output)

        for j in range(0, random.randint(0, max_peaks_per_sample)):

            mean = random.uniform(0, 90)

            peak_positions.append(mean)

            peak_size = random.uniform(min_peak_height, 1)
            peak = (
                1
                / (sigma_peaks * np.sqrt(2 * np.pi))
                * np.exp(-1 / (2 * sigma_peaks ** 2) * (pattern_x - mean) ** 2)
            ) * peak_size

            peak_sizes.append(peak_size)

            ys_unaltered += peak

        weight_peaks = np.sum(ys_unaltered)

        scaling = random.uniform(0, scaling_max)

        ys_altered = (
            background
            / weight_background
            * (weight_peaks if weight_peaks != 0 else 1)
            * scaling
            + ys_unaltered
        )

        if start_index is None or end_index is None:
            normalizer = np.max(ys_altered)
        else:
            normalizer = np.max(ys_altered[start_index : end_index + 1])

        ys_altered /= normalizer
        ys_unaltered /= normalizer

        if do_plot:
            plt.plot(pattern_x, ys_altered)
            plt.plot(pattern_x, ys_unaltered)

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

    generate_samples_gp(1, n_angles_gp=100, random_seed=1234, do_plot=False)
    generate_samples_gp(1, n_angles_gp=1000, random_seed=1234, do_plot=False)
    generate_samples_gp(1, n_angles_gp=2000, random_seed=1234, do_plot=False)
    plt.show()
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
