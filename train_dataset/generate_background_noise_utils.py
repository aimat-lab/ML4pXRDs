import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy import interpolate as ip
import pandas as pd
from scipy.stats import truncnorm


n_angles_gp = 60
max_peaks_per_sample = 22  # max number of peaks per sample
min_peak_height = 0.010

# GP parameters:
scaling = 1.0
variance = 10.0

# for background to peaks ratio:
scaling_max = 120.0

base_noise_level_max = 0.03
base_noise_level_min = 0.003


def convert_to_discrete(
    x_range, peak_positions, peak_sizes, n_angles=4016, do_print=False
):

    pattern_x = np.linspace(x_range[0], x_range[1], n_angles)

    peak_info_disc = np.zeros(n_angles)
    peak_size_disc = np.zeros(n_angles)

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


# TODO: Add bump
def theta_rel(theta, min_x, max_x):
    return (theta - min_x) / (max_x - min_x)


def f_bump(theta, h, n, min_x, max_x):
    return (
        h
        * (n * theta_rel(theta, min_x, max_x)) ** 2
        * np.exp(-1 * n * theta_rel(theta, min_x, max_x))
    )


def generate_samples_gp(
    n_samples,
    x_range,
    n_angles_gp=n_angles_gp,
    n_angles_output=4016,
    scaling=scaling,
    variance=variance,
    random_seed=None,
    mode="removal",
    do_plot=False,
    compare_to_exp=False,
):

    min_x, max_x = x_range

    # first, generate enough random functions using a gaussian process
    pattern_xs = np.linspace(min_x, max_x, n_angles_output)

    kernel = C(scaling, constant_value_bounds="fixed") * RBF(
        variance, length_scale_bounds="fixed"
    )
    gp = GaussianProcessRegressor(kernel=kernel)

    xs_gp = np.atleast_2d(np.linspace(min_x, max_x, n_angles_gp)).T

    ys_gp = gp.sample_y(xs_gp, random_state=random_seed, n_samples=n_samples,)

    xs_all = []
    ys_all = []

    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    trunc = get_truncated_normal(0, sd=0.3, low=min_peak_height, upp=1)

    for i in range(0, n_samples):

        # scale the output of the gp to the desired length
        if n_angles_output != n_angles_gp:
            f = ip.CubicSpline(xs_gp[:, 0], ys_gp[:, i], bc_type="natural")
            background = f(pattern_xs)
        else:
            background = ys_gp[:, i]

        background = background - np.min(background)
        weight_background = np.sum(background)

        scaling = random.uniform(0, scaling_max)
        background = background / weight_background * 10 * scaling

        sigma_peaks = random.uniform(0.1, 0.5)
        peak_positions = []
        peak_sizes = []

        ys_unaltered = np.zeros(n_angles_output)

        for j in range(0, random.randint(1, max_peaks_per_sample)):

            mean = random.uniform(min_x, max_x)

            peak_positions.append(mean)

            if j == 0:
                peak_size = 1.0
            # elif random.random() < 0.3:
            #    peak_size = random.uniform(min_peak_height, 1)
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
                * np.exp(-1 / (2 * sigma_peaks ** 2) * (pattern_xs - mean) ** 2)
            ) * peak_size

            peak_sizes.append(peak_size)

            ys_unaltered += peak

        ys_altered = background + ys_unaltered

        noise_level = np.random.uniform(
            low=base_noise_level_min, high=base_noise_level_max
        )
        ys_altered += np.random.normal(size=n_angles_output, scale=noise_level)
        ys_altered -= np.min(ys_altered)

        normalizer = np.max(ys_altered)

        ys_altered /= normalizer
        ys_unaltered /= normalizer

        if do_plot:
            plt.plot(pattern_xs, ys_altered)
            plt.plot(pattern_xs, ys_unaltered)

            if compare_to_exp:
                for i in range(5, 7):
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

    generate_samples_gp(100, (10, 50), do_plot=True, compare_to_exp=True)

