import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import time

N = 9018
start_x = 0
end_x = 90
pattern_x = np.linspace(0, 90, N)

max_peaks_per_sample = 50  # max number of peaks per sample
polynomial_degree = 6
polymomial_parameters_range = 1.0

min_peak_height = 0.01


def generate_background_and_noise():

    xs = np.linspace(-1.0, 1.0, N)
    coefficients = [
        random.uniform(-polymomial_parameters_range, polymomial_parameters_range)
        for _ in range(0, polynomial_degree)
    ]

    ys = sum(coefficient * xs ** (n + 1) for n, coefficient in enumerate(coefficients))

    ys -= np.min(ys)

    # fluct_noise_level_max = 1.0
    # ys *= np.random.normal(N) * fluct_noise_level_max

    # TODO: Implement signal-to-noise ratio as in paper

    weight_background = np.sum(ys)

    base_noise_level_max = 0.05
    base_noise_level_min = 0.01
    noise_level = np.random.uniform(low=base_noise_level_min, high=base_noise_level_max)

    # TODO: here is how the probability paper generates noise:
    # norm_signal = 100 * signal / max(signal)
    # noise = np.random.normal(0, 0.25, 4501)

    ys += np.random.normal(size=N) * noise_level

    ys -= np.min(ys)

    return ys, weight_background


def theta_rel(theta):
    return (theta - start_x) / (end_x - start_x)


def f_step_up(theta, h, alpha, theta_rel_step):
    return h * (1 / (1 + np.exp(alpha * (theta_rel(theta) - theta_rel_step))))


def f_step_down(theta, h, alpha, theta_rel_step):
    return h * (1 - (1 / (1 + np.exp(alpha * (theta_rel(theta) - theta_rel_step)))))


def f_polynomial(theta, n_max, alpha_n):
    return np.abs(
        np.sum([alpha_n[i] * theta_rel(theta) ** i for i in range(0, n_max + 1)])
    )


def f_bump(theta, h, n):
    return h * (n * theta_rel(theta)) ** 2 * np.exp(-1 * n * theta_rel(theta))


def trunc_normal(min, max, mean, std):
    while True:
        value = np.random.normal(mean, std)

        if value > min and value < max:
            return value
        else:
            continue


def generate_background_and_noise_paper():

    diffractogram = np.zeros(N)

    # TODO: find potentially better way of handling noise
    # patterns go from 0 to 1
    # diffractogram += np.random.uniform(0.0002, 0.002, N)

    choices = [1 if x < 0.5 else 0 for x in np.random.uniform(size=4)]
    T = 0.1 / np.sum(choices)

    if choices[0]:
        diffractogram += f_step_up(
            pattern_x,
            trunc_normal(0, T, T / 3, T / 7),
            np.random.uniform(10, 60),
            np.random.uniform(0, 1 / 7),
        )

    if choices[1]:
        diffractogram += f_step_down(
            pattern_x,
            trunc_normal(0, T, T / 3, T / 7),
            np.random.uniform(10, 60),
            np.random.uniform(1 - 1 / 7, 1),
        )

    if choices[2]:
        n_max = np.random.randint(0, 5)
        alpha_n = np.zeros(n_max + 1)
        for i, alpha in enumerate(alpha_n):
            if np.random.uniform() < 0.5:
                alpha_n[i] = 3 * T / (2 * (n_max + 1)) * np.random.uniform(-1, 1)

        diffractogram += f_polynomial(pattern_x, n_max, alpha_n)

    if choices[3]:
        diffractogram += f_bump(
            pattern_x,
            trunc_normal(0, 3 * T / 5, 2 * T / 5, 3 * T / 35),
            np.random.uniform(40, 70),
        )

    return diffractogram


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


def generate_samples(N=128, mode="removal", do_plot=False, do_print=False, scaler=None):

    xs_all = []
    ys_all = []

    total_time_discretizing = 0

    for i in range(0, N):

        if do_print and (i % 1000) == 0:
            print(f"Generated {i} samples.")

        background_noise, weight_background = generate_background_and_noise()
        # ys_altered = generate_background_and_noise_paper()

        ys_unaltered = np.zeros(len(pattern_x))

        sigma = random.uniform(0.1, 0.5)
        peak_positions = []
        peak_sizes = []

        for j in range(0, random.randint(1, max_peaks_per_sample)):

            mean = random.uniform(0, 90)

            peak_positions.append(mean)

            peak_size = random.uniform(min_peak_height, 1)
            peak = (
                1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-1 / (2 * sigma ** 2) * (pattern_x - mean) ** 2)
            ) * peak_size

            peak_sizes.append(peak_size)

            ys_unaltered += peak

        weight_peaks = np.sum(ys_unaltered)
        # scaling = random.uniform(0, 1.0)
        # print(scaling)

        scaling = 2

        ys_altered = (
            background_noise / weight_background * weight_peaks * scaling + ys_unaltered
        )

        scaler = np.max(ys_altered)
        ys_altered /= scaler
        ys_unaltered /= scaler

        if do_plot:
            plt.plot(pattern_x, ys_altered)
            plt.plot(pattern_x, ys_unaltered)

            plt.xlim(10, 50)
            plt.show()

        if mode == "removal":

            xs_all.append(ys_altered)
            ys_all.append(ys_unaltered)
            # ys_all.append(background_noise / scaler)

        elif mode == "info":

            start = time.time()
            peak_info_disc, peak_size_disc = convert_to_discrete(
                peak_positions, peak_sizes, do_print=False
            )
            end = time.time()
            total_time_discretizing += end - start

            xs_all.append(ys_altered)
            ys_all.append(peak_info_disc)

    print(f"Total time spent discretizing: {total_time_discretizing}")

    return np.array(xs_all), np.array(ys_all)


if __name__ == "__main__":
    start = time.time()
    x, y = generate_samples(
        N=128, mode="removal", do_print=False, do_plot=False
    )  # mode doesn't matter here, since we are only interested in the input
    end = time.time()
    print(f"Took {end-start} s")

    """
  
    # This code calculates the standard scaler for 100k samples, which is a good estimate for
    # the overall distribution
    n_samples = 50000
    path = "unet/scaler"
    x, y = generate_samples(
        N=n_samples, mode="removal", plot=False, do_print=True
    )  # mode doesn't matter here, since we are only interested in the input
    # scale features
    sc = StandardScaler()
    x_test = sc.fit_transform(x)
    print(sc.mean_)
    print(sc.var_)
    print(sc.scale_)
    # plt.plot(sc.var_)
    # plt.show()
    # Create new scaler where all means and stds are the same
    # new_scaler = StandardScaler()
    # new_scaler.mean_ = np.repeat(np.mean(sc.mean_), len(sc.mean_))
    # new_scaler.var_ = np.repeat(np.mean(sc.var_), len(sc.var_))
    # new_scaler.scale_ = np.repeat(np.mean(sc.scale_), len(sc.scale_))
    with open(path, "wb") as file:
        pickle.dump(sc, file)
    """
