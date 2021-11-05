import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# N = 9001
N = 4224
theta_min = 0
theta_max = 90

xs = np.linspace(theta_min, theta_max, N)
number_of_samples = 50000  # number of samples to generate
max_peaks_per_sample = 50  # max number of peaks per sample
polynomial_degree = 6
polymomial_parameters_range = 1.0


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

    base_noise_level_max = 0.05
    base_noise_level_min = 0.01
    noise_level = np.random.uniform(low=base_noise_level_min, high=base_noise_level_max)
    ys += np.random.normal(size=N) * noise_level

    ys -= np.min(ys)

    return ys


def theta_rel(theta):
    return (theta - theta_min) / (theta_max - theta_min)


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
            xs,
            trunc_normal(0, T, T / 3, T / 7),
            np.random.uniform(10, 60),
            np.random.uniform(0, 1 / 7),
        )

    if choices[1]:
        diffractogram += f_step_down(
            xs,
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

        diffractogram += f_polynomial(xs, n_max, alpha_n)

    if choices[3]:
        diffractogram += f_bump(
            xs,
            trunc_normal(0, 3 * T / 5, 2 * T / 5, 3 * T / 35),
            np.random.uniform(40, 70),
        )

    return diffractogram


def convert_to_discrete(peak_positions, peak_sizes):
    peak_info_disc = np.zeros(N)
    peak_size_disc = np.zeros(N)

    for i, peak_pos in enumerate(peak_positions):
        index = np.argwhere(xs < peak_pos)[-1]
        peak_info_disc[index] += 1
        peak_size_disc[index] += peak_sizes[i]

    return peak_info_disc, peak_size_disc


ys_all_altered = []
ys_all_unaltered = []
all_peak_info_disc = []
all_peak_size_disc = []


for i in range(0, number_of_samples):

    if (i % 1000) == 0:
        print(f"Generated {i} samples.", flush=True)

    ys_altered = generate_background_and_noise()
    # ys_altered = generate_background_and_noise_paper()

    ys_unaltered = np.zeros(N)

    sigma = random.uniform(0.1, 0.5)
    peak_positions = []
    peak_sizes = []

    for j in range(0, random.randint(1, max_peaks_per_sample)):

        mean = random.uniform(0, 90)

        peak_positions.append(mean)

        peak_size = random.uniform(0.01, 1)
        peak = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - mean) ** 2)
        ) * peak_size

        peak_sizes.append(peak_size)

        ys_altered += peak
        ys_unaltered += peak

    ys_all_altered.append(ys_altered)
    ys_all_unaltered.append(ys_unaltered)

    peak_info_disc, peak_size_disc = convert_to_discrete(peak_positions, peak_sizes)

    all_peak_info_disc.append(peak_info_disc)
    all_peak_size_disc.append(peak_size_disc)

    """
    plt.plot(xs, ys_altered)
    plt.plot(xs, ys_unaltered)
    plt.scatter(xs, peak_info_disc)
    plt.scatter(xs, peak_size_disc)
    plt.show()
    """

with open("patterns/noise_background/data", "wb") as file:
    pickle.dump(
        (
            np.array(ys_all_altered),
            np.array(ys_all_unaltered),
            np.array(all_peak_info_disc),
            np.array(all_peak_size_disc),
        ),
        file,
    )
