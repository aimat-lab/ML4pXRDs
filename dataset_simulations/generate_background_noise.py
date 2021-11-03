import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import lzma

# N = 9001
N = 4224
theta_min = 0
theta_max = 90

xs = np.linspace(theta_min, theta_max, N)
number_of_samples = 100000  # number of samples to generate
max_peaks_per_sample = 20  # max number of peaks per sample
polynomial_degree = 3
polymomial_parameters_range = 0.5

"""
def bg(l):
    a, b, c = np.random.random(3) - 0.5
    xs = np.linspace(-1.0, 1.0, l)
    ys = a * xs ** 3.0 + b * xs ** 2.0 + c * xs
    ys -= np.min(ys)
    ys /= np.max(ys)
    fluct_noise_level_max = 100
    ys *= (np.random.random() * 0.8 + 0.2) * fluct_noise_level_max
    base_noise_level_max = 100.0
    ys += (np.random.random() * 0.8 + 0.2) * base_noise_level_max
    return ys
"""


def generate_background_and_noise():

    xs = np.linspace(-1.0, 1.0, N)
    coefficients = [
        random.uniform(-polymomial_parameters_range, polymomial_parameters_range)
        for _ in range(0, polynomial_degree)
    ]

    ys = sum(coefficient * xs ** (n + 1) for n, coefficient in enumerate(coefficients))

    ys -= np.min(ys)

    ys += random.uniform(0, 0.2)

    """
    fluct_noise_level_max = 1.0
    ys *= np.random.normal(N) * fluct_noise_level_max
    """
    base_noise_level_max = 0.004
    ys += np.random.normal(size=N) * base_noise_level_max

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


ys_all_altered = []
ys_all_unaltered = []

for i in range(0, number_of_samples):

    if (i % 1000) == 0:
        print(f"Generated {i} samples.")

    # ys_altered = generate_background_and_noise()
    ys_altered = generate_background_and_noise_paper()

    plt.plot(xs, ys_altered)
    plt.show()

    ys_unaltered = np.zeros(N)

    sigma = random.uniform(0.1, 0.4)

    for j in range(0, random.randint(1, max_peaks_per_sample)):

        mean = random.uniform(0, 90)

        peak = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - mean) ** 2)
        ) * random.uniform(0, 1)

        ys_altered += peak
        ys_unaltered += peak

    ys_all_altered.append(ys_altered)
    ys_all_unaltered.append(ys_unaltered)

with open("patterns/noise_background/data", "wb") as file:
    pickle.dump((np.array(ys_all_altered), np.array(ys_all_unaltered)), file)
