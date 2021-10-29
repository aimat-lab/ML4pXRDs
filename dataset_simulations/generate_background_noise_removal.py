import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import lzma

# N = 9001
N = 4224
xs = np.linspace(0, 90, N)
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


ys_all_altered = []
ys_all_unaltered = []

for i in range(0, number_of_samples):

    if (i % 1000) == 0:
        print(f"Generated {i} samples.")

    ys_altered = generate_background_and_noise()
    ys_unaltered = np.zeros(N)

    sigma = random.uniform(0.1, 0.4)

    for j in range(0, random.randint(1, max_peaks_per_sample)):

        mean = random.uniform(0, 90)

        peak = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - mean) ** 2)
        ) * random.uniform(0, 0.1)

        ys_altered += peak
        ys_unaltered += peak

    # plt.plot(xs, ys_altered)

    ys_all_altered.append(ys_altered)
    ys_all_unaltered.append(ys_unaltered)

with open("patterns/noise_background/data", "wb") as file:
    pickle.dump((np.array(ys_all_altered), np.array(ys_all_unaltered)), file)
