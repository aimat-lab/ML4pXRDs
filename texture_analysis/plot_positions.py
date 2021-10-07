import matplotlib.pyplot as plt
import numpy as np

# plot position (theta) of first two peaks for each sample
data = np.genfromtxt("ratios.csv", delimiter=";", skip_header=1)

ys_0 = data[:, 3]
ys_1 = data[:, 9]

ys_0 = ys_0[~np.isnan(ys_0)]
ys_1 = ys_1[~np.isnan(ys_1)]

# plt.scatter(list(range(0, len(ys_0))), ys_1 - ys_0)

# plt.scatter(list(range(0, len(ys_0))), np.sin(ys_0 / 2 / 360 * 2 * np.pi), s=0.7)
# plt.scatter(list(range(0, len(ys_0))), np.sin(ys_1 / 2 / 360 * 2 * np.pi), s=0.7)

plt.scatter(list(range(0, len(ys_0))), ys_0, s=0.7)
plt.scatter(list(range(0, len(ys_0))), ys_1, s=0.7)

plt.show()

