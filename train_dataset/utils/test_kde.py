from scipy.stats.kde import gaussian_kde
import numpy as np
from sklearn.neighbors import KernelDensity

"""
test = np.random.rand(12, 10000)

test = gaussian_kde(test, bw_method=3)

# print(test.resample(5).T)

print(test.covariance)
"""

test = np.random.rand(10000, 12)

kd = KernelDensity(bandwidth=2, kernel="gaussian")

kd.fit(test)

print(kd.sample(1000))
