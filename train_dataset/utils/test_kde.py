from scipy.stats.kde import gaussian_kde
import numpy as np

test = np.random.rand(12, 10000)

test = gaussian_kde(test)

print(test.resample(5).T)
