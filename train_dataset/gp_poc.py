import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import random

kernel = C(1.0, (1e-3, 1e3)) * RBF(60, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

xs_drawn = np.atleast_2d(np.linspace(0, 100, 1000)).T

for i in range(0, 10):
    # TODO: Fix this random thing
    ys_drawn = gp.sample_y(xs_drawn, random_state=random.randint(0, 10000))
    plt.plot(xs_drawn, ys_drawn)
plt.show()
