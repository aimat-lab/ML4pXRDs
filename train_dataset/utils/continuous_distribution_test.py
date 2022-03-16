from scipy.stats import kde
import matplotlib.pyplot as plt
import numpy as np

x = [2, 2, 1.2, 1.8, 1.9, 1.5, 2.5, 5, 5.3, 5.2, 5.7, 4.3, 4.1, 4.5]
density = kde.gaussian_kde(x)

grid = np.linspace(min(x), max(x), 1000)

plt.plot(grid, density(grid))
plt.hist(x, density=True, bins=20)

plt.show()
