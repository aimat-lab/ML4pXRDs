from scipy.stats import kde
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

if False:  # 1D

    x = [2, 2, 1.2, 1.8, 1.9, 1.5, 2.5, 5, 5.3, 5.2, 5.7, 4.3, 4.1, 4.5]
    density = kde.gaussian_kde(x)

    grid = np.linspace(min(x), max(x), 1000)

    plt.plot(grid, density(grid))
    plt.hist(x, density=True, bins=20)

    points = density.resample(100)
    plt.hist(points[0, :], density=True, bins=20)

    test = np.linspace(-1000, 1000, 10000)
    print(np.sum(density(test)) * (test[1] - test[0]))

    plt.show()

if True:

    x = [
        [2, 1],
        [2, 1],
        [1.2, 1.2],
        [1.8, 1.3],
        [1.9, 1.4],
        [1.5, 1.2],
        [2.5, 1.1],
        [5, 1.1],
        [5.3, 0.9],
        [5.2, 0.9],
        [5.7, 0.9],
        [4.3, 0.8],
        [4.1, 0.8],
        [4.5, 0.85],
    ]
    x = np.array(x).T

    density = kde.gaussian_kde(x)

    lin_grid_1 = np.linspace(
        min([item[0] for item in x.T]), max([item[0] for item in x.T]), 30
    )
    lin_grid_2 = np.linspace(
        min([item[1] for item in x.T]), max([item[1] for item in x.T]), 30
    )
    grid = np.array(np.meshgrid(lin_grid_1, lin_grid_2)).T.reshape(-1, 2)

    cm = plt.cm.get_cmap("RdYlBu")
    sc = plt.scatter(
        [item[0] for item in grid],
        [item[1] for item in grid],
        c=[density(item)[0] for item in grid],
        s=20,
        cmap=cm,
    )
    plt.scatter([item[0] for item in x.T], [item[1] for item in x.T], s=50)
    plt.colorbar(sc)

    # Sampling a point:
    # points = density.resample(100).T

    # Get the conditional probability: p(x|y) = p(x,y) / p(y) = p(x,y) / int (p(x,y) dx)

    conditional = (
        lambda x, y: density([x, y])[0]
        / integrate.quad(
            lambda x_1, y_1: density([x_1, y_1])[0], -np.inf, +np.inf, args=(y)
        )[0]
    )

    plt.figure()
    cm = plt.cm.get_cmap("RdYlBu")
    sc = plt.scatter(
        [item[0] for item in grid],
        [item[1] for item in grid],
        c=[conditional(*item) for item in grid],
        s=20,
        cmap=cm,
    )
    plt.scatter([item[0] for item in x.T], [item[1] for item in x.T], s=50)
    plt.colorbar(sc)
    plt.show()

    y_0 = 0.4
    print(integrate.quad(lambda x: conditional(x, y_0), -np.inf, np.inf))
