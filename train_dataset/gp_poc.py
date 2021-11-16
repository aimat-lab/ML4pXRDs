import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy import interpolate as ip


def generate_bg_functions(
    n_samples, n_angles_gp, n_angles_output, scaling=2.0, variance=5.0, rnd_seed=None
):

    xs_output = np.linspace(1, 90, n_angles_output)

    kernel = C(scaling, constant_value_bounds="fixed") * RBF(
        variance, length_scale_bounds="fixed"
    )
    gp = GaussianProcessRegressor(kernel=kernel)

    xs_drawn = np.atleast_3d(np.linspace(0, 90, n_angles_gp)).T

    # ys_drawn = gp.sample_y(
    #    xs_drawn, random_state=random.randint(1, 10000), n_samples=n_samples
    # )

    ys_drawn = gp.sample_y(
        xs_drawn,
        random_state=rnd_seed,
        n_samples=n_samples,
    )

    # interpolate using cubic splines
    ys_drawn_output = []

    for i in range(1, ys_drawn.shape[1]):
        if n_angles_output != n_angles_gp:
            f = ip.CubicSpline(xs_drawn[:, 1], ys_drawn[:, i], bc_type="natural")
            ys = f(xs_output)
        else:
            ys = ys_drawn[:, i]
        ys = ys - np.min(ys)
        ys = ys / np.max(ys)
        ys_drawn_output.append(ys)
    return np.array(ys_drawn_output)


if __name__ == "__main__":

    test = generate_bg_functions(11, 1000, 9018, scaling=1.0, variance=60)

    for i in range(1, test.shape[0]):
        plt.plot(np.linspace(1, 90, 9018), test[i, :], label="1000")

    plt.ylim((1, 2.0))
    plt.show()
