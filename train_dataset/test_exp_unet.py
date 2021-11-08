import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate as ip
from UNet_1DCNN import UNet


def load_experimental_data(mode="classification"):
    if mode == "classification":

        path = "exp_data/XRDdata_classification.csv"
        data = pd.read_csv(path, delimiter=",", skiprows=1)
        xs = np.array(data.iloc[:, list(range(0, len(data.columns.values), 2))])
        ys = np.array(data.iloc[:, list(range(1, len(data.columns.values), 2))])

        return (xs, ys)

    elif mode == "texture":  # TODO: watch out: these have a different range

        data_file_path = "exp_data/XRD_6_component_systems_repeat.csv"
        df = pd.read_csv(data_file_path, sep=";")

        x = np.array(df.iloc[:, 0])
        xs = np.repeat(x[:, np.newaxis], len(df.columns.values) - 1, axis=1)
        ys = np.array(df.iloc[:, list(range(1, len(df.columns.values)))])

        return (xs, ys)

    else:

        raise Exception("Mode for loading experimental data not supported.")


if __name__ == "__main__":

    xs_exp, ys_exp = load_experimental_data(mode="classification")

    N = 9018
    pattern_x = np.linspace(0, 90, N)

    start_x = 10
    end_x = 50
    start_index = np.argwhere(pattern_x >= start_x)[0][0]
    end_index = np.argwhere(pattern_x <= end_x)[-1][0]
    pattern_x = pattern_x[start_index : end_index + 1]
    N = len(pattern_x)

    my_unet = UNet(N, 3, 1, 5, 64, output_nums=1, problem_type="Regression")
    model = my_unet.UNet()
    model.load_weights("unet/removal_cps/weights20")

    """
    with open("unet/removal_cps/scaler", "rb") as file:
        sc = pickle.load(file)
    """

    for i in range(0, xs_exp.shape[1]):

        current_xs = xs_exp[:, i]
        current_ys = ys_exp[:, i]

        # plt.plot(current_xs, current_ys)
        # plt.show()

        f = ip.CubicSpline(current_xs, current_ys, bc_type="natural")

        ys = f(pattern_x)
        ys -= np.min(ys)
        ys = ys / np.max(ys)

        plt.plot(pattern_x, ys, label="Experimental rescaled")

        # ys = np.expand_dims(sc.transform([ys]), axis=2)
        ys = np.expand_dims([ys], axis=2)

        corrected = model.predict(ys)

        plt.plot(
            pattern_x, corrected[0, :, 0], label="Corrected via U-Net",
        )

        plt.plot(pattern_x, np.zeros(len(pattern_x)))

        # plt.plot(
        #    pattern_x, ys[0, :, 0] - corrected[0, :, 0], label="Background and noise"
        # )

        plt.legend()
        plt.show()

        """
        baseline_arPLS = baseline_arPLS(current_ys)
        current_ys_arPLS = current_ys - baseline_arPLS

        current_ys_rolling_ball = rolling_ball(
            xs[:, i],
            ys[:, i],
            sphere_x=rolling_ball_sphere_x,
            sphere_y=rolling_ball_sphere_y,
            min_x=10,
            max_x=50,
            n_xs=5000,
        )
        """
