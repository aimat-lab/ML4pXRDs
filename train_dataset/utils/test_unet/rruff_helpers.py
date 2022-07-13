import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from lmfit import Model
from pyxtal.symmetry import Group
from lmfit import Parameters

from jax import grad
from jax import jacrev

import jax.numpy as jnp
from jax import jit
import pickle
from glob import glob
import os

########## Peak profile functions from https://en.wikipedia.org/wiki/Rietveld_refinement ##########
# Parameter ranges from: file:///home/henrik/Downloads/PowderDiff26201188-93%20(1).pdf


def fn_x(theta, mean, H):
    return (2 * theta - 2 * mean) / H


def fn_H(theta, U, V, W):

    H_squared = (
        U * jnp.tan(theta / 360 * 2 * jnp.pi) ** 2
        + V * jnp.tan(theta / 360 * 2 * jnp.pi)
        + W
    )

    # return np.nan_to_num(np.sqrt(H_squared)) + 0.0001 # TODO: Changed
    return jnp.sqrt(H_squared)


def fn_H_dash(theta, X, Y):
    return X / jnp.cos(theta / 360 * 2 * jnp.pi) + Y * jnp.tan(theta / 360 * 2 * jnp.pi)


def fn_eta(theta, eta_0, eta_1, eta_2):
    return eta_0 + eta_1 * 2 * theta + eta_2 * theta**2


def peak_function(theta, mean, U, V, W, X, Y, eta_0, eta_1, eta_2):

    C_G = 4 * jnp.log(2)
    C_L = 4

    H = fn_H(theta, U, V, W)
    H_dash = fn_H_dash(theta, X, Y)

    eta = fn_eta(theta, eta_0, eta_1, eta_2)

    # Lambda = (
    #    H**5
    #    + 2.69269 * H**4 * H_dash
    #    + 2.42843 * H**3 * H_dash**2
    #    + 4.47163 * H**2 * H_dash**3
    #    + 0.07842 * H * H_dash**4
    #    + H_dash**5
    # ) ** (1 / 5)

    # eta = (
    #    1.36603 * (H_dash / Lambda)
    #    - 0.47719 * (H_dash / Lambda) ** 2
    #    + 0.11116 * (H_dash / Lambda) ** 3
    # )

    x = fn_x(theta, mean, H)

    return eta * C_G ** (1 / 2) / (jnp.sqrt(jnp.pi) * H) * jnp.exp(
        -1 * C_G * x**2
    ) + (1 - eta) * C_L ** (1 / 2) / (jnp.sqrt(jnp.pi) * H_dash) * (
        1 + C_L * x**2
    ) ** (
        -1
    )


# Copper:
lambda_K_alpha_1 = 1.54056  # angstrom
lambda_K_alpha_2 = 1.54439  # angstrom


def smeared_peaks(
    xs,
    pattern_angles,
    pattern_intensities,
    U,
    V,
    W,
    X,
    Y,
    eta_0,
    eta_1,
    eta_2,
    K_alpha_splitting=False,
    wavelength=1.541838,  # needed if K_alpha_splitting=True; wavelength from the DIF file.
    print_thetas=False,
):

    # Splitting Kalpha_1, Kalpha_2: https://physics.stackexchange.com/questions/398724/why-is-k-alpha-3-2-always-more-intense-than-k-alpha-1-2-in-copper
    # => ratio 2:1
    # Only the lorentz polarization correction depends on theta, can most likely be ignored
    # n * lambda = 2*d*sin(theta)
    # => lambda_1 / lambda_2 =sin(theta_1) / sin(theta_2)
    # => sin(theta_2) = sin(theta_1) * lambda_2 / lambda_1

    ys = jnp.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        if not K_alpha_splitting:

            peak = intensity * peak_function(
                xs / 2, twotheta / 2, U, V, W, X, Y, eta_0, eta_1, eta_2
            )

            # For more accurate normalization
            # delta_x = xs[1] - xs[0]
            # volume = delta_x * np.sum(ys)
            # ys = y * ys / volume

            ys += peak

        else:

            theta_1 = (
                360
                / (2 * jnp.pi)
                * jnp.arcsin(
                    jnp.sin(twotheta / 2 * 2 * jnp.pi / 360)
                    * lambda_K_alpha_1
                    / wavelength
                )
            )
            theta_2 = (
                360
                / (2 * jnp.pi)
                * jnp.arcsin(
                    jnp.sin(twotheta / 2 * 2 * jnp.pi / 360)
                    * lambda_K_alpha_2
                    / wavelength
                )
            )

            # if print_thetas:
            #    print(f"{2*theta_1} {2*theta_2}")

            peak_1 = (
                intensity
                * peak_function(xs / 2, theta_1, U, V, W, X, Y, eta_0, eta_1, eta_2)
                * 2
                / 3
            )
            peak_2 = (
                intensity
                * peak_function(xs / 2, theta_2, U, V, W, X, Y, eta_0, eta_1, eta_2)
                * 1
                / 3
            )

            ys += peak_1 + peak_2

            pass

    return ys


def fit_function(
    xs,
    a0=0.0,
    a1=0.0,
    a2=0.0,
    a3=0.0,
    a4=0.0,
    a5=0.0,
    U=0.001,
    V=-0.001,
    W=0.001,
    X=1.001,
    Y=0.001,
    eta_0=1.0,
    eta_1=0.0,
    eta_2=0.0,
    intensity_scaling=0.03,
    angles=None,
    intensities=None,
    K_alpha_splitting=True,
    print_thetas=False,
):

    # V = 0

    polynomial = (
        a0 + a1 * xs + a2 * xs**2 + a3 * xs**3 + a4 * xs**4 + a5 * xs**5
    )  # + a4 * xs ** 4 + a5 * xs ** 5

    # add the code from the simulation
    peaks = intensity_scaling * smeared_peaks(
        xs,
        angles,
        intensities,
        U,
        V,
        W,
        X,
        Y,
        eta_0,
        eta_1,
        eta_2,
        K_alpha_splitting=K_alpha_splitting,
        print_thetas=print_thetas,
    )

    return peaks + polynomial


fit_function_jit = jit(fit_function)


def fit_diffractogram(x, y, angles, intensities, do_plot=True):
    def fit_function_wrapped(
        xs,
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        U,
        V,
        W,
        X,
        Y,
        eta_0,
        eta_1,
        eta_2,
        intensity_scaling,
        **angles_intensities,
    ):
        values = list(angles_intensities.values())
        output = fit_function_jit(
            xs,
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            U,
            V,
            W,
            X,
            Y,
            eta_0,
            eta_1,
            eta_2,
            intensity_scaling,
            values[::2],
            values[1::2],
        )
        return output

    model = Model(fit_function_wrapped)

    strategy = ["all_minus_peak_pos_intensity", "peak_by_peak_plus_bg", "all"]
    use_extended_synchrotron_range = False

    # TODO: Add eta here, too!
    current_bestfits = [
        0.0,  # a0
        0.0,  # a1
        0.0,  # a2
        0.0,  # a3
        0.0,  # a4
        0.0,  # a5
        0.7,  # U
        0.0,  # V
        0.001,  # W
        1.001,  # X
        0.001,  # Y
        1.0,  # eta_0
        0.0,  # eta_1
        0.0,  # eta_2
        3.0 / 10.0,  # intensity_scaling
    ]  # default values
    for angle, intensity in zip(angles, intensities):
        current_bestfits.append(angle)
        current_bestfits.append(intensity)

    if do_plot:
        plt.plot(x, y, label="Original")
        for angle in angles:
            plt.axvline(x=angle, ymin=0.0, ymax=1.0, color="b", linewidth=0.1)

    initial_ys = fit_function_wrapped(
        x,
        *current_bestfits[:15],
        **dict(
            zip(
                [str(item) for item in range(len(current_bestfits[15:]))],
                current_bestfits[15:],
            )
        ),
    )
    scaler = np.max(initial_ys)
    initial_ys /= scaler

    if False:

        # TODO: rather try to use jit for this
        # TODO: rather try to do this on the GPU

        # test = jit(fit_function_wrapped)

        gradient = jacrev(fit_function_wrapped, argnums=range(1, 13))
        # gradient = jacrev(test, argnums=range(1, 13))

        for i in range(0, 10):
            test_grad = gradient(
                x,
                *current_bestfits[:15],
                **dict(
                    zip(
                        [str(item) for item in range(len(current_bestfits[15:]))],
                        current_bestfits[15:],
                    )
                ),
            )

        print(test_grad)

    current_bestfits[11] /= scaler

    if do_plot:
        plt.plot(
            x,
            initial_ys,
            label="Initial",
        )
        plt.show()

    for strategy_item in strategy:

        if strategy_item == "peak_by_peak_plus_bg":
            sub_steps = range(len(angles))
        else:
            sub_steps = [0]

        for sub_step in sub_steps:

            params = Parameters()

            vary_background = True
            if (
                strategy_item == "all"
                or strategy_item == "all_minus_peak_pos_intensity"
            ):
                vary_instr_parameters = True
            else:
                vary_instr_parameters = False

            if strategy_item == "all":
                vary_all_peaks = True
            else:
                vary_all_peaks = False

            params.add("a0", current_bestfits[0], vary=vary_background)
            params.add("a1", current_bestfits[1], vary=vary_background)
            params.add("a2", current_bestfits[2], vary=vary_background)
            params.add("a3", current_bestfits[3], vary=vary_background)
            params.add("a4", current_bestfits[4], vary=vary_background)
            params.add("a5", current_bestfits[5], vary=vary_background)
            params.add(
                "U", current_bestfits[6], min=0, max=3, vary=vary_instr_parameters
            )
            params.add(
                "V",
                current_bestfits[7],
                min=-1,
                max=0,
                # vary=vary_instr_parameters,
                vary=use_extended_synchrotron_range,
            )
            params.add(
                "W",
                current_bestfits[8],
                min=0,
                max=4,
                vary=vary_instr_parameters,
            )
            params.add(
                "X",
                current_bestfits[9],
                min=1 if not use_extended_synchrotron_range else 0,
                max=3,
                vary=vary_instr_parameters,
            )
            params.add(
                "Y", current_bestfits[10], min=0, max=3, vary=vary_instr_parameters
            )
            params.add("eta_0", current_bestfits[11], vary=vary_instr_parameters)
            params.add("eta_1", current_bestfits[12], vary=vary_instr_parameters)
            params.add("eta_2", current_bestfits[13], vary=vary_instr_parameters)
            params.add("intensity_scaling", current_bestfits[14], min=0, max=np.inf)

            i = 0
            for angle, intensity in zip(
                current_bestfits[15:][::2], current_bestfits[16:][::2]
            ):

                if strategy_item == "all_minus_peak_pos_intensity":
                    vary = False
                else:
                    vary = vary_all_peaks or (i == sub_step)

                params.add(
                    f"peak_pos_{i}",
                    angle,
                    min=angle - 2,
                    max=angle + 2,
                    vary=vary,
                )  # +- 2°
                params.add(
                    f"peak_int_{i}",
                    intensity,
                    min=intensity - intensity * 0.3,
                    max=intensity + intensity * 0.3,
                    vary=vary,
                )  # +- 30%

                i += 1

            result = model.fit(
                y,
                xs=x,
                params=params,
                # method="basinhopping",
            )

            current_bestfits = list(result.best_values.values())

            result_ys = fit_function_wrapped(
                x,
                *current_bestfits[:15],
                **dict(
                    zip(
                        [str(item) for item in range(len(current_bestfits[15:]))],
                        current_bestfits[15:],
                    )
                ),
            )

            score = r2_score(y, result_ys)
            print(f"R2 score: {score}")

        # if score < 0.6:
        #    print("Bad R2 score.")
        #    return None

        # Only plot after all peaks were fitted separately

        if do_plot:
            plt.plot(x, y, label="Original")
            plt.plot(x, result_ys, label="Fitted")

            plt.plot(
                x,
                current_bestfits[0]
                + current_bestfits[1] * x
                + current_bestfits[2] * x**2
                + current_bestfits[3] * x**3
                + current_bestfits[4] * x**4
                + current_bestfits[5] * x**5,
                label="BG",
            )

            plt.legend()
            plt.show()

    return params, score


def dif_parser(path):

    try:

        with open(path, "r") as file:
            content = file.readlines()

        relevant_content = []
        is_reading = False
        wavelength = None

        for line in content:

            if "X-RAY WAVELENGTH" in line:
                wavelength = float(line.replace("X-RAY WAVELENGTH:", "").strip())

            if "SPACE GROUP" in line:
                spg_specifier = (
                    line.replace("SPACE GROUP:", "")
                    .replace("ALTERNATE SETTING FOR", "")
                    .strip()
                    .replace("_", "")
                )

                spg_number = None

                if spg_specifier == "Fm3m":
                    spg_specifier = "Fm-3m"
                elif spg_specifier == "Pncm":
                    spg_number = 53
                elif spg_specifier == "C-1":
                    spg_number = 1
                elif (
                    spg_specifier == "P21/n"
                    or spg_specifier == "P21/a"
                    or spg_specifier == "P21/b"
                ):
                    spg_number = 14
                elif (
                    spg_specifier == "Pbnm"
                    or spg_specifier == "Pcmn"
                    or spg_specifier == "Pnam"
                ):
                    spg_number = 62
                elif spg_specifier == "Amma":
                    spg_number = 63
                elif spg_specifier == "Fd2d":
                    spg_number = 43
                elif spg_specifier == "Fd3m":
                    spg_specifier = "Fd-3m"
                elif (
                    spg_specifier == "A2/a"
                    or spg_specifier == "I2/a"
                    or spg_specifier == "I2/c"
                ):
                    spg_number = 15
                elif spg_specifier == "P4/n":
                    spg_number = 85
                elif spg_specifier == "I41/acd":
                    spg_number = 142
                elif spg_specifier == "I41/amd":
                    spg_number = 141
                elif spg_specifier == "Pmcn":
                    spg_number = 62
                elif spg_specifier == "I41/a":
                    spg_number = 88
                elif spg_specifier == "Pbn21" or spg_specifier == "P21nb":
                    spg_number = 33
                elif spg_specifier == "P2cm":
                    spg_number = 28
                elif spg_specifier == "P4/nnc":
                    spg_number = 126
                elif spg_specifier == "Pn21m":
                    spg_number = 31
                elif spg_specifier == "B2/b":
                    spg_number = 15
                elif spg_specifier == "Cmca":
                    spg_number = 64
                elif spg_specifier == "I2/m" or spg_specifier == "A2/m":
                    spg_number = 12
                elif spg_specifier == "Pcan":
                    spg_number = 60
                elif spg_specifier == "Ia3d":
                    spg_specifier = "Ia-3d"
                elif spg_specifier == "P4/nmm":
                    spg_number = 129
                elif spg_specifier == "Pa3":
                    spg_specifier = "Pa-3"
                elif spg_specifier == "P4/ncc":
                    spg_number = 130
                elif spg_specifier == "Imam":
                    spg_number = 74
                elif spg_specifier == "Pmmn":
                    spg_number = 59
                elif spg_specifier == "Pncn" or spg_specifier == "Pbnn":
                    spg_number = 52
                elif spg_specifier == "Bba2":
                    spg_number = 41
                elif spg_specifier == "C1":
                    spg_number = 1
                elif spg_specifier == "Pn3":
                    spg_specifier = "Pn-3"
                elif spg_specifier == "Fddd":
                    spg_number = 70
                elif spg_specifier == "Pcab":
                    spg_number = 61
                elif spg_specifier == "P2/a":
                    spg_number = 13
                elif spg_specifier == "Pmnb":
                    spg_number = 62
                elif spg_specifier == "I-1":
                    spg_number = 2
                elif spg_specifier == "Pmnb":
                    spg_number = 154
                elif spg_specifier == "B2mb":
                    spg_number = 40
                elif spg_specifier == "Im3":
                    spg_specifier = "Im-3"
                elif spg_specifier == "Pn21a":
                    spg_number = 33
                elif spg_specifier == "Pm2m":
                    spg_number = 25
                elif spg_specifier == "Fd3":
                    spg_specifier = "Fd-3"
                elif spg_specifier == "Im3m":
                    spg_specifier = "Im-3m"
                elif spg_specifier == "Cmma":
                    spg_number = 67
                elif spg_specifier == "Pn3m":
                    spg_specifier = "Pn-3m"
                elif spg_specifier == "F2/m":
                    spg_number = 12
                elif spg_specifier == "Pnm21":
                    spg_number = 31

                if spg_number is None:
                    spg_object = Group(spg_specifier)
                    spg_number = spg_object.number

            if (
                "==========" in line
                or "XPOW Copyright" in line
                or "For reference, see Downs" in line
            ) and is_reading:
                break

            if is_reading:
                relevant_content.append(line)

            if "2-THETA" in line and "INTENSITY" in line and "D-SPACING" in line:
                is_reading = True
            elif "2-THETA" in line and "D-SPACING" in line and not "INTENSITY" in line:
                print(f"Error processing file {path}:")
                print("No intensity data found.")
                return None, None, None

        data = np.genfromtxt(relevant_content)[:, 0:2]

        if wavelength is None:
            print(f"Error for file {path}:")
            print("No wavelength information found.")
            return None, None, None

        return data, wavelength, spg_number

    except Exception as ex:
        print(f"Error processing file {path}:")
        print(ex)
        return None, None, None


def get_rruff_patterns(
    reduced_resolution=True, only_refitted_patterns=True, only_if_dif_exists=False
):

    if only_refitted_patterns:
        with open("to_test_on.pickle", "rb") as file:
            raw_files = pickle.load(file)
    else:
        raw_files = glob("../RRUFF_data/XY_RAW/*.txt")

    xs = []
    ys = []
    dif_files = []

    for i, raw_file in enumerate(raw_files):

        raw_filename = os.path.basename(raw_file)
        raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")

        dif_file = dif_file[0]

        if len(raw_xy) == 0:
            print("Skipped empty pattern.")
            continue

        # Model:
        pattern_x_unet = np.arange(0, 90.24, 0.02)
        start_x_unet = pattern_x_unet[0]
        end_x_unet = pattern_x_unet[-1]
        N_unet = len(pattern_x_unet)  # UNet works without error for N ~ 2^model_depth

        x_test = raw_xy[:, 0]
        y_test = raw_xy[:, 1]

        # Remove nans:
        not_nan_indices = np.where(~np.isnan(x_test))[0]

        # nan_indices = np.where(np.isnan(x_test))[0] # This only happens for the first three indices for some of the patterns; so this is fine
        # if len(nan_indices > 0):
        #    print()

        x_test = x_test[not_nan_indices]
        y_test = y_test[not_nan_indices]

        not_nan_indices = np.where(~np.isnan(y_test))[0]

        # nan_indices = np.where(np.isnan(y_test))[0] # This actually doesn't happen at all
        # if len(nan_indices > 0):
        #    print()

        x_test = x_test[not_nan_indices]
        y_test = y_test[not_nan_indices]

        if not min(abs((x_test[0] % 0.02) - 0.02), abs(x_test[0] % 0.02)) < 0.0000001:
            print(f"Skipping pattern due to different x-steps.")
            continue

        if not np.all(np.diff(x_test) >= 0):  # not ascending
            print("Skipped pattern, inconsistent x axis.")
            continue

        dx = x_test[1] - x_test[0]
        if abs(dx - 0.01) < 0.0000001:  # allow some tollerance
            if do_unet_preprocessing or compare_UNet_heuristic:
                x_test = x_test[::2]
                y_test = y_test[::2]
        elif abs(dx - 0.02) < 0.0000001:
            print("")
            pass
        else:
            print(f"Skipping pattern with dx={dx}.")
            continue

        if (
            not do_unet_preprocessing and not compare_UNet_heuristic
        ):  # both need fitting dimensions for UNet
            dx = 0.01
            start_angle = 5
            end_angle = 90
        else:
            dx = 0.02
            start_angle = 0
            end_angle = 90.24

        y_test = np.array(y_test)
        y_test -= min(y_test)
        y_test = y_test / np.max(y_test)

        # For now don't use those:
        if x_test[0] > 5.0 or x_test[-1] < 90.0:
            continue

        if x_test[0] > start_angle:
            to_add = np.arange(0.0, x_test[0], dx)
            x_test = np.concatenate((to_add, x_test), axis=0)
            y_test = np.concatenate(
                (np.repeat([y_test[0]], len(to_add)), y_test), axis=0
            )

        if x_test[-1] < end_angle:
            to_add = np.arange(x_test[-1] + dx, end_angle, dx)
            x_test = np.concatenate((x_test, to_add), axis=0)
            y_test = np.concatenate(
                (y_test, np.repeat([y_test[-1]], len(to_add))), axis=0
            )

        # print(x_test[-300:])

        if (
            (do_unet_preprocessing or compare_UNet_heuristic) and len(x_test) != 4512
        ) or (
            (not do_unet_preprocessing and not compare_UNet_heuristic)
            and len(x_test) != 8501
        ):
            print("Skipping pattern due to wrong dimensions of xs.")
            continue

        dif_file = os.path.join(
            "../RRUFF_data/DIF/",
            "__".join(raw_filename.split("__")[:-2]) + "__DIF_File__*.txt",
        )
        dif_file = glob(dif_file)

        if len(dif_file) == 0 and only_if_dif_exists:
            dif_files.append(None)
            continue

    return patterns, difs
