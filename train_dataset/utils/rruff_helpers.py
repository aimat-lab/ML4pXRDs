import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from lmfit import Model
from pyxtal.symmetry import Group

########## Peak profile functions from https://en.wikipedia.org/wiki/Rietveld_refinement :
# Parameter ranges from: file:///home/henrik/Downloads/PowderDiff26201188-93%20(1).pdf


def fn_x(theta, mean, H):
    return (2 * theta - 2 * mean) / H


def fn_H(theta, U, V, W):

    H_squared = (
        U * np.tan(theta / 360 * 2 * np.pi) ** 2
        + V * np.tan(theta / 360 * 2 * np.pi)
        + W
    )

    # return np.nan_to_num(np.sqrt(H_squared)) + 0.0001 # TODO: Changed
    return np.sqrt(H_squared)


def fn_H_dash(theta, X, Y):
    return X / np.cos(theta / 360 * 2 * np.pi) + Y * np.tan(theta / 360 * 2 * np.pi)


def fn_eta(theta, eta_0, eta_1, eta_2):
    return eta_0 + eta_1 * 2 * theta + eta_2 * theta**2


def peak_function(theta, mean, U, V, W, X, Y):

    C_G = 4 * np.log(2)
    C_L = 4

    H = fn_H(theta, U, V, W)
    H_dash = fn_H_dash(theta, X, Y)

    # eta = fn_eta(theta, eta_0, eta_1, eta_2)

    Lambda = (
        H**5
        + 2.69269 * H**4 * H_dash
        + 2.42843 * H**3 * H_dash**2
        + 4.47163 * H**2 * H_dash**3
        + 0.07842 * H * H_dash**4
        + H_dash**5
    ) ** (1 / 5)

    eta = (
        1.36603 * (H_dash / Lambda)
        - 0.47719 * (H_dash / Lambda) ** 2
        + 0.11116 * (H_dash / Lambda) ** 3
    )

    x = fn_x(theta, mean, H)

    return eta * C_G ** (1 / 2) / (np.sqrt(np.pi) * H) * np.exp(-1 * C_G * x**2) + (
        1 - eta
    ) * C_L ** (1 / 2) / (np.sqrt(np.pi) * H_dash) * (1 + C_L * x**2) ** (-1)


def smeared_peaks(xs, pattern_angles, pattern_intensities, U, V, W, X, Y):

    ys = np.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        peak = intensity * peak_function(xs / 2, twotheta / 2, U, V, W, X, Y)

        # For more accurate normalization
        # delta_x = xs[1] - xs[0]
        # volume = delta_x * np.sum(ys)
        # ys = y * ys / volume

        ys += peak

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
    intensity_scaling=0.03,
    angles=None,
    intensities=None,
):

    # V = 0

    polynomial = (
        a0 + a1 * xs + a2 * xs**2 + a3 * xs**3 + a4 * xs**4 + a5 * xs**5
    )  # + a4 * xs ** 4 + a5 * xs ** 5

    # add the code from the simulation
    peaks = intensity_scaling * smeared_peaks(xs, angles, intensities, U, V, W, X, Y)

    return peaks + polynomial


def fit_diffractogram(x, y, angles, intensities):

    plt.plot(x, y, label="Original")
    for angle in angles:
        plt.axvline(x=angle, ymin=0.0, ymax=1.0, color="b", linewidth=0.1)

    plt.plot(
        x, fit_function(x, angles=angles, intensities=intensities), label="Initial"
    )

    params, covs = curve_fit(
        partial(fit_function, angles=angles, intensities=intensities),
        x,
        y,
        maxfev=100000,
        bounds=(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -1, 0, 0, 0, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 3, 0, 4, 3, 3, np.inf],
        ),
    )

    """
    def fit_function_wrapped(xs,
        a0,
        a1,
        a2,
        a3,
        # a4,
        # a5,
        U,
        V,
        W,
        X,
        Y,
        eta_0,
        eta_1,
        eta_2,
        intensity_scaling,):

        output = fit_function(xs, a0, a1, a2, a3, U, V, W, X, Y, eta_0, eta_1, eta_2, intensity_scaling, angles, intensities)

        #if np.any(np.isnan(output)):
        #    print(repr([a0,a1,a2,a3,U,V,W,X,Y,eta_0, eta_1, eta_2, intensity_scaling]))

        print(output.shape)
        return output

    #fit_function_wrapped(np.linspace(5,90,1000), 1.7026161561556243, -0.1690326633239259, 0.0025585116643204145, -1.0123896779971408e-05, 0.8131574470688545, -1.7071279898103882, 0.7472324773831862, 0.2226751470996659, 0.772880962746507, 2193.5532476458857, -156.5848565793276, -265.9168201692516, 11.05304099502265)

    #model = Model(fit_function_wrapped, nan_policy="omit")
    # TODO: Fix x range
    #model.fit(y, xs=x, a0=1.0, a1=1.0, a2=1.0, a3=1.0, U=1.0, V=1.0, W=1.0, X=1.0, Y=1.0, eta_0 = 1.0, eta_1 = 1.0, eta_2 = 1.0, intensity_scaling=1.0)
    #exit()

    """

    fitted_curve = fit_function(x, *params, angles, intensities)

    score = r2_score(y, fitted_curve)
    print(f"R2 score: {score}")

    # if score < 0.6:
    #    print("Bad R2 score.")
    #    return None

    plt.plot(x, fitted_curve, label="Fitted")

    plt.plot(
        x,
        params[0]
        + params[1] * x
        + params[2] * x**2
        + params[3] * x**3
        + params[4] * x**4
        + params[5] * x**5,
        label="BG",
    )

    plt.legend()
    plt.show()

    return params


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
