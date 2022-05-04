import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functools import partial
from sklearn.metrics import r2_score
from lmfit import Model
import random

########## Peak profile functions from https://en.wikipedia.org/wiki/Rietveld_refinement :

def fn_x(theta, mean, H):
    return (2 * theta - 2 * mean) / H

# Constraint: U*tan_min**2 + V*tan_min + W > 0, U*tan_max**2 + V*tan_max + W > 0

def fn_H(theta, U, V, W):

    H_squared = U * np.tan(theta / 360 * 2 * np.pi) ** 2 + V * np.tan(theta / 360 * 2 * np.pi) + W

    #return np.nan_to_num(np.sqrt(H_squared)) + 0.0001 # TODO: Changed
    return np.sqrt(H_squared)


def fn_H_dash(theta, X, Y):
    return X / np.cos(theta / 360 * 2 * np.pi) + Y * np.tan(theta / 360 * 2 * np.pi)


def fn_eta(theta, eta_0, eta_1, eta_2):
    return eta_0 + eta_1 * 2 * theta + eta_2 * theta ** 2


# TODO: How to make eta 0 <= eta <= 1???
def peak_function(theta, mean, U, V, W, X, Y, eta_0, eta_1, eta_2):

    C_G = 4 * np.log(2)
    C_L = 4

    eta = fn_eta(theta, eta_0, eta_1, eta_2)
    H = fn_H(theta, U, V, W)
    H_dash = fn_H_dash(theta, X, Y)

    x = fn_x(theta, mean, H)

    return eta * C_G ** (1 / 2) / (np.sqrt(np.pi) * H) * np.exp(-1 * C_G * x ** 2) + (
        1 - eta
    ) * C_L ** (1 / 2) / (np.sqrt(np.pi) * H_dash) * (1 + C_L * x ** 2) ** (-1)


def smeared_peaks(
    xs, pattern_angles, pattern_intensities, U, V, W, X, Y, eta_0, eta_1, eta_2
):

    ys = np.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        peak = intensity * peak_function(
            xs / 2, twotheta / 2, U, V, W, X, Y, eta_0, eta_1, eta_2
        )

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
    # a4,
    # a5,
    U=0.001,
    V=0.001,
    W=0.001,
    X=0.001,
    Y=0.001,
    eta_0=0.0,
    eta_1=0.0,
    eta_2=0.0,
    intensity_scaling=1.0,
    angles=None,
    intensities=None,
):

    polynomial = (
        a0 + a1 * xs + a2 * xs ** 2 + a3 * xs ** 3
    )  # + a4 * xs ** 4 + a5 * xs ** 5

    # add the code from the simulation
    peaks = intensity_scaling * smeared_peaks(
        xs, angles, intensities, U, V, W, X, Y, eta_0, eta_1, eta_2
    )

    return peaks + polynomial


def fit_diffractogram(x, y, angles, intensities):

    plt.plot(x, y, label="Original")
    for angle in angles:
        plt.axvline(x=angle, ymin=0.0, ymax=1.0, color="b", linewidth=0.1)

    plt.plot(x, fit_function(x, angles=angles, intensities=intensities), label="Initial")

    # TODO: Search for reasonable parameter ranges; maybe make a good estimate of the background function based on what you already have (processed)?
    params, covs = curve_fit(
        partial(fit_function, angles=angles, intensities=intensities),
        x,
        y,
        maxfev=100000,
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

    plt.plot(x, params[0] + params[1] * x + params[2] * x ** 2 + params[3] * x ** 3, label="BG")

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
                return None, None

        data = np.genfromtxt(relevant_content)[:, 0:2]

        if wavelength is None:
            print(f"Error for file {path}:")
            print("No wavelength information found.")
            return None, None

        return data, wavelength

    except Exception as ex:
        print(f"Error processing file {path}:")
        print(ex)
        return None, None


raw_files = glob("../RRUFF_data/XY_RAW/*.txt")
processed_files = []
dif_files = []


processed_xys = []

raw_xys = []

angles = []
intensities = []


counter_processed = 0
counter_dif = 0

random.shuffle(raw_files)

for i, raw_file in enumerate(raw_files):

    print(f"{(i+1)/len(raw_files)*100:.2f}% processed")

    raw_filename = os.path.basename(raw_file)
    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")
    raw_xys.append(raw_xy)

    processed_file = os.path.join(
        "../RRUFF_data/XY_Processed/",
        "__".join(raw_filename.replace("RAW", "Processed").split("__")[:-1]) + "*.txt",
    )
    processed_file = glob(processed_file)
    if len(processed_file) > 0:
        counter_processed += 1
        found_processed_file = True
        processed_file = processed_file[0]
        processed_files.append(processed_file)
    else:
        found_processed_file = False
        processed_files.append(None)
        pass

    dif_file = os.path.join(
        "../RRUFF_data/DIF/",
        "__".join(raw_filename.split("__")[:-2]) + "__DIF_File__*.txt",
    )
    dif_file = glob(dif_file)

    data = None
    if len(dif_file) > 0:
        counter_dif += 1
        dif_file = dif_file[0]
        dif_files.append(dif_file)

        data, wavelength = dif_parser(dif_file)

    else:

        dif_files.append(None)

    if found_processed_file:

        processed_xy = np.genfromtxt(
            processed_file, dtype=float, delimiter=",", comments="#"
        )
        processed_xys.append(processed_xy)

    if True and data is not None: # and found_processed_file:  # if nothing went wrong

        angles.append(data[:, 0])
        intensities.append(data[:, 1])

        #plt.plot(processed_xy[:,0][::10], processed_xy[:,1][::10], label="Processed")
        #plt.plot(raw_xy[:,0][::10], raw_xy[:,1][::10], label="Raw")
        #plt.plot(raw_xy[:,0][::10], raw_xy[:,1][::10] - processed_xy[:,1][::10], label="BG")
        #plt.legend()
        #plt.show()

        result = fit_diffractogram(
            raw_xys[-1][:, 0],
            raw_xys[-1][:, 1] / np.max(raw_xys[-1][:,1]),
            #processed_xy[:,0],
            #processed_xy[:,1] / np.max(processed_xy[:,1]),
            angles[-1],
            intensities[-1] / np.max(intensities[-1]),
        )

    else:

        angles.append(None)
        intensities.append(None)

    pass

    # This alone is not enough, unfortunately:
    # difference = raw_xy[:,1] - processed_xy[:,1]
    # plt.plot(raw_xy[:,0], difference)
    # plt.plot(raw_xy[:,0], processed_xy[:,1])
    # plt.plot(raw_xy[:,0], raw_xy[:,1])
    # plt.show()

print(f"{counter_processed} processed files found.")
print(f"{counter_dif} dif files found.")

assert len(dif_files) == len(processed_files)

counter_both = 0
for i, dif_file in enumerate(dif_files):
    if dif_file is not None and processed_files[i] is not None:
        counter_both += 1

print(f"{counter_both} files with dif and processed file found.")
