import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functools import partial
from sklearn.metrics import r2_score


def calc_std_dev(two_theta, tau, wavelength):

    ## Calculate FWHM based on the Scherrer equation
    K = 0.9  ## shape factor
    wavelength = wavelength * 0.1  ## angstrom to nm
    theta = np.radians(two_theta / 2.0)  ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau)  # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
    return sigma


def smeared_peaks(xs, pattern_angles, pattern_intensities, domain_size, wavelength):

    ys = np.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        sigma = calc_std_dev(twotheta, domain_size, wavelength)

        peak = (
            intensity
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - twotheta) ** 2)
        )

        # For more accurate normalization
        # delta_x = xs[1] - xs[0]
        # volume = delta_x * np.sum(ys)
        # ys = y * ys / volume

        ys += peak

    return ys / np.max(ys)


# TODO: Add lorentzian grain size / pseudo-voigt for peaks!
# Use this https://lmfit.github.io/lmfit-py/builtin_models.html
# Also use this to do the fitting.


def fit_function(
    xs,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    grain_size,
    intensity_scaling,
    angles,
    intensities,
    wavelength,
):

    polynomial = (
        a0 + a1 * xs + a2 * xs ** 2 + a3 * xs ** 3 + a4 * xs ** 4 + a5 * xs ** 5
    )

    # add the code from the simulation
    peaks = intensity_scaling * smeared_peaks(
        xs, angles, intensities, grain_size, wavelength
    )

    return peaks + polynomial


def fit_diffractogram(x, y, angles, intensities, wavelength):

    plt.plot(x, y)
    for angle in angles:
        plt.axvline(x=angle, ymin=0.0, ymax=1.0, color="b", linewidth=0.1)

    params, covs = curve_fit(
        partial(
            fit_function, angles=angles, intensities=intensities, wavelength=wavelength
        ),
        x,
        y,
    )

    fitted_curve = fit_function(x, *params, angles, intensities, wavelength)

    score = r2_score(y, fitted_curve)
    print(f"R2 score: {score}")

    if score < 0.6:
        print("Bad R2 score.")
        return None

    plt.plot(x, fitted_curve)

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
        processed_file = processed_file[0]
        processed_files.append(processed_file)
    else:
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

    if data is not None:  # if nothing went wrong

        angles.append(data[:, 0])
        intensities.append(data[:, 1])

        result = fit_diffractogram(
            raw_xys[-1][:, 0],
            raw_xys[-1][:, 1],
            angles[-1],
            intensities[-1],
            wavelength,
        )

    else:

        angles.append(None)
        intensities.append(None)

    """
        processed_xy = np.genfromtxt(
        processed_file, dtype=float, delimiter=",", comments="#"
    )
    processed_xys.append(processed_xy)
    """

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