import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functools import partial

raw_files = glob("RUFF_data/XY_RAW/*.txt")
processed_files = []

processed_xys = []
raw_xys = []

counter = 0

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


def fit_function(xs, a0, a1, a2, a3, a4, a5, grain_size, angles, intensities, wavelength):

    polynomial = a0 + a1*xs + a2 * xs**2 + a3*xs**3 + a4*xs**4 + a5*xs**5

    # add the code from the simulation
    peaks = smeared_peaks(xs, angles, intensities, grain_size, wavelength)

    return peaks + polynomial

def fit_diffractogram():

    # TODO: If this doesn't work from the get-go, make an estimate of grain_size using the processed diffractogram first
    # TODO: Then only allow it to vary a little bit.
     
    # TODO: Add partial!
    params, covs = curve_fit(partial(fit_function, angles=angles, intensities=intensities, wavelength=intensities), x, y, bounds=(5.0,90.0)) # TODO: Are the bounds always like this or do they vary?

for i, raw_file in enumerate(raw_files):

    raw_filename = os.path.basename(raw_file)

    processed_file = os.path.join("RUFF_data/XY_Processed/", "__".join(raw_filename.replace("RAW", "Processed").split("__")[:-1]) + "*.txt")
    processed_file = glob(processed_file)

    if not len(processed_file) > 0:
        del raw_files[i]
        continue

    counter += 1

    processed_file = processed_file[0]
    processed_files.append(processed_file)

    raw_xy = np.genfromtxt(raw_file, dtype=float, delimiter=",", comments="#")
    raw_xys.append(raw_xy)

    processed_xy = np.genfromtxt(processed_file, dtype=float, delimiter=",", comments="#")
    processed_xys.append(processed_xy)

    # TODO: Read in the DIF file

    #difference = raw_xy[:,1] - processed_xy[:,1]

    #plt.plot(raw_xy[:,0], difference)
    #plt.plot(raw_xy[:,0], processed_xy[:,1])
    #plt.plot(raw_xy[:,0], raw_xy[:,1])

    #plt.show()

print(f"{counter} matching files found.")