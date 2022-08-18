import time
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numba
from train_dataset.interp_utils import spline_numba
from glob import glob
import os
import pickle
from dataset_simulations.simulation import Simulation
from train_dataset.utils.test_unet.rruff_helpers import get_rruff_patterns

n_angles_gp = 60
max_peaks_per_sample = 200  # max number of peaks per sample
# min_peak_height = 0.010
min_peak_height = 0

# GP parameters:
gp_scaling = 1.0
# variance = 10.0
# variance = 4.0

gp_min_variance = 3.0
gp_max_variance = 20.0

# for background to peaks ratio:
scaling_max = 15.0
use_uniform_scaling = False

use_fluct_noise = False
if not use_fluct_noise:
    base_noise_level_min = 0.0
    base_noise_level_max = 0.02

    fluct_noise_level_min = 0
    fluct_noise_level_max = 0
else:
    base_noise_level_min = 0.0
    # base_noise_level_min = 0.09
    base_noise_level_max = 0.09

    fluct_noise_level_min = 0.0
    # fluct_noise_level_min = 0.04
    fluct_noise_level_max = 0.04

# sigma_min = 0.1
# sigma_max = 0.5

crystallite_size_gauss_min = 15
# crystallite_size_gauss_max = 100
crystallite_size_gauss_max = (
    50  # TODO: maybe use this altered range for the classification / simulation, too!
)
# NOTE: When using the ICSD patterns directly from the simulation, then this
# range for crystallite sizes doesn't apply!

K_alpha_splitting = True  # only if caglioti is used
wavelength = 1.541838  # needed if K_alpha_splitting = True


########## Modified caglioti peak functions ##########


@numba.njit
def fn_H(theta, U, V, W):

    H_squared = (
        U * np.tan(theta / 360 * 2 * np.pi) ** 2
        + V * np.tan(theta / 360 * 2 * np.pi)
        + W
    )

    return np.sqrt(H_squared)


@numba.njit
def peak_function_pseudo_voigt(theta, mean, U, V, W, eta):

    H = fn_H(mean, U, V, W)
    # H_dash = fn_H_dash(mean, X, Y)
    H_dash = H

    sigma_gaussian = H / (
        2 * np.sqrt(2 * np.log(2))
    )  # with respect to the width in theta (see maple)!
    gamma_lorentzian = 0.5 * H_dash

    return eta * 1 / (sigma_gaussian * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((theta - mean) / sigma_gaussian) ** 2
    ) + (1 - eta) * 1 / (np.pi * gamma_lorentzian) * (
        gamma_lorentzian**2 / ((theta - mean) ** 2 + gamma_lorentzian**2)
    )


# Copper:
lambda_K_alpha_1 = 1.54056  # angstrom
lambda_K_alpha_2 = 1.54439  # angstrom


@numba.njit
def smeared_peaks_pseudo_voigt(
    xs,
    pattern_angles,
    pattern_intensities,
    U,
    V,
    W,
    eta,
):

    # Splitting Kalpha_1, Kalpha_2: https://physics.stackexchange.com/questions/398724/why-is-k-alpha-3-2-always-more-intense-than-k-alpha-1-2-in-copper
    # => ratio 2:1
    # Only the lorentz polarization correction depends on theta, can most likely be ignored
    # n * lambda = 2*d*sin(theta)
    # => lambda_1 / lambda_2 =sin(theta_1) / sin(theta_2)
    # => sin(theta_2) = sin(theta_1) * lambda_2 / lambda_1

    ys = np.zeros(len(xs))

    # for twotheta, intensity in zip(pattern_angles, pattern_intensities):
    for i in range(len(pattern_angles)):
        twotheta = pattern_angles[i]
        intensity = pattern_intensities[i]

        if not K_alpha_splitting:

            peak = intensity * peak_function_pseudo_voigt(
                xs / 2,
                twotheta / 2,
                U,
                V,
                W,
                eta,
            )

            # For more accurate normalization
            # delta_x = xs[1] - xs[0]
            # volume = delta_x * np.sum(ys)
            # ys = y * ys / volume

            ys += peak

        else:

            theta_1 = (
                360
                / (2 * np.pi)
                * np.arcsin(
                    np.sin(twotheta / 2 * 2 * np.pi / 360)
                    * lambda_K_alpha_1
                    / wavelength
                )
            )
            theta_2 = (
                360
                / (2 * np.pi)
                * np.arcsin(
                    np.sin(twotheta / 2 * 2 * np.pi / 360)
                    * lambda_K_alpha_2
                    / wavelength
                )
            )

            peak_1 = (
                intensity
                * peak_function_pseudo_voigt(xs / 2, theta_1, U, V, W, eta)
                * 2
                / 3
            )
            peak_2 = (
                intensity
                * peak_function_pseudo_voigt(xs / 2, theta_2, U, V, W, eta)
                * 1
                / 3
            )

            ys += peak_1 + peak_2

    return ys


@numba.njit
def smeared_peaks_pseudo_voigt_random(xs, pattern_angles, pattern_intensities):

    # H_squared = (
    #    U * np.tan(theta / 360 * 2 * np.pi) ** 2
    #    + V * np.tan(theta / 360 * 2 * np.pi)
    #    + W
    # )

    # U = np.random.uniform(0.0, 3.0)
    U = np.random.uniform(0.0, 0.1)
    V = 0.0
    # W = np.random.uniform(0.0, 4.0)
    W = np.random.uniform(0.0, 0.1)

    eta = np.random.uniform(0.0, 1.0)

    return smeared_peaks_pseudo_voigt(
        xs, pattern_angles, pattern_intensities, U, V, W, eta
    )


##########


def convert_to_discrete(
    x_range, peak_positions, peak_sizes, n_angles=4016, do_print=False
):

    pattern_x = np.linspace(x_range[0], x_range[1], n_angles)

    peak_info_disc = np.zeros(n_angles)
    peak_size_disc = np.zeros(n_angles)

    for i, peak_pos in enumerate(peak_positions):
        index = np.argwhere(pattern_x < peak_pos)[-1][0]

        while True:

            if peak_info_disc[index] == 0:

                peak_info_disc[index] += 1
                peak_size_disc[index] += peak_sizes[i]
                break

            else:

                if do_print:
                    print("Two or more peaks in the same spot!")
                index += 1  # move peak to the right

    return peak_info_disc, peak_size_disc


@numba.njit
def calc_std_dev(two_theta, tau, wavelength=1.207930):
    """
    calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.9  ## shape factor
    wavelength = wavelength * 0.1  ## angstrom to nm
    theta = np.radians(two_theta / 2.0)  ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau)  # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
    return sigma  # watch out!  this is not squared.


def generate_samples_gp(
    n_samples,
    x_range,
    n_angles_gp=n_angles_gp,
    n_angles_output=4016,
    scaling=gp_scaling,
    random_seed=None,
    icsd_patterns=None,
    original_range=False,
    use_caglioti=False,
    icsd_angles=None,
    icsd_intensities=None,
    use_ICSD_patterns=False,
):

    # x_test = np.linspace(10, 50, 1000)
    # plt.plot(x_test, calc_std_dev(x_test, 20))
    # plt.show()

    min_x, max_x = x_range

    # first, generate enough random functions using a gaussian process
    pattern_xs = np.linspace(min_x, max_x, n_angles_output)

    xs_gp = np.atleast_2d(np.linspace(min_x, max_x, n_angles_gp)).T

    ys_gp = np.zeros(shape=(n_angles_gp, n_samples))

    # Use the same variance for each batch, this should be fine
    variance = random.uniform(gp_min_variance, gp_max_variance)
    kernel = C(scaling, constant_value_bounds="fixed") * RBF(
        variance, length_scale_bounds="fixed"
    )
    gp = GaussianProcessRegressor(kernel=kernel)
    ys_gp = gp.sample_y(xs_gp, random_state=random_seed, n_samples=n_samples)

    if use_ICSD_patterns or use_caglioti:
        n_indices = len(icsd_patterns) if use_ICSD_patterns else len(icsd_intensities)
        indices_to_select = [random.randint(0, n_indices - 1) for i in range(n_samples)]

    # start = time.time()
    result = add_peaks(
        n_samples,
        n_angles_output,
        xs_gp,
        ys_gp,
        pattern_xs,
        min_x,
        max_x,
        icsd_patterns=[icsd_patterns[index] for index in indices_to_select]
        if use_ICSD_patterns
        else [
            np.random.random(8501)
        ],  # to please numba type system (TODO: Maybe file a bug)
        original_range=original_range,
        use_caglioti=use_caglioti,
        icsd_angles=[np.array(icsd_angles[index]) for index in indices_to_select]
        if use_caglioti
        else [np.random.random(100)],
        icsd_intensities=[
            np.array(icsd_intensities[index]) for index in indices_to_select
        ]
        if use_caglioti
        else [np.random.random(100)],
        use_icsd_patterns=use_ICSD_patterns,
    )

    if False:
        for i in range(len(result[0])):
            plt.plot(pattern_xs, result[0][i])
            plt.show()

    # stop = time.time()
    # print(f"Took {(stop-start)} for add_peaks")

    return result


@numba.njit
def samples_truncnorm(loc, scale, bounds):
    while True:
        s = np.random.normal(loc, scale)
        if bounds[0] <= s <= bounds[1]:
            break
    return s


@numba.njit
def add_peaks(
    n_samples,
    n_angles_output,
    xs_gp,
    ys_gp,
    pattern_xs,
    min_x,
    max_x,
    icsd_patterns=None,
    original_range=False,
    use_caglioti=False,
    icsd_angles=None,
    icsd_intensities=None,
    use_icsd_patterns=False,
):

    # gp.fit(np.atleast_2d([13]).T, np.atleast_2d([2]).T)
    # ys_gp = gp.sample_y(xs_gp, random_state=random_seed, n_samples=n_samples,)
    # ys_gp = ys_gp[:, 0, :]

    # for i in range(0, 10):
    #    plt.plot(xs_gp[:, 0], ys_gp[:, i])
    # plt.show()

    ys_altered_all = np.zeros(shape=(n_samples, n_angles_output))
    ys_unaltered_all = np.zeros(shape=(n_samples, n_angles_output))

    for i in range(0, n_samples):

        # scale the output of the gp to the desired length
        if n_angles_output != n_angles_gp:
            # f = ip.CubicSpline(xs_gp[:, 0], ys_gp[:, i], bc_type="natural")
            # ys_altered_all[i, :] = f(pattern_xs)

            ys_altered_all[i, :] = spline_numba(
                xs_gp[:, 0].copy(), ys_gp[:, i].copy(), pattern_xs
            )

        ys_altered_all[i, :] -= np.min(ys_altered_all[i, :])
        weight_background = np.sum(ys_altered_all[i, :])

        if use_uniform_scaling:
            scaling = np.random.uniform(0, scaling_max)
        else:
            scaling = samples_truncnorm(0, scaling_max / 2, [0, scaling_max])

        ys_altered_all[i, :] = ys_altered_all[i, :] / weight_background * 10 * scaling

        if not use_icsd_patterns and not use_caglioti:

            domain_size = np.random.uniform(
                crystallite_size_gauss_min, crystallite_size_gauss_max
            )
            # domain_size = crystallite_size_gauss_max

            peak_sizes = []

            NO_peaks = np.random.randint(1.0, max_peaks_per_sample)
            means = np.random.uniform(min_x, max_x, NO_peaks)
            sigma_peaks = calc_std_dev(means, domain_size)
            # peak_positions = means

            for j in range(0, NO_peaks):

                if j == 0:
                    peak_size = 1.0
                # elif random.random() < 0.3:
                #    peak_size = random.uniform(min_peak_height, 1)
                # loc = 0.1
                # scale = 0.3
                # peak_size = truncnorm.rvs(
                #    (min_peak_height - loc) / scale, (1 - loc) / scale, loc, scale
                # )

                ## change
                # if j == 0:
                #    peak_size = min_peak_height
                # elif j == 1:
                #    peak_size = 1.0
                else:

                    # peak_size = trunc.rvs()
                    peak_size = samples_truncnorm(0, 0.1, [min_peak_height, 1.0])

                ys_unaltered_all[i, :] += (
                    1
                    / (sigma_peaks[j] * np.sqrt(2 * np.pi))
                    * np.exp(
                        -1 / (2 * sigma_peaks[j] ** 2) * (pattern_xs - means[j]) ** 2
                    )
                ) * peak_size

                peak_sizes.append(peak_size)

            ys_unaltered_all[i, :] = ys_unaltered_all[i, :] / np.max(
                ys_unaltered_all[i, :]
            )

        elif not use_caglioti:

            if not original_range:
                ys_unaltered_all[i, 250:4501] = icsd_patterns[i][
                    ::2
                ]  # this has already been normalized during the simulation
            else:
                ys_unaltered_all[i, :] = icsd_patterns[i][:]

        else:

            ys_unaltered_all[i, :] = smeared_peaks_pseudo_voigt_random(
                pattern_xs, icsd_angles[i], icsd_intensities[i]
            )
            ys_unaltered_all[i, :] = ys_unaltered_all[i, :] / np.max(
                ys_unaltered_all[i, :]
            )

            # print(np.max(ys_unaltered_all[i, :]))
        ys_altered_all[i, :] += ys_unaltered_all[i, :]

        base_noise_level = np.random.uniform(base_noise_level_min, base_noise_level_max)
        # base_noise_level = base_noise_level_max

        ys_altered_all[i, :] += np.random.normal(0.0, base_noise_level, n_angles_output)

        fluct_noise_level = np.random.uniform(
            fluct_noise_level_min, fluct_noise_level_max
        )
        # fluct_noise_level = fluct_noise_level_max

        ys_altered_all[i, :] *= np.random.normal(
            1.0, fluct_noise_level, n_angles_output
        )

        ys_altered_all[i, :] -= np.min(ys_altered_all[i, :])

        normalizer = np.max(ys_altered_all[i, :])

        ys_altered_all[i, :] /= normalizer
        ys_unaltered_all[i, :] /= normalizer

    return ys_altered_all, ys_unaltered_all


if __name__ == "__main__":

    pattern_x = np.arange(0, 90.24, 0.02)
    start_x = pattern_x[0]
    end_x = pattern_x[-1]
    N = len(pattern_x)  # UNet works without error for N ~ 2^model_depth

    raw_files = glob("./RRUFF_data/XY_RAW/*.txt")

    with open("../dataset_simulations/prepared_training/meta", "rb") as file:
        data = pickle.load(file)

        per_element = data[6]

        counter_per_spg_per_element = data[0]
        if per_element:
            counts_per_spg_per_element_per_wyckoff = data[1]
        else:
            counts_per_spg_per_wyckoff = data[1]
        NO_wyckoffs_prob_per_spg = data[2]
        NO_unique_elements_prob_per_spg = data[3]

        if per_element:
            NO_repetitions_prob_per_spg_per_element = data[4]
        else:
            NO_repetitions_prob_per_spg = data[4]
        denseness_factors_per_spg = data[5]

        statistics_metas = data[7]
        statistics_labels = data[8]
        statistics_match_metas = data[9]
        statistics_match_labels = data[10]
        test_metas = data[11]
        test_labels = data[12]
        corrected_labels = data[13]
        test_match_metas = data[14]
        test_match_pure_metas = data[15]

    path_to_patterns = "../dataset_simulations/patterns/icsd_vecsei/"
    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        icsd_sim_statistics = Simulation(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
        icsd_sim_statistics.output_dir = path_to_patterns
    else:  # local
        icsd_sim_statistics = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        icsd_sim_statistics.output_dir = path_to_patterns

    statistics_match_metas_flat = [item[0] for item in statistics_match_metas]

    icsd_sim_statistics.load(
        load_only_N_patterns_each=1, metas_to_load=statistics_match_metas_flat, stop=4
    )
    statistics_patterns = [j for i in icsd_sim_statistics.sim_patterns for j in i]
    statistics_angles = icsd_sim_statistics.sim_angles
    statistics_intensities = icsd_sim_statistics.sim_intensities

    if True:
        xs_generated_cag, ys_generated_cag = generate_samples_gp(
            1,
            (start_x, end_x),
            n_angles_output=N,
            icsd_patterns=None,
            icsd_angles=statistics_angles[0:1],
            icsd_intensities=statistics_intensities[0:1],
            use_caglioti=True,
            use_ICSD_patterns=False,
        )
        xs_generated_comp, ys_generated_comp = generate_samples_gp(
            1,
            (start_x, end_x),
            n_angles_output=N,
            icsd_patterns=statistics_patterns[0:1],
            icsd_angles=None,
            icsd_intensities=None,
            use_caglioti=False,
            use_ICSD_patterns=True,
        )

        plt.plot(ys_generated_cag[0], label="Caglioti")
        plt.plot(ys_generated_comp[0], label="Gaussian (directly from simulation)")
        plt.legend()
        plt.show()

        exit()

    # For jit:
    _, _ = generate_samples_gp(
        1,
        (start_x, end_x),
        n_angles_output=N,
        icsd_patterns=statistics_patterns,
        icsd_angles=statistics_angles,
        icsd_intensities=statistics_intensities,
        use_caglioti=True,
    )
    start = time.time()
    xs_generated, ys_generated = generate_samples_gp(
        100,
        (start_x, end_x),
        n_angles_output=N,
        icsd_patterns=None,
        icsd_angles=statistics_angles,
        icsd_intensities=statistics_intensities,
        use_caglioti=True,
        use_ICSD_patterns=False,
    )
    print(time.time() - start)

    xs, ys, difs, raw_files = get_rruff_patterns(
        only_refitted_patterns=False,
        only_selected_patterns=True,
        only_if_dif_exists=True,
        start_angle=5.0,
        end_angle=90.0,
        reduced_resolution=False,
        return_refitted_parameters=False,
    )

    for i in range(len(ys)):
        x_test = xs[i]
        y_test = ys[i]

        y_test = np.array(y_test)
        y_test -= min(y_test)
        y_test = y_test / np.max(y_test)

        plt.plot(x_test, y_test, label="Experimental")
        plt.plot(pattern_x, xs_generated[i, :], label="Generated")
        plt.legend()
        plt.show()
