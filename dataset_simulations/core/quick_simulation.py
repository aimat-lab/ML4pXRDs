# The core of the following code is taken from the pymatgen package.
# It is altered and optimized to run using numba.
# If performance is not a concern to you, you should use the original pymatgen code.

import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./")

from dataset_simulations.random_simulation_utils import generate_structures
import json
import os
from math import asin, cos, degrees, pi, radians, sin
import numpy as np
import collections
from pymatgen.io.cif import CifParser
import time
import numba
import matplotlib.pyplot as plt
from dataset_simulations.spectrum_generation.peak_broadening import BroadGen

SCALED_INTENSITY_TOL = 0.001
TWO_THETA_TOL = 1e-05

# TODO: Maybe rather use the range from UNet pattern generation
pymatgen_crystallite_size_gauss_min = 5
pymatgen_crystallite_size_gauss_max = 100

with open(
    os.path.join(os.path.dirname(__file__), "atomic_scattering_params.json")
) as f:
    ATOMIC_SCATTERING_PARAMS = json.load(f)


@numba.njit(cache=True, fastmath=True)
def __get_pattern_optimized(
    wavelength,
    zs,
    coeffs,
    fcoords,
    occus,
    dwfactors,
    recip_pts_sorted_0,
    recip_pts_sorted_1,
    recip_pts_sorted_2,
):

    peaks = {}
    two_thetas = []

    for i in range(0, len(recip_pts_sorted_0)):

        hkl = recip_pts_sorted_0[i]
        g_hkl = recip_pts_sorted_1[i]
        ind = recip_pts_sorted_2[i]

        # Force miller indices to be integers.
        hkl = [round(i) for i in hkl]
        if g_hkl != 0:

            # Bragg condition
            theta = asin(wavelength * g_hkl / 2)

            # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
            # 1/|ghkl|)
            s = g_hkl / 2

            # Store s^2 since we are using it a few times.
            s2 = s ** 2

            # Vectorized computation of g.r for all fractional coords and
            # hkl.
            hkl_temp = np.array([hkl], numba.types.float64)
            # hkl_temp = np.array([hkl], float)
            g_dot_r = np.dot(fcoords, hkl_temp.T).T[0]

            # Highly vectorized computation of atomic scattering factors.
            # Equivalent non-vectorized code is::
            #
            #   for site in structure:
            #      el = site.specie
            #      coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
            #      fs = el.Z - 41.78214 * s2 * sum(
            #          [d[0] * exp(-d[1] * s2) for d in coeff])
            fs = zs - 41.78214 * s2 * np.sum(
                coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1
            )

            dw_correction = np.exp(-dwfactors * s2)

            # Structure factor = sum of atomic scattering factors (with
            # position factor exp(2j * pi * g.r and occupancies).
            # Vectorized computation.
            f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r) * dw_correction)

            # Lorentz polarization correction for hkl
            lorentz_factor = (1 + cos(2 * theta) ** 2) / (sin(theta) ** 2 * cos(theta))

            # Intensity for hkl is modulus square of structure factor.
            i_hkl = (f_hkl * f_hkl.conjugate()).real

            two_theta = degrees(2 * theta)

            # Deal with floating point precision issues.
            ind = np.where(np.abs(np.array(two_thetas) - two_theta) < TWO_THETA_TOL)
            if len(ind[0]) > 0:
                peaks[two_thetas[ind[0][0]]] += i_hkl * lorentz_factor
            else:
                peaks[two_theta] = i_hkl * lorentz_factor
                two_thetas.append(two_theta)

    # Scale intensities so that the max intensity is 100.
    max_intensity = max([v for v in peaks.values()])
    x = []
    y = []
    for k in sorted(peaks.keys()):
        v = peaks[k]

        if v / max_intensity * 100 > SCALED_INTENSITY_TOL:
            x.append(k)
            y.append(v)

    y = np.array(y) / max(y)

    return x, y


def get_pattern_optimized(
    structure, wavelength, two_theta_range=(0, 90), do_print=False
):

    latt = structure.lattice
    debye_waller_factors = {}

    # Obtained from Bragg condition. Note that reciprocal lattice
    # vector length is 1 / d_hkl.
    min_r, max_r = (
        (0, 2 / wavelength)
        if two_theta_range is None
        else [2 * sin(radians(t / 2)) / wavelength for t in two_theta_range]
    )

    # Obtain crystallographic reciprocal lattice points within range
    recip_latt = latt.reciprocal_lattice_crystallographic
    recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
    if min_r:
        recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

    # Create a flattened array of zs, coeffs, fcoords and occus. This is
    # used to perform vectorized computation of atomic scattering factors
    # later. Note that these are not necessarily the same size as the
    # structure as each partially occupied specie occupies its own
    # position in the flattened array.
    zs = []
    coeffs = []
    fcoords = []
    occus = []
    dwfactors = []

    for site in structure:
        for sp, occu in site.species.items():
            zs.append(sp.Z)
            try:
                c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
            except KeyError:
                raise ValueError(
                    "Unable to calculate XRD pattern as "
                    "there is no scattering coefficients for"
                    " %s." % sp.symbol
                )
            coeffs.append(c)
            dwfactors.append(debye_waller_factors.get(sp.symbol, 0))
            fcoords.append(site.frac_coords)
            occus.append(occu)

    zs = np.array(zs)
    coeffs = np.array(coeffs)
    fcoords = np.array(fcoords)
    occus = np.array(occus)
    dwfactors = np.array(dwfactors)

    recip_pts_sorted = sorted(
        recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])
    )
    recip_pts_sorted = list(map(list, zip(*recip_pts_sorted)))

    start = time.time()
    result = __get_pattern_optimized(
        wavelength,
        zs,
        coeffs,
        fcoords,
        occus,
        dwfactors,
        recip_pts_sorted[0],
        recip_pts_sorted[1],
        recip_pts_sorted[2],
    )
    stop = time.time()
    if do_print:
        print("Took {} s for numba portion.".format(stop - start))

    return result


###################################### NON-OPTIMIZED #############################################


def get_unique_families_non_opt(hkls):
    """
    Returns unique families of Miller indices. Families must be permutations
    of each other.
    Args:
        hkls ([h, k, l]): List of Miller indices.
    Returns:
        {hkl: multiplicity}: A dict with unique hkl and multiplicity.
    """

    def is_perm(hkl1, hkl2):
        h1 = np.abs(hkl1)
        h2 = np.abs(hkl2)
        return all(i == j for i, j in zip(sorted(h1), sorted(h2)))

    unique = collections.defaultdict(list)
    for hkl1 in hkls:
        found = False
        for hkl2, v2 in unique.items():
            if is_perm(hkl1, hkl2):
                found = True
                v2.append(hkl1)
                break
        if not found:
            unique[hkl1].append(hkl1)

    pretty_unique = {}
    for k, v in unique.items():
        pretty_unique[sorted(v)[-1]] = len(v)

    return pretty_unique


def get_pattern(structure, wavelength, two_theta_range=(0, 90)):

    latt = structure.lattice
    is_hex = latt.is_hexagonal()
    debye_waller_factors = {}

    # Obtained from Bragg condition. Note that reciprocal lattice
    # vector length is 1 / d_hkl.
    min_r, max_r = (
        (0, 2 / wavelength)
        if two_theta_range is None
        else [2 * sin(radians(t / 2)) / wavelength for t in two_theta_range]
    )

    # Obtain crystallographic reciprocal lattice points within range
    recip_latt = latt.reciprocal_lattice_crystallographic
    recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
    if min_r:
        recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

    # Create a flattened array of zs, coeffs, fcoords and occus. This is
    # used to perform vectorized computation of atomic scattering factors
    # later. Note that these are not necessarily the same size as the
    # structure as each partially occupied specie occupies its own
    # position in the flattened array.
    zs = []
    coeffs = []
    fcoords = []
    occus = []
    dwfactors = []

    for site in structure:
        for sp, occu in site.species.items():
            zs.append(sp.Z)
            try:
                c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
            except KeyError:
                raise ValueError(
                    "Unable to calculate XRD pattern as "
                    "there is no scattering coefficients for"
                    " %s." % sp.symbol
                )
            coeffs.append(c)
            dwfactors.append(debye_waller_factors.get(sp.symbol, 0))
            fcoords.append(site.frac_coords)
            occus.append(occu)

    zs = np.array(zs)
    coeffs = np.array(coeffs)
    fcoords = np.array(fcoords)
    occus = np.array(occus)
    dwfactors = np.array(dwfactors)
    peaks = {}
    two_thetas = []

    for hkl, g_hkl, ind, _ in sorted(
        recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])
    ):
        # Force miller indices to be integers.
        hkl = [int(round(i)) for i in hkl]
        if g_hkl != 0:

            d_hkl = 1 / g_hkl

            # Bragg condition
            theta = asin(wavelength * g_hkl / 2)

            # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
            # 1/|ghkl|)
            s = g_hkl / 2

            # Store s^2 since we are using it a few times.
            s2 = s ** 2

            # Vectorized computation of g.r for all fractional coords and
            # hkl.
            g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

            # Highly vectorized computation of atomic scattering factors.
            # Equivalent non-vectorized code is::
            #
            #   for site in structure:
            #      el = site.specie
            #      coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
            #      fs = el.Z - 41.78214 * s2 * sum(
            #          [d[0] * exp(-d[1] * s2) for d in coeff])
            fs = zs - 41.78214 * s2 * np.sum(
                coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1
            )

            dw_correction = np.exp(-dwfactors * s2)

            # Structure factor = sum of atomic scattering factors (with
            # position factor exp(2j * pi * g.r and occupancies).
            # Vectorized computation.
            f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r) * dw_correction)

            # Lorentz polarization correction for hkl
            lorentz_factor = (1 + cos(2 * theta) ** 2) / (sin(theta) ** 2 * cos(theta))

            # Intensity for hkl is modulus square of structure factor.
            i_hkl = (f_hkl * f_hkl.conjugate()).real

            two_theta = degrees(2 * theta)

            if is_hex:
                # Use Miller-Bravais indices for hexagonal lattices.
                hkl = (hkl[0], hkl[1], -hkl[0] - hkl[1], hkl[2])

            # Deal with floating point precision issues.
            ind = np.where(np.abs(np.subtract(two_thetas, two_theta)) < TWO_THETA_TOL)
            if len(ind[0]) > 0:
                peaks[two_thetas[ind[0][0]]][0] += i_hkl * lorentz_factor
                peaks[two_thetas[ind[0][0]]][1].append(tuple(hkl))
            else:
                peaks[two_theta] = [i_hkl * lorentz_factor, [tuple(hkl)], d_hkl]
                two_thetas.append(two_theta)

    # Scale intensities so that the max intensity is 100.
    max_intensity = max([v[0] for v in peaks.values()])
    x = []
    y = []
    hkls = []
    d_hkls = []
    for k in sorted(peaks.keys()):
        v = peaks[k]
        fam = get_unique_families_non_opt(v[1])
        if v[0] / max_intensity * 100 > SCALED_INTENSITY_TOL:
            x.append(k)
            y.append(v[0])
            hkls.append(
                [{"hkl": hkl, "multiplicity": mult} for hkl, mult in fam.items()]
            )
            d_hkls.append(v[2])

    y = np.array(y) / np.max(y)

    return x, y


#######################################################################################################


@numba.njit(cache=True)
def calc_std_dev(two_theta, tau, wavelength):
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
    return sigma


@numba.njit(cache=True)
def smeared_peaks(xs, pattern_angles, pattern_intensities, domain_size, wavelength):

    ys = np.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        sigma = calc_std_dev(twotheta, domain_size, wavelength)

        peak = (
            intensity
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma ** 2) * (xs - twotheta) ** 2)
        )

        # delta_x = xs[1] - xs[0]
        # volume = delta_x * np.sum(ys)
        # ys = y * ys / volume

        ys += peak

    return ys / np.max(ys)


# timings:
timings_simulation_pattern = []
timings_simulation_smeared = []
timings_generation = []


def get_xy_patterns(
    structure, wavelength, xs, NO_corn_sizes=1, two_theta_range=(0, 90), do_print=False
):

    if do_print:
        start = time.time()
    angles, intensities = get_pattern_optimized(
        structure, wavelength, two_theta_range, False
    )
    if do_print:
        timings_simulation_pattern.append(time.time() - start)

    if do_print:
        start = time.time()

    result = []

    for i in range(0, NO_corn_sizes):
        smeared = smeared_peaks(xs, angles, intensities, np.random.uniform(
                        pymatgen_crystallite_size_gauss_min,
                        pymatgen_crystallite_size_gauss_max,
                    ), wavelength)
        result.append(smeared)

    if do_print:
        timings_simulation_smeared.append((time.time() - start)/NO_corn_sizes)

    return result


def get_random_xy_patterns(
    spgs,
    structures_per_spg,
    wavelength,
    N,
    NO_corn_sizes=1,
    two_theta_range=(0, 90),
    max_NO_elements=10,
    do_print=False,
    return_structures = False,
):

    result_patterns_y = []
    labels = []

    xs = np.linspace(two_theta_range[0], two_theta_range[1], N)

    for spg in spgs:
        if do_print:
            start = time.time()
        structures = generate_structures(spg, structures_per_spg, max_NO_elements)
        if do_print:
            timings_generation.append(time.time() - start)

        for structure in structures:
            
            try:

                patterns_ys = get_xy_patterns(
                    structure,
                    wavelength,
                    xs,
                    NO_corn_sizes,
                    two_theta_range,
                    do_print=do_print,
                )
            except Exception as ex:
                print("Error simulating pattern:")
                print(ex)
            else:
                labels.extend([spg]*NO_corn_sizes)
                result_patterns_y.extend(patterns_ys)

    if not return_structures:
        return result_patterns_y, labels
    else:
        return result_patterns_y, labels, structures

if __name__ == "__main__":

    test = get_random_xy_patterns([115], 1, 1.2, 9000, 5)

    plt.plot(test[0][3])
    plt.plot(test[0][2])
    plt.show()

    if False:
        # parser = CifParser("example.cif")
        # crystals = parser.get_structures()
        # crystal = crystals[0]

        structures = random_simulation_utils.generate_structures(223, 1)
        crystal = structures[0]

        total = 1

        start = time.time()
        for i in range(0, total):
            data_non_opt = get_pattern(
                crystal, 1.5406, (0, 90)  # for now, use Cu-K line
            )  # just specifying a different range / wavelength can yield a significant speed improvement, already!
        stop = time.time()
        time_non_optimized = (stop - start) / total

        data_opt = get_pattern_optimized(
            crystal, 1.5406, (0, 90), do_print=False
        )  # for now, use Cu-K line

        start = time.time()
        for i in range(0, total):
            data_opt = get_pattern_optimized(
                crystal, 1.5406, (0, 90)  # for now, use Cu-K line
            )  # just specifying a different range / wavelength can yield a significant speed improvement, already!
        stop = time.time()
        time_optimized = (stop - start) / total

        print("Took {} s for non-optimized version".format(time_non_optimized))
        print("Took {} s for optimized version".format(time_optimized))
        print(f"Optimized version is {time_non_optimized/time_optimized}x faster")

        difference_angles = np.sum(
            np.abs(np.array(data_opt[0]) - np.array(data_non_opt[0]))
        )
        difference_intensities = np.sum(
            np.abs(np.array(data_opt[1]) - np.array(data_non_opt[1]))
        )

        print("Numerical differences:")
        print(f"Angles: {difference_angles}")
        print(f"Intensities: {difference_intensities}")

    if False:

        structures = random_simulation_utils.generate_structures(223, 1)
        xs = np.linspace(10, 90, 8016)

        ys = get_xy_pattern(structures[0], 1.5, xs, 100, (10, 90))
        plt.plot(xs, ys, label="Optimized")

        broadener = BroadGen(
            structures[0],
            min_domain_size=pymatgen_crystallite_size_gauss_min,
            max_domain_size=pymatgen_crystallite_size_gauss_max,
            min_angle=10,
            max_angle=90,
            wavelength=1.5,
        )
        diffractogram, domain_size = broadener.broadened_spectrum(
            domain_size=100, N=8016
        )
        plt.plot(xs, diffractogram)
        plt.show()

    if False:

        repeat = 5

        patterns = get_random_xy_patterns(
            range(1, 30), 1, 1.5406, 8016, (10, 90), max_NO_elements=10
        )

        start = time.time()

        for i in range(0, repeat):
            print(i)
            patterns = get_random_xy_patterns(
                range(1, 231),
                1,
                1.5406,
                8016,
                (10, 90),
                max_NO_elements=25,  # Attention: 25
                do_print=True,
            )

        stop = time.time()
        print(f"{(stop-start)/repeat}s per swipe")

        print(f"Average timings generation: {np.mean(timings_generation)}")
        print(
            f"Average timings simulation pattern: {np.mean(timings_simulation_pattern)}"
        )
        print(
            f"Average timings simulation smeared: {np.mean(timings_simulation_smeared)}"
        )

        print(f"Max timings generation: {np.max(timings_generation)}")
        print(f"Max timings simulation pattern: {np.max(timings_simulation_pattern)}")
        print(f"Max timings simulation smeared: {np.max(timings_simulation_smeared)}")
