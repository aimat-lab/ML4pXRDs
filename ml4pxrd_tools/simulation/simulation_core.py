# Code to calculate positions and intensities of peaks in pXRD patterns.
# The following code is based on the pXRD simulation code from the `pymatgen` library.
# It is altered and optimized to run using numba.
# If performance is not a concern to you, you should use the original pymatgen implementation.

import os
import json
import numba
from math import asin, cos, degrees, pi, radians, sin
import numpy as np
import collections

# This is very useful for debugging:
if "NUMBA_DISABLE_JIT" in os.environ:
    is_debugging = os.environ["NUMBA_DISABLE_JIT"] == "1"
else:
    is_debugging = False

SCALED_INTENSITY_TOL = 0.001
TWO_THETA_TOL = 1e-05

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
            s2 = s**2

            # Vectorized computation of g.r for all fractional coords and
            # hkl.
            hkl_temp = np.array(
                [hkl], numba.types.float64 if not is_debugging else float
            )
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

    return x, y


def get_pattern_optimized(
    structure,
    wavelength,
    two_theta_range=(0, 90),
    return_max_unscaled_intensity_angle=False,
):
    """Simulate the pattern of a pymatgen `structure`.
    This is the optimized version using numba.

    Args:
        structure (pymatgen.core.structure): Crystal structure
        wavelength (float): wavelength in angstroms
        two_theta_range (tuple, optional): (min, max) tuple that defines the range in which to find peak positions and intensities. Defaults to (0, 90).
        return_max_unscaled_intensity_angle (bool, optional): Whether or not to output the maximum unscaled intensity and its angle.

    Returns:
        tuple: (np array of angles, np array of scaled intensities, (angle of maximum unscaled intensity, maximum unscaled intensity) if return_max_unscaled_intensity_angle)
    """

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

    x, y = __get_pattern_optimized(
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

    if not return_max_unscaled_intensity_angle:
        return x, np.array(y) / max(y)
    else:
        index = np.argmax(y)
        max_unscaled_intensity = y[index]
        max_unscaled_intensity_angle = x[index]

        return (
            x,
            np.array(y) / max(y),
            (max_unscaled_intensity, max_unscaled_intensity_angle),
        )


##### NON-OPTIMIZED code
# To make a direct comparison of the outputs possible, we also keep the original functions.
# They are directly taken from the `pymatgen` code.


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
    """Simulate the pattern of a pymatgen `structure`.
    This is the non-optimized version from `pymatgen`.

    Args:
        structure (_type_): pymatgen structure
        wavelength (_type_): wavelength in angstroms
        two_theta_range (tuple, optional): (min, max) tuple that defines the range in which to find peak positions and intensities. Defaults to (0, 90).

    Returns:
        tuple: (list of angles, list of intensities)
    """

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
            s2 = s**2

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
