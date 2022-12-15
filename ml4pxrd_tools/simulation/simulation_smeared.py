# Contains functions to generate patterns with smeared peaks.
# This code uses simulation_core.py to simulate the peak positions and intensities.
# We also provide functions to generate a synthetic crystal and simulate it in one go.

from ml4pxrd_tools.generation.structure_generation import generate_structures
from ml4pxrd_tools.simulation.simulation_core import get_pattern_optimized
import numpy as np
import time
import numba
import matplotlib.pyplot as plt
import traceback
from sklearn.linear_model import LinearRegression
import ml4pxrd_tools.matplotlib_defaults as matplotlib_defaults

# Range of crystallite sizes to choose uniformly from (nm)
pymatgen_crystallite_size_gauss_min = 20
pymatgen_crystallite_size_gauss_max = 100


@numba.njit(cache=True)
def calc_std_dev(two_theta, tau, wavelength):
    # This function originates from https://github.com/njszym/XRD-AutoAnalyzer
    """
    Calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
        wavelength: anstrom
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
def smeared_peaks(
    xs,
    pattern_angles,
    pattern_intensities,
    domain_size,
    wavelength,
):
    """This function can be used to generate a diffractogram with broadened peaks
    with the angles and intensities of the peaks as input. It essentially computes
    a convolution with gaussian peak profiles. The FWHM of the gaussian peaks
    is determined by the crystallite size (`domain_size`).

    Args:
        xs (numpy array): Points (2-theta) on which to evaluate the diffractogram.
        pattern_angles (list): List of angles of peaks
        pattern_intensities (list): List of intensities of peaks
        domain_size (float): crystallite size
        wavelength (float): x-ray wavelength in angstroms. Needed to calculate the FWHM of the gaussians based on the crystallite size.

    Returns:
        numpy array: 1D array of intensities at the given positions `xs`
    """

    ys = np.zeros(len(xs))

    for twotheta, intensity in zip(pattern_angles, pattern_intensities):

        sigma = calc_std_dev(twotheta, domain_size, wavelength)
        peak = (
            intensity
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 / (2 * sigma**2) * (xs - twotheta) ** 2)
        )
        ys += peak

    return ys / np.max(ys)


def get_smeared_patterns(
    structure,
    wavelength,
    xs,
    NO_corn_sizes=1,
    two_theta_range=(0, 90),
    return_corn_sizes=False,
    return_angles_intensities=False,
    return_max_unscaled_intensity_angle=False,
    do_benchmark=False,
    timings_simulation_pattern=None,
    timings_simulation_smeared=None,
):
    """Get a smeared (gaussian peaks) of the given structure.
    Crystallite sizes will be uniformly chosen.

    Args:
        structure (pymatgen.core.structure): Crystal structure
        wavelength (float): wavelength in angstroms
        xs (numpy array): Array of 2-theta points on which to evaluate the diffractogram.
        NO_corn_sizes (int, optional): How many patterns with different crystallite sizes should be generated? Defaults to 1.
        two_theta_range (tuple, optional): 2-theta range in which to simulate peaks. Defaults to (0, 90).
        return_corn_sizes (bool, optional): Whether or not to return the chosen corn sizes of each pattern. Defaults to False.
        return_angles_intensities (bool, optional): Whether or not to additionally return the angles and intensities of all peaks. Defaults to False.
        return_max_unscaled_intensity_angle (bool, optional): Whether or not to additionally return the angle of the maximum unscaled intensity. Defaults to False.
        do_benchmark (bool, optional): Should the time it takes to generate and simulate the patterns be recorded in `timings_simulation_pattern` and `timings_simulation_smeared`?.
            Defaults to False.
        timings_simulation_pattern (list, optional): List to store the timings of the simulation in. Defaults to None.
        timings_simulation_smeared (list, optional): List to store the timings of the smearing in. Defaults to None.

    Returns:
        tuple: (patterns, [corn_sizes], [angles], [intensities], [return_max_unscaled_intensity_angle])
        If only patterns is returned, the output is simply the list of patterns, not a tuple.
    """

    if return_corn_sizes:
        corn_sizes = []

    if do_benchmark:
        start = time.time()

    if not return_max_unscaled_intensity_angle:
        angles, intensities = get_pattern_optimized(
            structure, wavelength, two_theta_range, False
        )
    else:
        angles, intensities, max_unscaled_intensity_angle = get_pattern_optimized(
            structure, wavelength, two_theta_range, False, True
        )

    if do_benchmark:
        timings_simulation_pattern.append(time.time() - start)

    if do_benchmark:
        start = time.time()

    results = []

    for i in range(0, NO_corn_sizes):

        corn_size = np.random.uniform(
            pymatgen_crystallite_size_gauss_min,
            pymatgen_crystallite_size_gauss_max,
        )

        smeared = smeared_peaks(xs, angles, intensities, corn_size, wavelength)
        results.append(smeared)

        if return_corn_sizes:
            corn_sizes.append(corn_size)

    if do_benchmark:
        timings_simulation_smeared.append((time.time() - start) / NO_corn_sizes)

    if not return_corn_sizes:
        if return_angles_intensities:
            if not return_max_unscaled_intensity_angle:
                return results, angles, intensities
            else:
                return results, angles, intensities, max_unscaled_intensity_angle
        else:
            if not return_max_unscaled_intensity_angle:
                return results
            else:
                return results, max_unscaled_intensity_angle
    else:
        if return_angles_intensities:
            if not return_max_unscaled_intensity_angle:
                return results, corn_sizes, angles, intensities
            else:
                return (
                    results,
                    corn_sizes,
                    angles,
                    intensities,
                    max_unscaled_intensity_angle,
                )
        else:
            if not return_max_unscaled_intensity_angle:
                return results, corn_sizes
            else:
                return results, corn_sizes, max_unscaled_intensity_angle


def get_synthetic_smeared_patterns(
    spgs,
    N_structures_per_spg,
    wavelength,
    two_theta_range=(0, 90),
    N=8501,
    NO_corn_sizes=1,
    max_NO_atoms_asymmetric_unit=100,
    max_volume=7000,
    fixed_volume=None,
    probability_per_spg_per_element=None,
    probability_per_spg_per_element_per_wyckoff=None,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg_per_element=None,
    per_element=False,
    do_symmetry_checks=True,
    denseness_factors_density_per_spg=None,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    seed=-1,
    is_verbose=False,
    return_structures_and_corn_sizes=False,
    group_object_per_spg=None,
    probability_per_spg=None,
    do_benchmark=False,
    timings_simulation_pattern=None,
    timings_simulation_smeared=None,
    timings_generation=None,
    NO_wyckoffs_log=None,
):
    """Return smeared patterns based on randomly generated synthetic crystals.

    Args:
        spgs (list of int): List of spgs to process. If probability_per_spg is not None,
            this list will be overwritten by randomly sampled spg labels.
        N_structures_per_spg (int): How many structures per spg to generate?
        wavelength (float): angstroms
        two_theta_range (tuple, optional): 2-theta range to use for simulation. Defaults to (0, 90).
        N (int, optional): Number of datapoints in the 2-theta range. Defaults to 8501.
        NO_corn_sizes (int, optional): How many different crystallite sizes should be used
            to generate smeared patterns for each generated synthetic crystal? Defaults to 1.
        max_NO_atoms_asymmetric_unit (int, optional): Maximum number of atoms in the asymmetric unit of the crystal. Defaults to 100.
        max_volume (float, optional): Maximum volume of the conventional unit cell of the crystal. Defaults to 7000.
        fixed_volume (float, optional): If not None, the volume of all crystals will be set to `fixed_volume`. Defaults to None.
        probability_per_spg_per_element (optional): Indexed using [spg][element]. Returns the probability that an element of the given space group
            occurrs in a crystal. Defaults to None.
        probability_per_spg_per_element_per_wyckoff (optional): Indexed using [spg][element][wyckoff_index]. Returns the probability
            for the given spg that the given element is placed on the given wyckoff position. Defaults to None.
        NO_unique_elements_prob_per_spg (optional): Indexed using [spg]. For the given spg, return a list of probabilities
            that the number of unique elements in a crystal is equal to the list index + 1. Defaults to None.
        NO_repetitions_prob_per_spg_per_element (optional): Indexed using [spg][element]. For the given spg and element, return a list of probabilities
            that the number of appearances of the element on wyckoff sites in a crystal is equal to the list index + 1. Defaults to None.
        per_element (bool, optional): If this setting is True, NO_repetitions_prob_per_spg_per_element and probability_per_spg_per_element_per_wyckoff
            are indexed using the element. If False, they are independent of the element. Defaults to False.
        do_symmetry_checks (bool, optional): Whether or not to check the spg of the resulting crystals using `spglib`. Defaults to True.
        denseness_factors_density_per_spg (dict of scipy.stats.kde.gaussian_kde, optional): Dictionary of KDEs to generate the denseness factor for each spg. Defaults to None.
        denseness_factors_conditional_sampler_seeds_per_spg (dict of tuple, optional): dictionary containing tuple for each spg:
        lattice_paras_density_per_lattice_type (dict of scipy.stats.kde.gaussian_kde): Dictionary yielding the KDE for each lattice type.
            seed (int, optional): Seed to initialize the random generators. If -1, no seed is used. Defaults to -1.
            is_verbose (bool, optional): Whether or not to print additional info. Defaults to False.
        return_structures_and_corn_sizes (bool, optional): Wether or not to additionally return the generated structures and corn sizes. Defaults to False.
        group_object_per_spg (dict of pyxtal.symmetry.Group, optional): Pass in a group object for each spg to speed up the generation. Defaults to None.
        probability_per_spg (dict of floats, optional): If this is not None, spgs will be sampled using this categorical distribution.
            The dictionary returns for each spg index the probability to be chosen. Defaults to None.
        do_benchmark (bool, optional): Should the time it takes to generate and simulate the patterns be recorded in `timings_simulation_pattern` and `timings_simulation_smeared`?.
            Defaults to False.
        timings_simulation_pattern (list, optional): List to store the timings of the simulation in. Defaults to None.
        timings_simulation_smeared (list, optional): List to store the timings of the smearing in. Defaults to None.
        timings_generation (list, optional): List to store the timings of the generation in. Defaults to None.
        NO_wyckoffs_log (list, optional): List to store the number of atoms in the asymmetric for each of the generated crystals in. Defaults to None.

    Returns:
        tuple: (list of patterns, list of labels, [list of structures], [list of corn sizes])
    """

    result_patterns_y = []
    labels = []
    all_corn_sizes = []
    all_structures = []

    xs = np.linspace(two_theta_range[0], two_theta_range[1], N)

    for spg in spgs:

        structures = []
        current_spgs = []
        for _ in range(N_structures_per_spg):

            if do_benchmark:
                start = time.time()

            if probability_per_spg is not None:
                spg = np.random.choice(
                    list(probability_per_spg.keys()),
                    size=1,
                    p=list(probability_per_spg.values()),
                )[0]

            current_spgs.append(spg)

            if group_object_per_spg is not None and spg in group_object_per_spg.keys():
                group_object = group_object_per_spg[spg]
            else:
                group_object = None

            generated_structures = generate_structures(
                spacegroup_number=spg,
                group_object=group_object,
                N=1,  # only generate one structure
                max_NO_atoms_asymmetric_unit=max_NO_atoms_asymmetric_unit,
                max_volume=max_volume,
                fixed_volume=fixed_volume,
                probability_per_spg_per_element=probability_per_spg_per_element,
                probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
                NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
                per_element=per_element,
                return_original_pyxtal_object=do_benchmark,
                do_symmetry_checks=do_symmetry_checks,
                denseness_factors_density_per_spg=denseness_factors_density_per_spg,
                denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                seed=seed,
                is_verbose=is_verbose,
            )

            if do_benchmark:
                NO_wyckoffs_log.append(len(generated_structures[0][1].atom_sites))
                generated_structures = [item[0] for item in generated_structures]

            structures.extend(generated_structures)

            if do_benchmark:
                timings_generation.append(time.time() - start)

        for i, structure in enumerate(structures):
            current_spg = current_spgs[i]

            try:

                patterns_ys = get_smeared_patterns(
                    structure=structure,
                    wavelength=wavelength,
                    xs=xs,
                    NO_corn_sizes=NO_corn_sizes,
                    two_theta_range=two_theta_range,
                    return_corn_sizes=return_structures_and_corn_sizes,
                    do_benchmark=do_benchmark,
                    timings_simulation_pattern=timings_simulation_pattern,
                    timings_simulation_smeared=timings_simulation_smeared,
                )

                if return_structures_and_corn_sizes:
                    patterns_ys, corn_sizes = patterns_ys

            except Exception as ex:

                print("Error simulating pattern:")
                print(ex)
                print("".join(traceback.format_exception(None, ex, ex.__traceback__)))

            else:

                labels.extend([current_spg] * NO_corn_sizes)
                result_patterns_y.extend(patterns_ys)

                if return_structures_and_corn_sizes:
                    all_corn_sizes.extend(corn_sizes)

                all_structures.extend(structures)

    if not return_structures_and_corn_sizes:
        return result_patterns_y, labels
    else:
        return result_patterns_y, labels, all_structures, all_corn_sizes


def plot_timings(
    timings_per_volume, NO_wyckoffs_per_volume, label, show_legend=True, make_fit=True
):
    """Plot the timings of a benchmark run.

    Args:
        timings_per_volume (dict of list): For each volume, this dictionary provides a list of timings.
        NO_wyckoffs_per_volume (dict of list): For each volume, this dictionary provides a list of number of atoms in the asymmetric unit for the generated structure.
        label (str): Label used for the name of the output file.
        show_legend (bool, optional): Whether or not to show the legend. Defaults to True.
        make_fit (bool, optional): Whether or not to make a linear fit for each volume. Defaults to True.
    """

    figure_double_width_pub = matplotlib_defaults.pub_width
    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95 * 0.5,
            figure_double_width_pub * 0.7 * 0.5,
        )
    )
    plt.xlabel("Number of atoms in asymmetric unit")
    plt.ylabel("Time / s")

    for current_volume in timings_per_volume.keys():

        if make_fit:
            model = LinearRegression()
            model.fit(
                np.expand_dims(NO_wyckoffs_per_volume[current_volume], -1),
                np.expand_dims(timings_per_volume[current_volume], -1),
            )

        color = next(plt.gca()._get_lines.prop_cycler)["color"]

        plt.scatter(
            NO_wyckoffs_per_volume[current_volume],
            timings_per_volume[current_volume],
            label=f"Volume {current_volume} " + r"$Å^3$",
            color=color,
            s=2,
        )

        if make_fit:
            xs = np.linspace(0, 100, 100)
            plt.plot(xs, model.predict(np.expand_dims(xs, -1))[:, 0], color=color)

    if show_legend:
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"benchmark_{label}.pdf", bbox_inches="tight")
    plt.show()


def remove_outliers(timings, NO_wyckoffs):
    """Remove entries from both lists when the item in timings is an outlier.

    Args:
        timings (list of float): List of timings
        NO_wyckoffs (list of int): List of numbers of atoms in the asymmetric unit

    Returns:
        tuple: (timings, NO_wyckoffs, number of removed entries)
    """

    timings = np.array(timings)
    NO_wyckoffs = np.array(NO_wyckoffs)

    before = timings.shape[0]
    distance_from_mean = abs(timings - np.mean(timings))
    selector = distance_from_mean < 4 * np.std(timings)

    # Only keep the timings and NO_wyckoffs, where the criterion is fulfilled
    timings = timings[selector]
    NO_wyckoffs = NO_wyckoffs[selector]

    return timings, NO_wyckoffs, before - len(timings)


def perform_benchmark(
    parameters_of_simulation,
    volumes_to_probe=[500, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
):
    """Perform a benchmark that tests the speed of the simulation, smearing, and generation of synthetic patterns
    as a function of the unit cell volume.

    Args:
        parameters_of_simulation (dict): Parameters to pass to `get_synthetic_smeared_patterns` in a key-value format
        (except for `spgs`, `N_structures_per_spg`, and fixed_volume)
        wavelength (float): Wavelength in angstroms
    """

    N_per_probe = 500  # For each volume, how many structures to generate?

    timings_simulation_pattern_per_volume = {}
    timings_simulation_smeared_per_volume = {}
    timings_simulation_both_per_volume = {}
    timings_generation_per_volume = {}
    NO_wyckoffs_pattern_per_volume = {}
    NO_wyckoffs_smeared_per_volume = {}
    NO_wyckoffs_both_per_volume = {}
    NO_wyckoffs_generation_per_volume = {}

    for current_volume in volumes_to_probe:

        timings_simulation_pattern = []
        timings_simulation_smeared = []
        timings_generation = []
        NO_wyckoffs_log = []

        patterns, labels = get_synthetic_smeared_patterns(
            spgs=[0] * N_per_probe,
            N_structures_per_spg=1,
            fixed_volume=current_volume,
            **parameters_of_simulation,
            do_benchmark=True,
            timings_simulation_pattern=timings_simulation_pattern,
            timings_simulation_smeared=timings_simulation_smeared,
            timings_generation=timings_generation,
            NO_wyckoffs_log=NO_wyckoffs_log,
        )

        timings_simulation_both = [
            timings_simulation_pattern[i] + timings_simulation_smeared[i]
            for i in range(len(timings_simulation_pattern))
        ]

        (
            timings_simulation_pattern,
            NO_wyckoffs_simulation_pattern,
            removed,
        ) = remove_outliers(timings_simulation_pattern, NO_wyckoffs_log)
        print(f"Volume {current_volume}: Rejected {removed} outliers for simulation")
        (
            timings_simulation_smeared,
            NO_wyckoffs_simulation_smeared,
            removed,
        ) = remove_outliers(timings_simulation_smeared, NO_wyckoffs_log)
        print(f"Volume {current_volume}: Rejected {removed} outliers for smearing")
        (
            timings_simulation_both,
            NO_wyckoffs_simulation_both,
            removed,
        ) = remove_outliers(timings_simulation_both, NO_wyckoffs_log)
        print(
            f"Volume {current_volume}: Rejected {removed} outliers for simulation + smearing"
        )
        timings_generation, NO_wyckoffs_generation, removed = remove_outliers(
            timings_generation, NO_wyckoffs_log
        )
        print(f"Volume {current_volume}: Rejected {removed} outliers for generation")

        timings_simulation_pattern_per_volume[
            current_volume
        ] = timings_simulation_pattern
        timings_simulation_smeared_per_volume[
            current_volume
        ] = timings_simulation_smeared
        timings_simulation_both_per_volume[current_volume] = timings_simulation_both
        timings_generation_per_volume[current_volume] = timings_generation

        NO_wyckoffs_pattern_per_volume[current_volume] = NO_wyckoffs_simulation_pattern
        NO_wyckoffs_smeared_per_volume[current_volume] = NO_wyckoffs_simulation_smeared
        NO_wyckoffs_both_per_volume[current_volume] = NO_wyckoffs_simulation_both
        NO_wyckoffs_generation_per_volume[current_volume] = NO_wyckoffs_generation

    plot_timings(
        timings_simulation_pattern_per_volume,
        NO_wyckoffs_pattern_per_volume,
        "simulation",
    )
    plot_timings(
        timings_simulation_smeared_per_volume,
        NO_wyckoffs_smeared_per_volume,
        "smearing",
    )
    plot_timings(
        timings_simulation_both_per_volume,
        NO_wyckoffs_both_per_volume,
        "both",
    )

    # For the generation timings, the volume really doesn't matter at all
    # So, concatenate all results:
    flattened_generation_timings = []
    flattened_generation_NO_wyckoffs = []
    for key in timings_generation_per_volume.keys():
        flattened_generation_timings.extend(timings_generation_per_volume[key])
        flattened_generation_NO_wyckoffs.extend(NO_wyckoffs_generation_per_volume[key])
    plot_timings(
        {1000: flattened_generation_timings},
        {1000: flattened_generation_NO_wyckoffs},
        "generation",
        show_legend=False,
        make_fit=False,
    )
