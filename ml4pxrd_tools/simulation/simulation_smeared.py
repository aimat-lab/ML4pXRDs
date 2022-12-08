# Contains functions to generate patterns with smeared peaks.
# This code uses simulation_core.py to simulate the peak positions and intensities.
# We also provide functions to generate a synthetic crystal and simulate it in one go.

# TODO: Go through the whole code of this script file
# TODO: State that the implementation of the timings is a little bit hacky, but should be fine
# TODO: Pass-in all the required parameters of the timings function from outside
# TODO: Create a separate script in the root directory to run the benchmark: load the dataset and call the benchmark function
# TODO: Actually test if the benchmark runs through! Everything correctly implemented?
# TODO: Fix the occurring bugs, especially in manage_dataset.py

from ml4pxrd_tools.generation.structure_generation import generate_structures
from ml4pxrd_tools.simulation.simulation_core import get_pattern_optimized
import numpy as np
import time
import numba
import matplotlib.pyplot as plt
import traceback
from pyxtal.symmetry import Group
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


# The following lists are used when the function "run_benchmark" is called
# It is not the nicest way of implementing the benchmark, but it is easy and works well
timings_simulation_pattern = []
timings_simulation_smeared = []
timings_generation = []
NO_wyckoffs_log = []


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
):
    """Get a smeared (gaussian peaks) of the given structure.

    Args:
        structure (pymatgen.core.structure): Crystal structure
        wavelength (float): _description_
        xs (_type_): _description_
        NO_corn_sizes (int, optional): _description_. Defaults to 1.
        two_theta_range (tuple, optional): _description_. Defaults to (0, 90).
        return_corn_sizes (bool, optional): _description_. Defaults to False.
        return_angles_intensities (bool, optional): _description_. Defaults to False.
        return_max_unscaled_intensity_angle (bool, optional): _description_. Defaults to False.
        do_benchmark (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
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
    structures_per_spg,
    wavelength,
    N,
    NO_corn_sizes=1,
    two_theta_range=(0, 90),
    max_NO_elements=10,
    do_print=False,
    return_additional=False,
    do_distance_checks=True,
    fixed_volume=None,
    do_merge_checks=True,
    use_icsd_statistics=False,
    probability_per_spg_per_element=None,
    probability_per_spg_per_element_per_wyckoff=None,
    max_volume=None,
    NO_wyckoffs_prob_per_spg=None,
    do_symmetry_checks=True,
    set_NO_elements_to_max=False,
    force_wyckoff_indices=True,
    use_element_repetitions_instead_of_NO_wyckoffs=False,
    NO_unique_elements_prob_per_spg=None,
    NO_repetitions_prob_per_spg_per_element=None,
    denseness_factors_density_per_spg=None,
    kde_per_spg=None,
    all_data_per_spg=None,
    use_coordinates_directly=False,
    use_lattice_paras_directly=False,
    group_object_per_spg=None,
    denseness_factors_conditional_sampler_seeds_per_spg=None,
    lattice_paras_density_per_lattice_type=None,
    per_element=False,
    verbosity=2,
    probability_per_spg=None,
    add_background_and_noise=False,
    use_vecsei_bg_noise=False,
    caglioti_broadening=False,
):

    result_patterns_y = []
    labels = []
    all_corn_sizes = []
    all_structures = []

    xs = np.linspace(two_theta_range[0], two_theta_range[1], N)

    for spg in spgs:

        if do_print:
            start = time.time()

        structures = []
        current_spgs = []
        for _ in range(structures_per_spg):

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
                spg,
                1,
                max_NO_elements,
                do_distance_checks=do_distance_checks,
                fixed_volume=fixed_volume,
                do_merge_checks=do_merge_checks,
                use_icsd_statistics=use_icsd_statistics,
                probability_per_spg_per_element=probability_per_spg_per_element,
                probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
                max_volume=max_volume,
                NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
                do_symmetry_checks=do_symmetry_checks,
                set_NO_elements_to_max=set_NO_elements_to_max,
                force_wyckoff_indices=force_wyckoff_indices,
                use_element_repetitions_instead_of_NO_wyckoffs=use_element_repetitions_instead_of_NO_wyckoffs,
                NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
                denseness_factors_density_per_spg=denseness_factors_density_per_spg,
                kde_per_spg=kde_per_spg,
                all_data_per_spg=all_data_per_spg,
                use_coordinates_directly=use_coordinates_directly,
                use_lattice_paras_directly=use_lattice_paras_directly,
                group_object=group_object,
                denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
                per_element=per_element,
                verbosity=verbosity,
                return_original_pyxtal_object=do_print,
            )
            if do_print:
                NO_wyckoffs_log.append(len(generated_structures[0][1].atom_sites))
                generated_structures = [item[0] for item in generated_structures]

            structures.extend(generated_structures)

        if do_print:
            timings_generation.append(time.time() - start)

        for i, structure in enumerate(structures):

            current_spg = current_spgs[i]

            # print(structure.volume)

            try:

                patterns_ys = get_smeared_patterns(
                    structure,
                    wavelength,
                    xs,
                    NO_corn_sizes,
                    two_theta_range,
                    do_benchmark=do_print,
                    return_corn_sizes=return_additional,
                    add_background_and_noise=add_background_and_noise,
                    use_vecsei_bg_noise=use_vecsei_bg_noise,
                    caglioti_broadening=caglioti_broadening,
                )

                if return_additional:
                    patterns_ys, corn_sizes = patterns_ys

            except Exception as ex:

                print("Error simulating pattern:")
                print(ex)
                print("".join(traceback.format_exception(None, ex, ex.__traceback__)))

                if "list index" in str(ex):
                    print(structure)

            else:

                labels.extend([current_spg] * NO_corn_sizes)
                result_patterns_y.extend(patterns_ys)

                if return_additional:
                    all_corn_sizes.extend(corn_sizes)

                all_structures.extend(structures)

    if not return_additional:
        return result_patterns_y, labels
    else:
        return result_patterns_y, labels, all_structures, all_corn_sizes


def plot_timings(
    timings_per_volume, NO_wyckoffs_per_volume, label, show_legend=True, make_fit=True
):

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

    timings = np.array(timings)
    NO_wyckoffs = np.array(NO_wyckoffs)

    before = timings.shape[0]
    distance_from_mean = abs(timings - np.mean(timings))
    selector = distance_from_mean < 4 * np.std(timings)

    timings = timings[selector]
    NO_wyckoffs = NO_wyckoffs[selector]

    return timings, NO_wyckoffs, before - len(timings)


def time_swipe_with_fixed_volume():

    global timings_simulation_pattern
    global timings_simulation_smeared
    global timings_generation
    global NO_wyckoffs_log

    volumes_to_probe = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    N_per_probe = 500

    (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg,
        kde_per_spg,
        all_data_per_spg_tmp,
        denseness_factors_conditional_sampler_seeds_per_spg,
        lattice_paras_density_per_lattice_type,
        per_element,
        represented_spgs,
        (
            statistics_metas,
            statistics_labels,
            statistics_crystals,
            statistics_match_metas,
            statistics_match_labels,
            test_metas,
            test_labels,
            test_crystals,
            corrected_labels,
            test_match_metas,
            test_match_pure_metas,
        ),
    ) = load_dataset_info()

    group_object_per_spg = {}
    for spg in represented_spgs:
        group_object_per_spg[spg] = Group(spg, dim=3)

    probability_per_spg = {}
    for i, label in enumerate(statistics_match_labels):
        if label[0] in represented_spgs:
            if label[0] in probability_per_spg.keys():
                probability_per_spg[label[0]] += 1
            else:
                probability_per_spg[label[0]] = 1
    total = np.sum(list(probability_per_spg.values()))
    for key in probability_per_spg.keys():
        probability_per_spg[key] /= total

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
            structures_per_spg=1,
            wavelength=1.5406,  # Cu-K line
            # wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
            N=8501,
            NO_corn_sizes=1,
            two_theta_range=(5, 90),
            max_NO_elements=100,
            do_print=True,  # get timings
            do_distance_checks=False,
            do_merge_checks=False,
            use_icsd_statistics=True,
            probability_per_spg_per_element=probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
            max_volume=7001,
            NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
            do_symmetry_checks=True,
            force_wyckoff_indices=True,
            use_element_repetitions_instead_of_NO_wyckoffs=True,
            NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
            denseness_factors_density_per_spg=denseness_factors_density_per_spg,
            kde_per_spg=None,
            all_data_per_spg=None,
            use_coordinates_directly=False,
            use_lattice_paras_directly=False,
            group_object_per_spg=group_object_per_spg,
            denseness_factors_conditional_sampler_seeds_per_spg=denseness_factors_conditional_sampler_seeds_per_spg,
            lattice_paras_density_per_lattice_type=lattice_paras_density_per_lattice_type,
            per_element=per_element,
            verbosity=2,
            probability_per_spg=probability_per_spg,
            add_background_and_noise=False,
            use_vecsei_bg_noise=False,
            fixed_volume=current_volume,
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

    # For the generation, the volume really doesn't matter at all
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


if __name__ == "__main__":
    time_swipe_with_fixed_volume()
