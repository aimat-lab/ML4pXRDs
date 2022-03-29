import os
import numpy as np
from dataset_simulations.simulation import Simulation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt
from dataset_simulations.random_simulation_utils import load_dataset_info
from dataset_simulations.core.quick_simulation import get_random_xy_patterns
import ray

if __name__ == "__main__":

    ray.init(num_cpus=8)

    spgs_to_analyze = [2, 15, 14, 129, 176]
    validation_max_NO_wyckoffs = 100
    validation_max_volume = 7000
    angle_range = np.linspace(5, 90, 8501)

    path_to_patterns = "../../dataset_simulations/patterns/icsd_vecsei/"
    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        icsd_sim = Simulation(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
        icsd_sim.output_dir = path_to_patterns
    else:  # local
        icsd_sim = Simulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
        icsd_sim.output_dir = path_to_patterns

    icsd_sim.load(start=0, stop=20)

    n_patterns_per_crystal = len(icsd_sim.sim_patterns[0])

    icsd_patterns_match = icsd_sim.sim_patterns
    icsd_labels_match = icsd_sim.sim_labels
    icsd_variations_match = icsd_sim.sim_variations
    icsd_crystals_match = icsd_sim.sim_crystals
    icsd_metas_match = icsd_sim.sim_metas

    # Mainly to make the volume constraints correct:
    conventional_errors_counter = 0
    print("Calculating conventional structures...")
    for i in reversed(range(0, len(icsd_crystals_match))):

        try:
            current_struc = icsd_crystals_match[i]
            analyzer = SpacegroupAnalyzer(current_struc)
            conv = analyzer.get_conventional_standard_structure()
            icsd_crystals_match[i] = conv

        except Exception as ex:

            print("Error calculating conventional cell of ICSD:")
            print(ex)
            conventional_errors_counter += 1

    print(
        f"{conventional_errors_counter} of {len(icsd_crystals_match)} failed to convert to conventional cell."
    )

    for i in reversed(range(0, len(icsd_patterns_match))):
        if (
            np.any(np.isnan(icsd_variations_match[i][0]))
            or icsd_labels_match[i][0] not in spgs_to_analyze
        ):
            del icsd_patterns_match[i]
            del icsd_labels_match[i]
            del icsd_variations_match[i]
            del icsd_crystals_match[i]
            del icsd_metas_match[i]

    NO_wyckoffs_cached = {}
    for i in reversed(range(0, len(icsd_patterns_match))):

        if validation_max_NO_wyckoffs is not None:
            is_pure, NO_wyckoffs, _, _, _, _, _, _ = icsd_sim.get_wyckoff_info(
                icsd_metas_match[i][0]
            )

            if icsd_metas_match[i][0] not in NO_wyckoffs_cached.keys():
                NO_wyckoffs_cached[icsd_metas_match[i][0]] = is_pure, NO_wyckoffs

        if (
            validation_max_volume is not None
            and icsd_crystals_match[i].volume > validation_max_volume
        ) or (
            validation_max_NO_wyckoffs is not None
            and NO_wyckoffs > validation_max_NO_wyckoffs
        ):
            del icsd_patterns_match[i]
            del icsd_labels_match[i]
            del icsd_variations_match[i]
            del icsd_crystals_match[i]
            del icsd_metas_match[i]

    ############################################################

    (
        probability_per_spg_per_element,
        probability_per_spg_per_element_per_wyckoff,
        NO_wyckoffs_prob_per_spg,
        corrected_labels,
        files_to_use_for_test_set,
        represented_spgs,
        NO_unique_elements_prob_per_spg,
        NO_repetitions_prob_per_spg_per_element,
        denseness_factors_density_per_spg,
        kde_per_spg,
        all_data_per_spg,
    ) = load_dataset_info()

    @ray.remote(num_cpus=1, num_gpus=0)
    def batch_generator_with_additional(
        spgs,
        structures_per_spg,
        N,
        start_angle,
        end_angle,
        max_NO_elements,
        NO_corn_sizes,
    ):

        patterns, labels, structures, corn_sizes = get_random_xy_patterns(
            spgs=spgs,
            structures_per_spg=structures_per_spg,
            wavelength=1.5406,  # Cu-Ka line
            # wavelength=1.207930,  # until ICSD has not been re-simulated with Cu-K line
            N=N,
            NO_corn_sizes=NO_corn_sizes,
            two_theta_range=(start_angle, end_angle),
            max_NO_elements=max_NO_elements,
            do_print=False,
            return_additional=True,
            do_distance_checks=True,
            do_merge_checks=False,
            use_icsd_statistics=True,
            probability_per_spg_per_element=probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff=probability_per_spg_per_element_per_wyckoff,
            max_volume=7000,
            NO_wyckoffs_prob_per_spg=NO_wyckoffs_prob_per_spg,
            do_symmetry_checks=True,
            force_wyckoff_indices=True,
            use_element_repetitions_instead_of_NO_wyckoffs=True,
            NO_unique_elements_prob_per_spg=NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element=NO_repetitions_prob_per_spg_per_element,
            denseness_factors_density_per_spg=denseness_factors_density_per_spg,
            kde_per_spg=kde_per_spg,
            all_data_per_spg=all_data_per_spg,
        )

        patterns = np.array(patterns)
        labels = np.array(labels)

        return patterns, labels, structures, corn_sizes

    random_comparison_crystals = []
    random_comparison_corn_sizes = []

    object_refs = []
    for i in range(1000):
        ref = batch_generator_with_additional.remote(
            spgs_to_analyze, 1, 8501, 5, 90, 100, 1
        )
        object_refs.append(ref)

    results = ray.get(object_refs)

    random_patterns = []
    random_labels = []

    for result in results:
        patterns, labels, crystals, corn_sizes = result

        random_comparison_crystals.extend(crystals)
        random_comparison_corn_sizes.extend(corn_sizes)

        random_patterns.extend(patterns)
        random_labels.extend(labels)

    ############################################################

    patterns_per_spg_icsd = {}
    for spg in spgs_to_analyze:
        patterns_per_spg_icsd[spg] = []

    patterns_per_spg_random = {}
    for spg in spgs_to_analyze:
        patterns_per_spg_random[spg] = []

    for i, patterns in enumerate(icsd_patterns_match):
        patterns_per_spg_icsd[icsd_labels_match[i][0]].append(patterns[0])

    for i, pattern in enumerate(random_patterns):
        patterns_per_spg_random[random_labels[i]].append(pattern)

    for spg in spgs_to_analyze:
        print(f"ICSD {spg}: {len(patterns_per_spg_icsd[spg])} patterns")
        print(f"Random {spg}: {len(patterns_per_spg_random[spg])} patterns")

        pattern_average_icsd = np.average(patterns_per_spg_icsd[spg], axis=0)
        pattern_average_random = np.average(patterns_per_spg_random[spg], axis=0)

        plt.plot(angle_range, pattern_average_icsd, label=f"ICSD, spg {spg}")
        plt.plot(angle_range, pattern_average_random, label=f"Random, spg {spg}")
        plt.legend()
        plt.show()
