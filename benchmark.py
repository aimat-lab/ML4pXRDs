"""This script can be used to run a benchmark of the generation and simulation of patterns
(see the benchmark figure in the paper).
"""

from training.manage_dataset import load_dataset_info
from ml4pxrd_tools.simulation.simulation_smeared import perform_benchmark
from pyxtal.symmetry import Group


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
    probability_per_spg,
) = load_dataset_info()

group_object_per_spg = {}
for spg in represented_spgs:
    group_object_per_spg[spg] = Group(spg, dim=3)

parameters = {
    "wavelength": 1.5406,
    "two_theta_range": (5, 90),
    "probability_per_spg_per_element": probability_per_spg_per_element,
    "probability_per_spg_per_element_per_wyckoff": probability_per_spg_per_element_per_wyckoff,
    "NO_unique_elements_prob_per_spg": NO_unique_elements_prob_per_spg,
    "NO_repetitions_prob_per_spg_per_element": NO_repetitions_prob_per_spg_per_element,
    "per_element": per_element,
    "do_symmetry_checks": True,
    "denseness_factors_conditional_sampler_seeds_per_spg": denseness_factors_conditional_sampler_seeds_per_spg,
    "lattice_paras_density_per_lattice_type": lattice_paras_density_per_lattice_type,
    "group_object_per_spg": group_object_per_spg,
    "probability_per_spg": probability_per_spg,
}

print("per_element", per_element)

perform_benchmark()
