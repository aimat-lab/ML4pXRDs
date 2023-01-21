# This script allows plotting the unit cell volume and number of atoms in the asymmetric unit over the spg label

from ml4pxrd_tools.manage_dataset import load_dataset_info
from ml4pxrd_tools.manage_dataset import get_wyckoff_info
from pyxtal import pyxtal
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    process_only_N = 5000

    (
        (
            probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff,
            NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element,
            denseness_factors_density_per_spg,
            denseness_factors_conditional_sampler_seeds_per_spg,
            lattice_paras_density_per_lattice_type,
            per_element,
            represented_spgs,
            probability_per_spg,
        ),
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

    volumes_spg = []
    NO_wyckoffs_spg = []

    i = 0
    for crystal, label in zip(
        statistics_crystals[:process_only_N], statistics_labels[:process_only_N]
    ):

        print(
            f"{(i/(process_only_N if process_only_N is not None else len(statistics_labels)))*100}%"
        )

        volumes_spg.append([crystal.volume, label[0]])

        try:

            struc = pyxtal()
            struc.from_seed(crystal)
            spg_number = (
                struc.group.number
            )  # use the group as calculated by pyxtal for statistics

            NO_wyckoffs, elements = get_wyckoff_info(struc)

            NO_wyckoffs_spg.append([NO_wyckoffs, label[0]])

        except Exception as ex:
            print(ex)

        i += 1

    volumes_spg = np.array(volumes_spg)
    NO_wyckoffs_spg = np.array(NO_wyckoffs_spg)

    plt.figure()
    plt.scatter(volumes_spg[:, 1], volumes_spg[:, 0], label="Volumes")
    plt.legend()
    plt.savefig("volumes_spg.pdf")

    plt.figure()
    plt.scatter(
        NO_wyckoffs_spg[:, 1],
        NO_wyckoffs_spg[:, 0],
        label="Number of atoms in asymmetric unit",
    )
    plt.legend()
    plt.savefig("NO_wyckoffs_spg.pdf")
