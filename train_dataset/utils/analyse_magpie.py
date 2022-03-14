from dataset_simulations.random_simulation_utils import generate_structures

from magpie_python.data.materials import CrystalStructureEntry
from magpie_python.vassal.data.Cell import Cell


def plot_magpie_features(crystal_lists, label_per_list):

    feature_1_lists = []

    for i, list in enumerate(crystal_lists):

        feature_1_lists.append([])

        for crystal in list:

            cell = Cell()

            # cell.set_basis()


if __name__ == "__main__":

    test_0 = generate_structures(15, 10)
    test_1 = generate_structures(15, 10)

    plot_magpie_features([test_0, test_1], ["test_0", "test_1"])
