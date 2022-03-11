from magpie_python import StoichiometricAttributeGenerator
from magpie_python.vassal.data import Cell


def plot_magpie_features(crystal_lists, label_per_list):

    feature_1_lists = []

    for i, list in enumerate(crystal_lists):

        feature_1_lists.append([])

        for crystal in list:

            cell = Cell()

            cell.set_basis()
