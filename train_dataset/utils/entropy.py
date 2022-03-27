from matminer.featurizers.structure.order import ChemicalOrdering
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.misc import StructureComposition

from spatialentropy import altieri_entropy
import numpy as np


def get_entropy(structure):

    # For structures where
    # each site is surrounded only by atoms of another type, this formula
    # yields large values of :math:`alpha`.
    chemical_order = ChemicalOrdering()

    # Shannon entropy, but not on coordinates but on occupation of symmetry sites
    complexity = StructuralComplexity()


if __name__ == "__main__":

    N = 500
    size_scaler = 200

    points_1 = size_scaler * np.random.randn(N, 2) + 1000
    points_2 = size_scaler * np.random.randn(N, 2) + 1000

    types = [0] * N + [1] * N

    print(altieri_entropy(np.concatenate((points_1, points_2), axis=0), types).entropy)
