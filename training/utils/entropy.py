from matminer.featurizers.structure.order import ChemicalOrdering
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.misc import StructureComposition

from spatialentropy import altieri_entropy
import numpy as np

from pymatgen.io.cif import CifParser
from pyxtal import pyxtal


def get_chemical_ordering(structure):

    # For structures where
    # each site is surrounded only by atoms of another type, this formula
    # yields large values of :math:`alpha`.
    chemical_order = ChemicalOrdering(shells=(1,))

    result = chemical_order.featurize(structure)

    return result[0]


def get_structural_complexity(structure):

    # Shannon entropy, but not on coordinates but on occupation of symmetry sites
    complexity = StructuralComplexity()

    data = complexity.featurize(structure)

    return data[0]


if __name__ == "__main__":

    """
    N = 500
    size_scaler = 200
    points_1 = size_scaler * np.random.randn(N, 2) + 1000
    points_2 = size_scaler * np.random.randn(N, 2) + 1000
    types = [0] * N + [1] * N
    print(altieri_entropy(np.concatenate((points_1, points_2), axis=0), types).entropy)
    """

    parser = CifParser("test_pure.cif")
    crystals = parser.get_structures()
    crystal = crystals[0]

    get_chemical_ordering(crystal)
    get_structural_complexity(crystal)
