from matminer.featurizers.structure.order import ChemicalOrdering
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.misc import StructureComposition


def get_entropy(structure):

    # For structures where
    # each site is surrounded only by atoms of another type, this formula
    # yields large values of :math:`alpha`.
    chemical_order = ChemicalOrdering()

    # Shannon entropy, but not on coordinates but on occupation of symmetry sites
    complexity = StructuralComplexity()
