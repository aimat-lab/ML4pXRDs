from dataset_simulations.random_simulation_utils import generate_structures

from magpie_python.vassal.data.Cell import Cell
from magpie_python.vassal.data.Atom import Atom
from magpie_python import CrystalStructureEntry
from magpie_python.data.materials.util.LookUpData import LookUpData

from magpie_python import EffectiveCoordinationNumberAttributeGenerator
from magpie_python import CoordinationNumberAttributeGenerator
from magpie_python import StoichiometricAttributeGenerator
from magpie_python import PackingEfficiencyAttributeGenerator
from magpie_python import StructuralHeterogeneityAttributeGenerator
from magpie_python import PRDFAttributeGenerator

from pyxtal.database.element import Element
import re


def get_magpie_features(crystal):

    cell = Cell()
    cell.set_basis(crystal.lattice.matrix)

    # From the vasp io interface:
    # for t in range(len(types)):
    #    for ti in range(type_count[t]):
    #        # Read position.
    #        x = [float(w) for w in lines[atom_start].split()]
    #        atom_start += 1
    #        if cartesian:
    #            x = structure.convert_cartesian_to_fractional(x)
    #        atom = Atom(x, t)
    #        structure.add_atom(atom)
    #    structure.set_type_name(t, types[t])

    atoms_per_element = {}

    for atom in crystal:
        specie = re.sub(r"\d*[,.]?\d*\+?$", "", atom.species_string)
        specie = re.sub(r"\d*[,.]?\d*\-?$", "", specie)

        lookup_dict = LookUpData.element_ids
        if specie not in lookup_dict.keys():
            print(f"Species {specie} not in lookup_dict.")
            return None

        if specie in atoms_per_element.keys():
            atoms_per_element[specie].append(atom)
        else:
            atoms_per_element[specie] = [atom]

    for i, key in enumerate(atoms_per_element.keys()):

        for atom in atoms_per_element[key]:

            a = Atom(atom.frac_coords, i)
            a.set_radius(Element(key).covalent_radius)

            cell.add_atom(a)

        cell.set_type_name(i, key)

    entry = CrystalStructureEntry(cell, "", None)

    generator = EffectiveCoordinationNumberAttributeGenerator()  # weighted by face size
    result = generator.generate_features([entry])
    # print(result)
    mean_effective_coord_number = result["mean_Coordination"][0]

    generator = CoordinationNumberAttributeGenerator()
    result = generator.generate_features([entry])
    mean_coord_number = result["mean_Coordination"][0]

    generator = StoichiometricAttributeGenerator()
    result = generator.generate_features([entry])
    L2_norm = result["Comp_L2Norm"][0]
    L3_norm = result["Comp_L3Norm"][0]

    generator = PackingEfficiencyAttributeGenerator()
    result = generator.generate_features([entry])
    max_packing_efficiency = result["MaxPackingEfficiency"][0]

    generator = StructuralHeterogeneityAttributeGenerator()
    result = generator.generate_features([entry])
    mean_bond_length_variation = result["mean_BondLengthVariation"][0]

    # generator = PRDFAttributeGenerator()
    # generator.set_elements([entry])
    # result = generator.generate_features([entry])
    # print(result)

    return (
        mean_effective_coord_number,
        mean_coord_number,
        L2_norm,
        L3_norm,
        max_packing_efficiency,
        mean_bond_length_variation,
    )


if __name__ == "__main__":

    test_0 = generate_structures(15, 1)

    get_magpie_features(test_0[0])
