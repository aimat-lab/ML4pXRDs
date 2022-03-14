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


def get_magpie_features(crystals):

    for crystal in crystals:

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

            lookup_dict = LookUpData.element_ids
            if atom.species_string not in lookup_dict.keys():
                raise Exception("Oh")

            if atom.species_string in atoms_per_element.keys():
                atoms_per_element[atom.species_string].append(atom)
            else:
                atoms_per_element[atom.species_string] = [atom]

        for i, key in enumerate(atoms_per_element.keys()):

            for atom in atoms_per_element[key]:

                a = Atom(atom.frac_coords, i)
                a.set_radius(Element(key).covalent_radius)

                cell.add_atom(a)

            cell.set_type_name(i, key)

        entry = CrystalStructureEntry(cell, "", None)

        generator = EffectiveCoordinationNumberAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        generator = CoordinationNumberAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        generator = StoichiometricAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        generator = PackingEfficiencyAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        generator = StructuralHeterogeneityAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        generator = PRDFAttributeGenerator()
        result = generator.generate_features([entry])
        print(result)

        print()


if __name__ == "__main__":

    test_0 = generate_structures(15, 10)
    test_1 = generate_structures(15, 10)

    get_magpie_features([test_0, test_1], ["test_0", "test_1"])
