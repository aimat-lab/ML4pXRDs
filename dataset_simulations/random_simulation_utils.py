from pyxtal import pyxtal
import matplotlib.pyplot as plt
import numpy as np
import random
from pyxtal.symmetry import Group
import time
from pymatgen.vis.structure_vtk import StructureVis
import multiprocessing
from functools import partial
from multiprocessing import set_start_method

max_NO_atoms = 5
# 10 atoms per unit cell should probably be already enough, at least in the beginning.
# probably even 5 is enough, at first

# extracted from pyxtal element.py:
all_elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
][0:94]


def track_job(job, update_interval=5):
    while job._number_left > 0:
        print(
            "Tasks remaining: {0} (chunk size {1})".format(
                job._number_left * job._chunksize, job._chunksize
            )
        )
        time.sleep(update_interval)


def generate_structure(_, spacegroup_number, multiplicities, names, letters, dofs):

    # TODO: maybe use slightly random volume factors later

    while True:

        number_of_atoms = np.zeros(len(names))

        NO_atoms = random.randint(1, max_NO_atoms)
        # NO_atoms = 10

        chosen_elements = []
        chosen_numbers = []
        chosen_wyckoff_positions = []
        chosen_wyckoff_letters = []

        counter_collisions = 0

        for i in range(0, NO_atoms):
            while True:

                if counter_collisions > 100:
                    print("More than 100 collisions.")
                    break

                chosen_index = random.randint(0, len(number_of_atoms) - 1)
                """
                # always first choose the general Wyckoff site:
                chosen_index = (
                    random.randint(0, len(number_of_atoms) - 1) if i > 0 else 0
                )
                """
                # TODO: Think about this again!
                """ See this from the documentation
                PyXtal starts with the largest available WP, which is the general position of the space group.
                If the number of atoms required is equal to or greater than the size of the general position,
                the algorithm proceeds. If fewer atoms are needed, the next largest WP (or set of WP’s) is
                chosen, in order of descending multiplicity. This is done to ensure that larger positions are
                preferred over smaller ones; this reflects the greater prevalence of larger multiplicities
                both statistically and in nature.
                """

                if dofs[chosen_index] == 0 and int(number_of_atoms[chosen_index]) == 1:
                    counter_collisions += 1
                    print(f"{counter_collisions} collisions.")
                    continue

                number_of_atoms[chosen_index] += 1

                chosen_elements.append(random.choice(all_elements))
                chosen_numbers.append(multiplicities[chosen_index])
                chosen_wyckoff_positions.append([names[chosen_index]])
                chosen_wyckoff_letters.append([letters[chosen_index]])

                break

        """ For spg C++ program
        output_str = ""
        second_output_str = ""
        for i, element in enumerate(chosen_elements):
            output_str += element + str(chosen_numbers[i])
            second_output_str += (
                f"forceWyckPos {element}        = {chosen_wyckoff_letters[i][0]}\n"
            )
        print(output_str)
        print(second_output_str)
        # forceWyckPos Mg        = a
        """

        # TODO: Maybe bring unique entries of chosen_elements together to form one?
        # probably not needed / additional overhead

        my_crystal = pyxtal()

        try:
            my_crystal.from_random(
                dim=3,
                group=spacegroup_number,
                species=chosen_elements,
                numIons=chosen_numbers,
                # sites=chosen_wyckoff_positions,
            )
        except:
            continue

        if not my_crystal.valid:
            continue
        
        try:
            crystal = my_crystal.to_pymatgen()
        except:
            continue

        # vis = StructureVis()
        # vis.set_structure(crystal)
        # vis.show()

        return crystal


def generate_structures(spacegroup_number, N):

    try:
        set_start_method("spawn")
    except:
        pass

    group = Group(spacegroup_number, dim=3)
    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)
    letters = [x.letter for x in group]

    """
    for i in range(0, N):
        print(i)
        generate_structure(
            None, spacegroup_number, multiplicities, names, letters, dofs
        )
    exit()
    """

    pool = multiprocessing.Pool(processes=80)

    handle = pool.map_async(
        partial(
            generate_structure,
            spacegroup_number=spacegroup_number,
            multiplicities=multiplicities,
            names=names,
            letters=letters,
            dofs=dofs,
        ),
        [None] * N,
    )

    track_job(handle)

    result = handle.get()

    print(f"Generated {len(result)} of {N} requested crystals")

    return result


if __name__ == "__main__":

    start = time.time()

    generate_structures(166, 80)

    stop = time.time()

    print(f"Total job took {stop-start} s")
