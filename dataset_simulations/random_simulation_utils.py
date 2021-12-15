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
from pyxtal.operations import filtered_coords
import pickle

# import warnings
# with warnings.catch_warnings():
#    warnings.simplefilter("error")

N_workers = 8

max_NO_elements = 10
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
            ),
            flush=True,
        )
        time.sleep(update_interval)


def generate_structure(
    _, spacegroup_number, group_object, multiplicities, names, letters, dofs, index=None
):

    # print()
    # print(f"Index: {index}")

    # if i is not None:
    #    print(i)

    # TODO: maybe use slightly random volume factors later

    while True:
        number_of_atoms_per_site = np.zeros(len(names))

        NO_elements = random.randint(1, max_NO_elements)
        # NO_elements = 10  # TODO: Change this back

        # print("NO_atoms:")
        # print(NO_elements)

        chosen_elements = []
        chosen_numbers = []
        chosen_wyckoff_positions = []
        chosen_wyckoff_letters = []

        counter_collisions = 0

        for i in range(0, NO_elements):
            while True:

                if counter_collisions > 100:
                    print("More than 100 collisions.", flush=True)
                    break

                chosen_index = random.randint(0, len(number_of_atoms_per_site) - 1)
                """
                # always first choose the general Wyckoff site:
                chosen_index = (
                    random.randint(0, len(number_of_atoms) - 1) if i > 0 else 0
                )
                """

                # TODO: Think about this again! Is this OK?
                """ See this from the documentation
                PyXtal starts with the largest available WP, which is the general position of the space group.
                If the number of atoms required is equal to or greater than the size of the general position,
                the algorithm proceeds. If fewer atoms are needed, the next largest WP (or set of WP’s) is
                chosen, in order of descending multiplicity. This is done to ensure that larger positions are
                preferred over smaller ones; this reflects the greater prevalence of larger multiplicities
                both statistically and in nature.
                """

                if (
                    dofs[chosen_index] == 0
                    and int(number_of_atoms_per_site[chosen_index]) == 1
                ):
                    counter_collisions += 1
                    # print(f"{counter_collisions} collisions.", flush=True)
                    continue

                number_of_atoms_per_site[chosen_index] += 1

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

        # print(number_of_atoms_per_site)

        # try:
        my_crystal.from_random(
            dim=3,
            group=group_object,
            species=chosen_elements,
            numIons=chosen_numbers,
            # sites=chosen_wyckoff_positions,
        )
        # except Exception as ex:
        #    print(flush=True)
        #    print(ex, flush=True)
        #    print(spacegroup_number, flush=True)
        #    print(chosen_elements, flush=True)
        #    print(chosen_numbers, flush=True)
        #    print(flush=True)

        if not my_crystal.valid:
            continue

        try:

            # TODO: Remove this again, later!
            for site in my_crystal.atom_sites:
                site.coords = filtered_coords(site.coords)

            crystal = my_crystal.to_pymatgen(special=(index == 55))
        except Exception as ex:
            print(flush=True)
            print(ex, flush=True)
            print(spacegroup_number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)

        # print(spacegroup_number)
        # print(chosen_elements)
        # print(chosen_numbers)
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

    print(flush=True)
    print(f"Current group: {spacegroup_number}", flush=True)
    print(names, flush=True)
    print(multiplicities, flush=True)
    print(flush=True)

    # TODO: Change back
    """
    pool = multiprocessing.Pool(processes=N_workers)

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
    """

    result = [
        generate_structure(
            None,
            spacegroup_number=spacegroup_number,
            group_object=group,
            multiplicities=multiplicities,
            names=names,
            letters=letters,
            dofs=dofs,
            index=i,
        )
        for i in range(0, N)
    ]

    with open("compare_original", "wb") as file:
        coords = []
        for crystal in result:
            coords.append(crystal.cart_coords)
        pickle.dump(coords, file)

    print(f"Generated {len(result)} of {N} requested crystals", flush=True)

    return result


if __name__ == "__main__":

    if True:

        random.seed(123)
        np.random.seed(123)

        generate_structures(13, 100)

        start = time.time()

        # timings = []
        # for spg in range(1, 231):
        #    #    start_inner = time.time()
        #    generate_structures(spg, 1)
        #    timings.append(time.time() - start_inner)
        # plt.scatter(list(range(0, len(timings))), timings)
        # plt.show()

        generate_structures(13, 100)

        stop = time.time()

        print(f"Total job took {stop-start} s", flush=True)

    else:

        with open("compare_debug", "rb") as file:
            coords_debug = pickle.load(file)

        with open("compare_original", "rb") as file:
            coords_original = pickle.load(file)

        for i, coor in enumerate(coords_debug):
            for j, coordinate in enumerate(coor):
                compare_to = coords_original[i][j]

                if np.sum(np.square(coordinate - compare_to)) > 10 ** (-15):
                    print("Oh oh")

                    if j == 0:
                        print(coords_original[i])
                        print(coords_debug[i])
