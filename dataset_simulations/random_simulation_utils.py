from pyxtal import pyxtal
import matplotlib.pyplot as plt
import numpy as np
import random
from pyxtal.symmetry import Group
import time
import pickle
from dataset_simulations.simulation import Simulation
from pymatgen.io.cif import CifParser

# import warnings
# with warnings.catch_warnings():
#    warnings.simplefilter("error")

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
    _,
    group_object,
    multiplicities,
    names,
    letters,
    dofs,
    max_NO_elements=10,
    seed=-1,
    do_distance_checks=True,
    fixed_volume = None,
    do_merge_checks = True,
):
    while True:

        if seed != -1:
            np.random.seed(seed)
            random.seed(seed)

        number_of_atoms_per_site = np.zeros(len(names))

        NO_elements = random.randint(1, max_NO_elements)

        chosen_elements = []
        chosen_numbers = []
        chosen_wyckoff_positions = []
        chosen_wyckoff_letters = []

        for i in range(0, NO_elements):

            counter_collisions = 0
            while True:

                if counter_collisions > 30:
                    print("More than 30 collisions setting one atom.", flush=True)
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

        # TODO: Maybe bring unique entries of chosen_elements together to form one?
        # probably not needed / additional overhead

        my_crystal = pyxtal()

        try:

            my_crystal.from_random(
                dim=3,
                group=group_object,
                species=chosen_elements,
                numIons=chosen_numbers,
                # sites=chosen_wyckoff_positions,
                my_seed=seed,
                factor=np.random.uniform(0.7, 5.0),
                #factor=1.1,
                do_distance_checks=do_distance_checks,
                fixed_volume = fixed_volume,
                do_merge_checks=do_merge_checks
            )

        except Exception as ex:
            print(flush=True)
            print(ex, flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)
            continue

        if not my_crystal.valid:
            print(flush=True)
            print("Generated a non-valid crystal. Something went wrong.", flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)
            continue

        try:

            # Only for comparing the debug code with the original code:
            # for site in my_crystal.atom_sites:
            #    site.coords = filtered_coords(site.coords)

            crystal = my_crystal.to_pymatgen()

        except Exception as ex:
            print(flush=True)
            print(ex, flush=True)
            print(group_object.number, flush=True)
            print(chosen_elements, flush=True)
            print(chosen_numbers, flush=True)
            print(flush=True)
            continue

        # print(spacegroup_number)
        # print(chosen_elements)
        # print(chosen_numbers)
        # vis = StructureVis()
        # vis.set_structure(crystal)
        # vis.show()

        return crystal


def generate_structures(spacegroup_number, N, max_NO_elements=10, seed=-1, do_distance_checks=True, fixed_volume=None, do_merge_checks = True):

    group = Group(spacegroup_number, dim=3)

    multiplicities = [x.multiplicity for x in group]
    names = [(str(x.multiplicity) + x.letter) for x in group]
    dofs = group.get_site_dof(names)
    letters = [x.letter for x in group]

    # print(flush=True)
    # print(f"Current group: {spacegroup_number}", flush=True)
    # print(names, flush=True)
    # print(multiplicities, flush=True)
    # print(flush=True)

    result = [
        generate_structure(
            None,
            group_object=group,
            multiplicities=multiplicities,
            names=names,
            letters=letters,
            dofs=dofs,
            max_NO_elements=max_NO_elements,
            seed=seed,
            do_distance_checks=do_distance_checks,
            fixed_volume=fixed_volume,
            do_merge_checks=do_merge_checks
        )
        for i in range(0, N)
    ]

    return result

def analyse_set_wyckoffs():

    icsd_sim = Simulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    counts_per_spg_per_wyckoff = {}

    counter_per_element = {}

    # pre-process the symmetry groups:

    for spg_number in range(1, 231):
        group = Group(spg_number, dim=3)
        names = [(str(x.multiplicity) + x.letter) for x in group]

        counts_per_spg_per_wyckoff[spg_number] = {}
        for name in names:
            counts_per_spg_per_wyckoff[spg_number][name] = 0

    for i, path in enumerate(icsd_sim.icsd_paths):

        try: 

            #spg_number = icsd_sim.get_space_group_number(icsd_sim.icsd_ids[i])

            parser = CifParser(path)
            crystals = parser.get_structures()

            if len(crystals) == 0:
                continue

            crystal = crystals[0]

            struc = pyxtal()
            struc.from_seed(crystal)

        except Exception as ex:

            print(f"Error reading {path}:")
            print(ex)

            continue

        #if spg_number != struc.group.number:
        #
        #   print("ohoh")

        spg_number = struc.group.number

        for site in struc.atom_sites:
            specie_str = str(site.specie)
            if specie_str in counter_per_element:
                counter_per_element[specie_str] += 1
            else:
                counter_per_element[specie_str] = 0

            name = str(site.wp.multiplicity) + site.wp.letter
            counts_per_spg_per_wyckoff[spg_number][name] += 1 

    print()

if __name__ == "__main__":

    if True:
        analyse_set_wyckoffs()

    if False:

        NO_chosen_elements = 50

        seed = 5215
        number_per_spg = 1
        low = 1
        high = 231
        np.random.seed(seed)
        random.seed(seed)

        structure_seeds = np.random.randint(0, 10000, 230 * number_per_spg)

        # To pre-compile functions:
        for spg in range(low, high):
            generate_structures(spg, number_per_spg, seed=int(structure_seeds[spg - 1]), do_distance_checks=False)

        start = time.time()
        for spg in range(low, high):
            generate_structures(
                spg, number_per_spg, seed=int(structure_seeds[spg - 1]), do_distance_checks=False, max_NO_elements=NO_chosen_elements
            )
        stop = time.time()
        print(f"No distance checks: {stop-start} s", flush=True)

        if False:

            start = time.time()
            for spg in range(low, high):
                generate_structures(
                    spg, number_per_spg, seed=int(structure_seeds[spg - 1]), do_distance_checks=True, max_NO_elements=NO_chosen_elements
                )
            stop = time.time()
            print(f"With distance checks: {stop-start} s", flush=True)

    if False:

        seed = 5215
        number_per_spg = 1

        low = 1
        high = 13

        np.random.seed(seed)
        random.seed(seed)

        structure_seeds = np.random.randint(0, 10000, 230 * number_per_spg)

        # generate_structures(13, 100, seed=seed)

        # To pre-compile functions:
        for spg in range(low, high):
            generate_structures(spg, number_per_spg, seed=int(structure_seeds[spg - 1]))

        start = time.time()

        results = []
        timings = []
        for spg in range(low, high):
            start_inner = time.time()
            results.extend(
                generate_structures(
                    spg, number_per_spg, seed=int(structure_seeds[spg - 1])
                )
            )
            timings.append(time.time() - start_inner)

        with open("compare_debug", "wb") as file:
            coords = []
            for crystal in results:
                coords.append(crystal.cart_coords)
            pickle.dump(coords, file)

        stop = time.time()

        print(f"Total job took {stop-start} s", flush=True)

    if False:

        with open("compare_debug", "rb") as file:
            coords_debug = pickle.load(file)

        with open("compare_original", "rb") as file:
            coords_original = pickle.load(file)

        for i, coor in enumerate(coords_debug):
            for j, coordinate in enumerate(coor):
                compare_to = coords_original[i][j]

                if np.sum(np.square(coordinate - compare_to)) > 10 ** (-10):
                    print(f"Oh oh {i} {j}")

                    # if j == 0:
                    #    print(coords_original[i])
                    #    print(coords_debug[i])
