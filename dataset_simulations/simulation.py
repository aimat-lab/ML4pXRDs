import numpy as np
import time
import math
import os
import pandas as pd
from glob import glob
import pickle
import random
from datetime import datetime
from subprocess import Popen
from math import ceil
import subprocess
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import xrayutilities as xu
from pymatgen.io.cif import CifParser
import re

num_files = 127
num_processes = 127
# num_files = 8
# num_processes = 8

simulation_software = (
    "pymatgen_numba"  # possible: pymatgen, xrayutilities and pymatgen_numba
)

# as Vecsei:
angle_min = 5
angle_max = 90
angle_n = 8501

# as Park:
# angle_min = 10
# angle_max = 110
# angle_n = 10001


class Simulation:
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        random.seed(1234)

        self.reset_simulation_status()

        self.icsd_info_file_path = icsd_info_file_path
        self.icsd_cifs_dir = icsd_cifs_dir
        self.read_icsd()

        self.output_dir = "patterns/default/"  # should be overwritten by child class

        print("Protocol started: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    def reset_simulation_status(self):
        self.sim_crystals = []
        self.sim_labels = []  # space group, etc.
        self.sim_metas = []  # meta data: icsd id, ... of the crystal
        self.sim_patterns = (
            []
        )  # for each crystal, the simulated patterns; this can be multiple, depending on sim_variations
        self.sim_variations = []  # things like crystallite size, etc.
        self.sim_angles = []
        self.sim_intensities = []

        self.crystal_files = []
        self.crystal_files_Ns = []

    def simulate_all(self, test_crystallite_sizes=False, start_from_scratch=False):

        batch_size = ceil(len(self.sim_crystals) / num_files)

        os.system(f"mkdir -p {self.output_dir}")

        for output_file in self.crystal_files:
            os.system(f"rm {output_file}")
        self.save()

        print()
        print(
            f"Simulating {len(self.sim_crystals)} structures with {num_processes} workers and batch size {batch_size}."
        )

        N_files_to_process = len(self.crystal_files)
        N_files_per_process = ceil(N_files_to_process / num_processes)
        handles = []
        status_files = []
        log_files = []
        crystals_per_process = []

        print(flush=True)

        for i in range(0, num_processes):

            if (i + 1) * N_files_per_process < N_files_to_process:
                files_of_process = self.crystal_files[
                    i * N_files_per_process : (i + 1) * N_files_per_process
                ]
                crystals_per_process.append(
                    np.sum(
                        self.crystal_files_Ns[
                            i * N_files_per_process : (i + 1) * N_files_per_process
                        ]
                    )
                )
            else:
                files_of_process = self.crystal_files[i * N_files_per_process :]
                crystals_per_process.append(
                    np.sum(self.crystal_files_Ns[i * N_files_per_process :])
                )

            status_file_of_process = os.path.join(
                self.output_dir, "process_" + str(i) + ".status"
            )
            status_files.append(status_file_of_process)

            log_file = os.path.join(self.output_dir, "process_" + str(i) + ".log")
            log_files.append(log_file)

            p = Popen(
                [
                    "python",
                    "simulation_worker.py",
                    status_file_of_process,
                    "True" if start_from_scratch else "False",
                    "True" if test_crystallite_sizes else "False",
                    simulation_software,
                    *files_of_process,
                ],
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
            )
            handles.append(p)

        while True:

            time.sleep(30)

            polls = [p.poll() for p in handles]

            if not None in polls:  # all are done

                if os.path.exists(os.path.join(self.output_dir, "STOP")):
                    print("Simulation stopped by user.")

                if all(poll == 0 for poll in polls):
                    print("Simulation ended successfully.")
                else:
                    print("Simulation ended with problems. Check log files.")

                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                break

            if not all(poll == None or poll == 0 for poll in polls):

                if os.path.exists(os.path.join(self.output_dir, "STOP")):
                    print("Simulation stopped by user.")

                print("One or more of the workers terminated.")
                print("Polls of workers:")
                print(polls)

                break

            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            total = 0
            for i, status_file in enumerate(status_files):
                try:
                    with open(status_file, "r") as file:
                        content = file.readline()
                        N = int(content)

                except:
                    N = 0
                    print(f"Was not able to access status file of worker {i}")

                total += N
                print(f"Worker {i}: {N} of {crystals_per_process[i]}")
            print(f"Total: {total} of {len(self.sim_crystals)}", flush=True)

            print(flush=True)

    def read_icsd(self):

        pickle_file = os.path.join(self.icsd_cifs_dir, "icsd_meta")
        if not os.path.exists(pickle_file):

            print("Rebuilding icsd meta information...", flush=True)

            icsd_info = pd.read_csv(self.icsd_info_file_path, sep=",", skiprows=1)

            self.icsd_ids = list(icsd_info["CollectionCode"])
            self.icsd_space_group_symbols = list(icsd_info["HMS"])
            self.icsd_formulas = list(icsd_info["StructuredFormula"])
            self.icsd_sumformulas = list(icsd_info["SumFormula"])
            self.icsd_structure_types = list(icsd_info["StructureType"])
            self.icsd_standardised_cell_parameters = list(
                icsd_info["StandardisedCellParameter"]
            )
            self.icsd_r_values = list(icsd_info["RValue"])
            self.__match_icsd_cif_paths()

            # Shuffle the entries, so that the simulation is more balanced accross the workers:
            (
                self.icsd_ids,
                self.icsd_space_group_symbols,
                self.icsd_formulas,
                self.icsd_sumformulas,
                self.icsd_structure_types,
                self.icsd_standardised_cell_parameters,
                self.icsd_r_values,
                self.icsd_paths,
            ) = shuffle(
                self.icsd_ids,
                self.icsd_space_group_symbols,
                self.icsd_formulas,
                self.icsd_sumformulas,
                self.icsd_structure_types,
                self.icsd_standardised_cell_parameters,
                self.icsd_r_values,
                self.icsd_paths,
            )

            to_pickle = (
                self.icsd_ids,
                self.icsd_space_group_symbols,
                self.icsd_formulas,
                self.icsd_sumformulas,
                self.icsd_structure_types,
                self.icsd_standardised_cell_parameters,
                self.icsd_r_values,
                self.icsd_paths,
            )

            with open(pickle_file, "wb") as file:
                pickle.dump(to_pickle, file)

        else:

            with open(pickle_file, "rb") as file:
                (
                    self.icsd_ids,
                    self.icsd_space_group_symbols,
                    self.icsd_formulas,
                    self.icsd_sumformulas,
                    self.icsd_structure_types,
                    self.icsd_standardised_cell_parameters,
                    self.icsd_r_values,
                    self.icsd_paths,
                ) = pickle.load(file)

    def __match_icsd_cif_paths(self):

        # from dir
        all_paths = glob(os.path.join(self.icsd_cifs_dir, "*.cif"))

        self.icsd_paths = [None] * len(self.icsd_ids)

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0

        for i, path in enumerate(all_paths):

            print(f"{i} of {len(self.icsd_paths)}", flush=True)

            with open(path, "r") as f:
                first_line = f.readline()

                if first_line.strip() == "":
                    print(f"Skipped cif file {path}")
                    counter_2 += 1
                    continue

                id = int(first_line.strip().replace("data_", "").replace("-ICSD", ""))

                if id in self.icsd_ids:
                    self.icsd_paths[self.icsd_ids.index(id)] = path
                else:
                    counter_0 += 1

        counter_1 = self.icsd_paths.count(None)

        print(f"{counter_0} entries where in the cif directory, but not in the csv")
        print(f"{counter_1} entries where in the csv, but not in the cif directory")
        print(f"{counter_2} cif files skipped")
        print()

    def save(self, i=None):  # i specifies which batch to save

        data_dir = os.path.join(self.output_dir, "data")
        os.system("mkdir -p " + data_dir)

        batch_size = ceil(len(self.sim_crystals) / num_files)

        if not os.path.exists(self.output_dir):
            os.system("mkdir -p " + self.output_dir)

        if i is not None:
            if ((i + 1) * batch_size) < len(self.sim_crystals):
                start = i * batch_size
                end = (i + 1) * batch_size
            else:
                start = i * batch_size
                end = len(self.sim_crystals)
        else:  # save all

            self.crystal_files_Ns = []
            self.crystal_files = []

            for j in range(0, math.ceil(len(self.sim_crystals) / batch_size)):
                self.crystal_files.append(os.path.join(data_dir, f"crystals_{j}.npy"))
                self.save(i=j)

            return

        self.crystal_files_Ns.append(end - start)

        sim_crystals = np.empty(
            shape=(
                len(
                    self.sim_crystals[start:end],
                )
            ),
            dtype=object,
        )

        # force crystals to be written as python objects
        for j, crystal in enumerate(self.sim_crystals[start:end]):
            sim_crystals[j] = crystal
        np.save(os.path.join(data_dir, f"crystals_{i}.npy"), sim_crystals)

        np.save(
            os.path.join(data_dir, f"labels_{i}.npy"),
            np.array(self.sim_labels[start:end]),
        )
        np.save(
            os.path.join(data_dir, f"metas_{i}.npy"),
            np.array(self.sim_metas[start:end]),
        )

        np.save(
            os.path.join(data_dir, f"patterns_{i}.npy"),
            np.array(self.sim_patterns[start:end], dtype=object),
        )
        np.save(
            os.path.join(data_dir, f"variations_{i}.npy"),
            np.array(self.sim_variations[start:end], dtype=object),
        )

        with open(os.path.join(data_dir, f"angles_{i}.npy"), "wb") as pickle_file:
            pickle.dump(self.sim_angles[start:end], pickle_file)

        with open(os.path.join(data_dir, f"intensities_{i}.npy"), "wb") as pickle_file:
            pickle.dump(self.sim_intensities[start:end], pickle_file)

    def load(self, start=None, stop=None, load_patterns_angles_intensities=True):

        self.reset_simulation_status()

        data_dir = os.path.join(self.output_dir, "data")

        crystals_files = glob(os.path.join(data_dir, "crystals_*.npy"))
        crystals_files = sorted(
            crystals_files,
            key=lambda x: int(
                os.path.basename(x).replace("crystals_", "").replace(".npy", "")
            ),
        )

        labels_files = glob(os.path.join(data_dir, "labels_*.npy"))
        labels_files = sorted(
            labels_files,
            key=lambda x: int(
                os.path.basename(x).replace("labels_", "").replace(".npy", "")
            ),
        )

        metas_files = glob(os.path.join(data_dir, "metas_*.npy"))
        metas_files = sorted(
            metas_files,
            key=lambda x: int(
                os.path.basename(x).replace("metas_", "").replace(".npy", "")
            ),
        )

        patterns_files = glob(os.path.join(data_dir, "patterns_*.npy"))
        patterns_files = sorted(
            patterns_files,
            key=lambda x: int(
                os.path.basename(x).replace("patterns_", "").replace(".npy", "")
            ),
        )

        variations_files = glob(os.path.join(data_dir, "variations_*.npy"))
        variations_files = sorted(
            variations_files,
            key=lambda x: int(
                os.path.basename(x).replace("variations_", "").replace(".npy", "")
            ),
        )

        angles_files = glob(os.path.join(data_dir, "angles_*.npy"))
        angles_files = sorted(
            angles_files,
            key=lambda x: int(
                os.path.basename(x).replace("angles_", "").replace(".npy", "")
            ),
        )

        intensities_files = glob(os.path.join(data_dir, "intensities_*.npy"))
        intensities_files = sorted(
            intensities_files,
            key=lambda x: int(
                os.path.basename(x).replace("intensities_", "").replace(".npy", "")
            ),
        )

        last_index = stop if stop is not None else len(crystals_files)
        first_index = start if start is not None else 0

        for file in crystals_files[first_index:last_index]:
            self.sim_crystals.extend(np.load(file, allow_pickle=True))

        for file in labels_files[first_index:last_index]:
            self.sim_labels.extend(np.load(file, allow_pickle=True))

        for file in metas_files[first_index:last_index]:
            self.sim_metas.extend(np.load(file, allow_pickle=True))

        for file in variations_files[first_index:last_index]:
            self.sim_variations.extend(np.load(file, allow_pickle=True))

        if load_patterns_angles_intensities:

            for file in patterns_files[first_index:last_index]:
                self.sim_patterns.extend(np.load(file, allow_pickle=True))

            for file in angles_files[first_index:last_index]:
                with open(file, "rb") as pickle_file:
                    self.sim_angles.extend(pickle.load(pickle_file))

            for file in intensities_files[first_index:last_index]:
                with open(file, "rb") as pickle_file:
                    self.sim_intensities.extend(pickle.load(pickle_file))

    def get_space_group_number(self, id):

        # TODO: Check if symmetry information is correct using https://spglib.github.io/spglib/python-spglib.html
        # TODO: Pymatgen seems to be doing this already!
        # TODO: Bring them both together!

        # Important: We don't really care about the bravais lattice type, here!

        """
        try:

            parser = CifParser(cif_path)
            structures = parser.get_structures()

        except Exception as error:

            return None

        if len(structures) == 0:

            return None

        else:

            structure = structures[0]

        # determine space group:

        try:

            analyzer = SpacegroupAnalyzer(structure)

            group_number = analyzer.get_space_group_number()
            crystal_system = analyzer.get_crystal_system()
            space_group_symbol = analyzer.get_space_group_symbol()[0]

        except Exception as error:

            return None

        crystal_system_letter = ""

        if crystal_system == "anortic" or crystal_system == "triclinic":
            crystal_system_letter = "a"
        elif crystal_system == "monoclinic":
            crystal_system_letter = "m"
        elif crystal_system == "orthorhombic":
            crystal_system_letter = "o"
        elif crystal_system == "tetragonal":
            crystal_system_letter = "t"
        elif crystal_system == "cubic":
            crystal_system_letter = "c"
        elif crystal_system == "hexagonal" or crystal_system == "trigonal":
            crystal_system_letter = "h"
        else:
            return None

        if space_group_symbol in "ABC":
            space_group_symbol = "S"  # side centered

        bravais = crystal_system_letter + space_group_symbol

        if bravais not in [
            "aP",
            "mP",
            "mS",
            "oP",
            "oS",
            "oI",
            "oF",
            "tP",
            "tI",
            "cP",
            "cI",
            "cF",
            "hP",
            "hR",
        ]:
            print(
                "Bravais lattice {} not recognized. Skipping structure.".format(bravais)
            )
            return None

        # alternative way of determining bravais lattice from space group numer:
        # (for safety)

        return [bravais, group_number]
        """

        """
        space_group_symbol = self.icsd_space_group_symbols[self.icsd_ids.index(id)]

        sg = gemmi.find_spacegroup_by_name(space_group_symbol)

        space_group_number = sg.number

        return space_group_number

        """

        cif_path = self.icsd_paths[self.icsd_ids.index(id)]

        if cif_path is None:
            return None

        # read space group number directly from the cif file
        with open(cif_path, "r") as file:
            for line in file:
                if "_space_group_IT_number" in line:
                    space_group_number = int(
                        line.replace("_space_group_IT_number ", "").strip()
                    )
                    # print(space_group_number)
                    return space_group_number

        return None

    def __get_wyckoff_info(cif_path):
        # return: is_pure_occupancy, number_of_placements, wyckoff_str

        if cif_path is None:
            return None

        # read number of wyckoff placements directly from the cif file
        with open(cif_path, "r") as file:
            counter = 0
            counting = False
            is_pure = True

            wyckoff_str = ""
            elements = []
            occupancies = []
            wyckoffs_per_element = {}

            for line in file:
                """Example entry
                _atom_site_occupancy
                Fe1 Fe0+ 4 f 0.33333 0.66667 0.17175(3) . 1.
                O1 O2- 6 h 0.3310(7) 0.0895(5) 0.25 . 1.
                O2 O2- 12 i 0.3441(5) 0.2817(5) 0.0747(1) . 1.
                C1 C2+ 6 h 0.3320(8) 0.9098(7) 0.25 . 1.
                C2 C2+ 12 i 0.3380(6) 0.4243(5) 0.1133(2) . 1.
                loop_
                _atom_site_aniso_label
                """

                if counting:
                    columns = line.strip().split()
                    NO_columns = len(columns)

                    if NO_columns == 9:
                        occ = float(columns[-1])

                        element = columns[1]
                        # element = re.sub(r"\d+$", "", element)
                        element = re.sub(r"\d*\+?$", "", element)
                        element = re.sub(r"\d*\-?$", "", element)

                        wyckoff_name = columns[2] + columns[3]
                        if element not in wyckoffs_per_element.keys():
                            wyckoffs_per_element[element] = [wyckoff_name]
                        else:
                            wyckoffs_per_element[element].append(wyckoff_name)

                        elements.append(element)
                        occupancies.append(occ)

                        # if abs((occ - 1.0)) > 0.02:
                        #    is_pure = False
                        if occ != 1.0:
                            is_pure = False

                        counter += 1
                        wyckoff_str += line.strip() + "\n"

                    else:
                        wyckoff_repetitions = []

                        for key in wyckoffs_per_element.keys():
                            wyckoffs_unique = np.unique(wyckoffs_per_element[key])

                            for item in wyckoffs_unique:
                                wyckoff_repetitions.append(
                                    np.sum(np.array(wyckoffs_per_element[key]) == item)
                                )

                        return (
                            is_pure,
                            counter,
                            elements,
                            occupancies,
                            wyckoff_repetitions,
                        )

                if "_atom_site_occupancy" in line or "_atom_site_occupance" in line:
                    counting = True

    def get_wyckoff_info(self, id):
        # return: is_pure_occupancy, number_of_placements, wyckoff_str
        cif_path = self.icsd_paths[self.icsd_ids.index(id)]
        return Simulation.__get_wyckoff_info(cif_path)

    def plot(self, together=1):

        xs = np.linspace(angle_min, angle_max, angle_n)

        patterns = self.sim_patterns

        counter = 0
        for i, pattern in enumerate(patterns):
            for j, variation in enumerate(pattern):
                if variation[0] is not np.nan:

                    # plt.plot(xs, variation, label=repr(self.sim_variations[i][j]))
                    plt.plot(
                        xs,
                        variation,
                        label=repr(self.sim_variations[i][j]),
                        rasterized=False,
                    )

                    lines = np.array(self.sim_angles[i])

                    plt.vlines(
                        lines,
                        1.05,
                        1.15,
                        lw=0.15,
                    )

                    # plt.xlim((0, 90))

                    counter += 1

                    if (counter % together) == 0:
                        plt.legend()
                        plt.show()

    def add_path_to_be_simulated(self, path_to_crystal, labels, metas):

        try:

            if simulation_software == "xrayutilities":
                crystal = xu.materials.Crystal.fromCIF(path_to_crystal)
            else:
                parser = CifParser(path_to_crystal)
                crystals = parser.get_structures()
                crystal = crystals[0]

        except Exception as error:

            print(
                "Error encountered adding cif with id {}, skipping structure:".format(
                    metas[0]
                )
            )
            print(error)
            return None

        self.sim_crystals.append(crystal)
        self.sim_labels.append(labels)
        self.sim_metas.append(metas)

        self.sim_patterns.append(None)
        self.sim_variations.append(None)
        self.sim_angles.append(None)
        self.sim_intensities.append(None)

        return 1

    def add_crystal_to_be_simulated(self, crystal, labels, metas):

        self.sim_crystals.append(crystal)
        self.sim_labels.append(labels)
        self.sim_metas.append(metas)

        self.sim_patterns.append(None)
        self.sim_variations.append(None)
        self.sim_angles.append(None)
        self.sim_intensities.append(None)

    def plot_most_complicated_diffractogram(self):

        indices = []
        counts = []

        for i, pattern in enumerate(self.sim_patterns):
            if pattern is not None:
                for j, variation in enumerate(pattern):
                    if variation[0] is not np.nan:
                        counts.append(len(self.sim_angles[i]))
                        indices.append(i)

        max_index = indices[int(np.argmax(counts))]
        print(counts[int(np.argmax(counts))])

        plt.plot(np.linspace(0, 90, 9018), self.sim_patterns[max_index][0])
        plt.show()
