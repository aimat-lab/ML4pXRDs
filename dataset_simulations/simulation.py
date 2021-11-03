import numpy as np
import matplotlib.pyplot as pl
import time
import math
import os
import pandas as pd
from glob import glob
import pickle
import random
import lzma
from datetime import datetime
from subprocess import Popen
from math import ceil
import subprocess
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

num_files = 16
num_processes = 8

angle_min = 0
angle_max = 90
angle_n = 9001


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
        self.sim_lines_list = []
        self.output_files = []
        self.output_files_Ns = []

    def simulate_all(self, test_crystallite_sizes=False, start_from_scratch=False):

        batch_size = ceil(len(self.sim_crystals) / num_files)

        os.system(f"mkdir -p {self.output_dir}")

        for output_file in self.output_files:
            os.system(f"rm {output_file}")
        self.save()

        print()
        print(
            f"Simulating {len(self.sim_crystals)} structures with {num_processes} workers and batch size {batch_size}."
        )

        N_files_to_process = len(self.output_files)
        N_files_per_process = ceil(N_files_to_process / num_processes)
        handles = []
        status_files = []
        log_files = []
        crystals_per_process = []

        for i in range(0, num_processes):

            if (i + 1) * N_files_per_process < N_files_to_process:
                files_of_process = self.output_files[
                    i * N_files_per_process : (i + 1) * N_files_per_process
                ]
                crystals_per_process.append(
                    np.sum(
                        self.output_files_Ns[
                            i * N_files_per_process : (i + 1) * N_files_per_process
                        ]
                    )
                )
            else:
                files_of_process = self.output_files[i * N_files_per_process :]
                crystals_per_process.append(
                    np.sum(self.output_files_Ns[i * N_files_per_process :])
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
                    *files_of_process,
                ],
                stdout=open(log_file, "w"),
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
                break

            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            total = 0
            for i, status_file in enumerate(status_files):
                with open(status_file, "r") as file:
                    N = int(file.readline())
                total += N
                print(f"Worker {i}: {N} of {crystals_per_process[i]}")
            print(f"Total: {total} of {len(self.sim_crystals)}")

            print(flush=True)

    def read_icsd(self):

        pickle_file = os.path.join(self.icsd_cifs_dir, "icsd_meta")
        if not os.path.exists(pickle_file):

            print("Rebuilding icsd meta information...")

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

            if (i % 1000) == 0:
                print(f"{i} of {len(self.icsd_paths)}")

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

            self.output_files_Ns = []
            self.output_files = []

            for j in range(0, math.ceil(len(self.sim_crystals) / batch_size)):
                self.output_files.append(
                    os.path.join(self.output_dir, f"data_{j}.lzma")
                )
                self.save(i=j)

            return

        self.output_files_Ns.append(end - start)

        pickle_file = os.path.join(self.output_dir, f"data_{i}.lzma")

        with lzma.open(pickle_file, "wb") as file:
            pickle.dump(
                (
                    self.sim_crystals[start:end],
                    self.sim_labels[start:end],
                    self.sim_metas[start:end],
                    self.sim_patterns[start:end],
                    self.sim_variations[start:end],
                    self.sim_lines_list[start:end],
                ),
                file,
            )

    def load(self):

        self.reset_simulation_status()

        pickle_files = glob(os.path.join(self.output_dir, f"data_*.lzma"))
        pickle_files = sorted(
            pickle_files,
            key=lambda x: int(
                os.path.basename(x).replace("data_", "").replace(".lzma", "")
            ),
        )

        for pickle_file in pickle_files:
            with lzma.open(pickle_file, "rb") as file:
                additional = pickle.load(file)
                self.sim_crystals.extend(additional[0])
                self.sim_labels.extend(additional[1])
                self.sim_metas.extend(additional[2])
                self.sim_patterns.extend(additional[3])
                self.sim_variations.extend(additional[4])
                self.sim_lines_list.extend(additional[5])

    def get_space_group_number(self, id):

        # TODO: Check if symmetry information is correct using https://spglib.github.io/spglib/python-spglib.html
        # TODO: Pymatgen seems to be doing this already!
        # TODO: Bring them both together!!!

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

        # read space group number directly from the cif file
        with open(cif_path, "r") as file:
            for line in file:
                if "_space_group_IT_number" in line:
                    space_group_number = int(
                        line.replace("_space_group_IT_number ", "").strip()
                    )
                    # print(space_group_number)
                    return space_group_number

    def plot(self, indices=None, together=1):

        xs = np.linspace(angle_min, angle_max, angle_n)

        if indices is None:
            patterns = self.sim_patterns
        else:
            patterns = [self.sim_patterns[index] for index in indices]

        counter = 0
        for i, pattern in enumerate(patterns):
            for j, variation in enumerate(pattern):
                if variation is not None:
                    plt.plot(xs, variation, label=self.sim_variations[i][j])
                    lines = np.array(self.sim_lines_list[i][j])[:, 0] * 2
                    lines = lines[lines < 90]
                    plt.vlines(
                        lines, 0.8, 1, lw=0.1,
                    )

                    counter += 1

                    if (counter % together) == 0:
                        plt.legend()
                        plt.show()
