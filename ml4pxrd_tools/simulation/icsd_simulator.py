"""
This script can be used to simulate pXRD patterns of the ICSD database. Before
running this script, make sure that you change the variables on top of this
script file, the file `simulation_worker.py`, and `simulation_smeared.py`.
"""

import numpy as np
import time
import math
import os
import pandas as pd
from glob import glob
import pickle
from datetime import datetime
from subprocess import Popen
from math import ceil
import subprocess
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pymatgen.io.cif import CifParser
import re
import ml4pxrd_tools.matplotlib_defaults as matplotlib_defaults
import functools

num_files = (
    8  # Into how many files should the simulation of all patterns be saved / split?
)
num_processes = 8  # How many processes to use in parallel?

# Same as Vecsei et al. 2019:
angle_min = 5  # minimum 2 theta
angle_max = 90  # maximum 2 theta
angle_n = 8501  # How many steps in the angle range?

# Where to save the simulated patterns?
path_to_patterns = (
    "patterns/icsd_vecsei/"  # relative to the main directory of this repository
)

# Path to the ICSD directory that contains the "ICSD_data_from_API.csv" file
# and the "cif" directory (which contains all the ICSD cif files)
# We provide two separate variables for local execution and execution on
# a cluster using slurm.
path_to_icsd_directory_local = os.path.expanduser("~/Dokumente/Big_Files/ICSD/")
path_to_icsd_directory_cluster = os.path.expanduser("~/Databases/ICSD/")


class ICSDSimulator:
    def __init__(
        self, icsd_info_file_path, icsd_cifs_dir, output_dir="patterns/icsd_vecsei/"
    ):
        """Main simulator class to simulate pXRD patterns.

        Args:
            icsd_info_file_path (str): path to the ICSD_data_from_API.csv file
            icsd_cifs_dir (str): path to the directory containing the ICSD cif files
            output_dir (str): In which directory to save the simulated patterns.
                This should be relative to the main directory of the repository.
        """

        self.reset_simulation_status()

        self.icsd_info_file_path = icsd_info_file_path
        self.icsd_cifs_dir = icsd_cifs_dir
        self.read_icsd()

        self.output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ),
            output_dir,
        )

        print("Protocol started: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    def reset_simulation_status(self):
        """Reset all lists containing simulation data."""

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

    def simulate_all(self, start_from_scratch=False):
        """Simulate all crystals placed in self.sim_crystals.
        This will spawn several worker processes working on batches of the
        total amount of simulations to perform.

        Args:
            start_from_scratch (bool, optional): If some of the structures
                have already been simulated, this determines if the simulation
                should start from scratch. Defaults to False.
        """

        batch_size = ceil(
            len(self.sim_crystals) / num_files
        )  # How many crystals / patterns placed in each file?

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

        # Spawn the processes
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
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "simulation_worker.py",
                    ),
                    status_file_of_process,
                    "True" if start_from_scratch else "False",
                    *files_of_process,
                ],
                stdout=open(log_file, "a"),  # Pass stdout and stderr to the log_file
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
            # Print the status of each worker:
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
        """Process all entries in the ICSD_data_from_API.csv and match them with their corresponding
        cif file. When running this function for the first time, it will take a while.
        However, the results will be saved in the file "icsd_meta" in the cif directory
        for later reuse.
        """

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
        """Match the cif files with the entries of the ICSD_data_from_API.csv file."""

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

    def save(self, i=None):
        """Save the current simulation state.

        Args:
            i (int, optional): Index of the batch so save. If None, all batches will be saved to
            their corresponding data files. Defaults to None.
        """

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

    def load(
        self,
        start=None,
        stop=None,
        load_patterns_angles_intensities=True,
        load_only_N_patterns_each=None,
        metas_to_load=None,
        load_only_angles_intensities=False,
    ):
        """Load the current simulation status from the data files of each batch.

        Args:
            start (int, optional): Index of first batch to load. Defaults to None.
            stop (int, optional): Index of last batch to load (exclusive). Defaults to None.
            load_patterns_angles_intensities (bool, optional): Whether or not to load the
                patterns, angles, and intensities. Setting this to False can save a lot of
                memory if the data is not needed. Defaults to True.
            load_only_N_patterns_each (int, optional): How many patterns to load for each crystal.
                If None, all patterns (crystallite sizes) are loaded. Defaults to None.
            metas_to_load (list of int, optional): List of ICSD ids to load. All others will be
                skipped. Defaults to None, meaning that all ids will be loaded.
            load_only_angles_intensities (bool, optional): If True, this will skip the loading
                of patterns. This can save memory, if the patterns are not needed. Defaults to False.
        """

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

        if load_only_N_patterns_each is not None:
            self.sim_variations = [
                item[0:load_only_N_patterns_each] for item in self.sim_variations
            ]

        if load_patterns_angles_intensities:

            if not load_only_angles_intensities:

                for file in patterns_files[first_index:last_index]:
                    if (
                        load_only_N_patterns_each is not None
                    ):  # load patterns using memory mapping
                        self.sim_patterns.extend(
                            np.load(file, allow_pickle=True, mmap_mode="c")[
                                :, 0:load_only_N_patterns_each
                            ]
                        )
                    elif metas_to_load is not None:
                        self.sim_patterns.extend(
                            np.load(file, allow_pickle=True, mmap_mode="c")
                        )
                    else:
                        self.sim_patterns.extend(np.load(file, allow_pickle=True))

            for file in angles_files[first_index:last_index]:
                with open(file, "rb") as pickle_file:
                    self.sim_angles.extend(pickle.load(pickle_file))

            for file in intensities_files[first_index:last_index]:
                with open(file, "rb") as pickle_file:
                    self.sim_intensities.extend(pickle.load(pickle_file))

        if metas_to_load is not None:

            for i in reversed(range(0, len(self.sim_metas))):

                if self.sim_metas[i][0] not in metas_to_load:

                    del self.sim_metas[i]
                    del self.sim_crystals[i]
                    del self.sim_variations[i]
                    del self.sim_labels[i]

                    if load_patterns_angles_intensities:
                        if not load_only_angles_intensities:
                            del self.sim_patterns[i]
                        del self.sim_angles[i]
                        del self.sim_intensities[i]

    def get_space_group_number(self, id):
        """Read the space group number of the specified ICSD id
        directly from the cif file.

        Args:
            id (int): ICSD id

        Returns:
            int: Space group number. Returns None if the processing was not successful.
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
                    return space_group_number

        return None

    def __get_wyckoff_info(cif_path):
        """Return information about the asymmetric unit of the crystal.
        This is read directly from the cif file.

        Args:
            cif_path (str): path to the cif file of the crystal

        Returns:
            tuple: (No partial occupancies?, number of atoms in asymmetric unit,
            elements of the atoms in the asymmetric unit, occupancies of the atoms
            in the asymmetric unit, how often the same wyckoff position is repeated,
            how many unique wyckoff positions are occupied?, How many different wyckoff
            positions are occupied summed over the unique elements?, Unique list of
            occupied wyckoff positions)
        """

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

                    else:  # done

                        wyckoff_repetitions = []

                        all_wyckoffs = []

                        for key in wyckoffs_per_element.keys():
                            wyckoffs_unique = np.unique(wyckoffs_per_element[key])

                            all_wyckoffs.extend(wyckoffs_unique)

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
                            len(
                                np.unique(all_wyckoffs)
                            ),  # how many different wyckoff sites are occupied? "NO_unique_wyckoffs"
                            len(
                                all_wyckoffs
                            ),  # how many different wyckoff sites are occupied summed over unique elements. "NO_unique_wyckoffs_summed_over_els"
                            np.unique(all_wyckoffs),
                        )

                if "_atom_site_occupancy" in line or "_atom_site_occupance" in line:
                    counting = True

    def get_wyckoff_info(self, id):
        """Return information about the asymmetric unit of the crystal.
        This is read directly from the cif file, using the specified ICSD `id`.

        Args:
            id (int): ICSD id of the crystal

        Returns:
            tuple: (No partial occupancies?, number of atoms in asymmetric unit,
            elements of the atoms in the asymmetric unit, occupancies of the atoms
            in the asymmetric unit, how often the same wyckoff position is repeated,
            how many unique wyckoff positions are occupied?, How many different wyckoff
            positions are occupied summed over the unique elements?, Unique list of
            occupied wyckoff positions)
        """

        cif_path = self.icsd_paths[self.icsd_ids.index(id)]
        return ICSDSimulator.__get_wyckoff_info(cif_path)

    def plot(self, together=1):
        """Plot the loaded patterns one after the other.

        Args:
            together (int, optional): How many patterns should
            be plotted together in one figure?. Defaults to 1.
        """

        xs = np.linspace(angle_min, angle_max, angle_n)

        patterns = self.sim_patterns

        counter = 0
        for i, pattern in enumerate(patterns):
            for j, variation in enumerate(pattern):
                if variation[0] is not np.nan:

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

                    counter += 1

                    if (counter % together) == 0:
                        plt.legend()
                        plt.show()

    def add_path_to_be_simulated(self, path_to_crystal, labels, metas):
        """Add a crystal to the list of crystals to be simulated.

        Args:
            path_to_crystal (str): Path to cif file of the crystal
            labels (list of int): Spg label of the crystal in the format [label]
            metas (list of int): ICSD id of the crystal to be simulated in the format [id]

        Returns:
            int|None: 1 if successful else None
        """

        try:
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

    def get_content_types(self):
        """Get the content types of the ICSD ids.

        Returns:
            tuple: (list of exp inorganic ids, list of exp metalorganic ids, list of theoretical ids)
        """

        path = os.path.join(
            os.path.dirname(self.icsd_info_file_path), "ICSD_content_type.csv"
        )

        indices = np.genfromtxt(path, delimiter=",", skip_header=2, dtype=int)

        exp_inorganic = indices[:, 0]
        exp_metalorganic = indices[:, 1]
        theoretical = indices[:, 2]

        exp_inorganic = exp_inorganic[~np.isnan(exp_inorganic)]
        exp_metalorganic = exp_metalorganic[~np.isnan(exp_metalorganic)]
        theoretical = theoretical[~np.isnan(theoretical)]

        return exp_inorganic, exp_metalorganic, theoretical

    def prepare_simulation(self, use_only_N_crystals=None):
        """Prepare the simulation of the ICSD crystals. Read the space group labels.

        Args:
            use_only_N_crystals (int|None): If not None, use only that many crystals and skip the rest.
        """

        self.reset_simulation_status()

        counter = 0

        for i, path in enumerate(
            (
                self.icsd_paths
                if use_only_N_crystals is None
                else self.icsd_paths[0:use_only_N_crystals]
            )
        ):

            print(f"Generated {i} structures.", flush=True)

            if path is None:
                counter += 1
                continue

            space_group_number = self.get_space_group_number(self.icsd_ids[i])

            if space_group_number is not None:
                result = self.add_path_to_be_simulated(
                    path,
                    [space_group_number],
                    [self.icsd_ids[i]],
                )

                if result is None:
                    counter += 1
                    continue
            else:
                counter += 1
                continue

        print(f"Skipped {counter} structures due to errors or missing path.")

    def plot_histogram_of_spgs(self, do_show=True, process_only_N=None, do_sort=False):
        """Generate a histogram of the representation of the different spgs in the ICSD.

        Args:
            do_show (bool, optional): Whether or not to call plt.show(). If False,
                the histogram will only be saved to the disk. Defaults to True.
            process_only_N (int|None, optional): Process only N icsd entries instead of
                all. Defaults to None.
            do_sort (bool, optional): Whether or not to sort by count. Default to False.
        """

        spgs = []
        for i, id in enumerate(self.icsd_ids[0:process_only_N]):
            if (i % 100) == 0:
                print(f"{i/len(self.icsd_ids)*100}%")

            spg = self.get_space_group_number(id)
            if spg is not None:
                spgs.append(spg)

        print(f"Number of ICSD entries with available spg number: {len(spgs)}")

        plt.figure(
            figsize=(
                matplotlib_defaults.pub_width * 0.65,
                matplotlib_defaults.pub_width * 0.40,
            )
        )

        if not do_sort:
            plt.hist(spgs, bins=np.arange(0, 231) + 0.5)
            plt.xlabel("International space group number")
        else:
            hist, bin_edges = np.histogram(spgs, bins=np.arange(0, 231) + 0.5)
            plt.bar(np.arange(1, 231), np.sort(hist), width=1.0)
            plt.xlabel("Space groups (sorted by count)")

        plt.ylabel("count")

        plt.tight_layout()
        plt.savefig(f"distribution_spgs.pdf")
        plt.yscale("log")
        plt.savefig(f"distribution_spgs_logscale.pdf")

        if do_show:
            plt.show()


if __name__ == "__main__":

    # make print statement always flush
    print = functools.partial(print, flush=True)

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        path_to_icsd_directory = path_to_icsd_directory_cluster
    else:
        path_to_icsd_directory = path_to_icsd_directory_local

    simulator = ICSDSimulator(
        os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
        os.path.join(path_to_icsd_directory, "cif/"),
        output_dir=path_to_patterns,
    )

    # simulation.load()
    # simulator.prepare_simulation()
    # simulator.save()
    # simulator.simulate_all(start_from_scratch=True)

    # TODO: Change back

    simulator.plot_histogram_of_spgs(do_show=False, do_sort=True)
