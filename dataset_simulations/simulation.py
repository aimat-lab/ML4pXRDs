import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as pl
import time
import math
import os
import multiprocessing
import multiprocessing.pool
import pandas as pd
from glob import glob
import pickle
import random
import itertools
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import lzma
import gc
from datetime import datetime

from xrayutilities.simpack import powdermodel

batch_size = 4000
num_threads = 80
return_mode = "pattern"  # only full pattern supported at the moment
simulation_mode = "xrayutilities"  # only xrayutilities supported at the moment

crystallite_size_gauss_min = 15 * 10 ** -9
crystallite_size_gauss_max = 50 * 10 ** -9
crystallite_size_lor_min = 15 * 10 ** -9
crystallite_size_lor_max = 50 * 10 ** -9

angle_min = 0
angle_max = 90
angle_n = 9001


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


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
        self.sim_batches_simulated = 0

    def __track_job(job, update_interval=5):
        while job._number_left > 0:
            print(
                "Tasks remaining in this batch of {}: {} (chunksize: {})".format(
                    batch_size, job._number_left * job._chunksize, job._chunksize
                )
            )
            time.sleep(update_interval)

    def simulate_all(self, test_crystallite_sizes=False, start_from_scratch=False):

        # only for testing:
        # self.sim_crystals = self.sim_crystals[0:16]

        os.system(f"mkdir -p {self.output_dir}")

        print(f"Simulating {len(self.sim_crystals)} structures.")
        print(f"Processing {batch_size} structures in each batch.")
        print(f"Saving after each batch.")

        crystals_to_process = self.sim_crystals  # process all crystals
        crystals_to_process_indices = range(
            len(self.sim_crystals)
        )  # needed, so the parallely processed jobs know where to put the results

        if start_from_scratch == True:

            self.sim_patterns = [[]] * len(self.sim_crystals)
            self.sim_variations = [[]] * len(self.sim_crystals)

            print("Starting simulation from scratch...")

            self.sim_batches_simulated = 0

        else:

            sim_batches_simulated_file = os.path.join(
                self.output_dir, "sim_batches_simulated"
            )
            if os.path.exists(sim_batches_simulated_file):
                with open(sim_batches_simulated_file, "r") as file:
                    self.sim_batches_simulated = int(file.readline().strip())

        for i in range(
            self.sim_batches_simulated, math.ceil(len(crystals_to_process) / batch_size)
        ):

            gc.collect()

            print(
                f"Processing batch {i+1} of {math.ceil(len(crystals_to_process) / batch_size)} with batch size {batch_size}"
            )
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            if ((i + 1) * batch_size) < len(crystals_to_process):
                end = (i + 1) * batch_size
                current_crystals = crystals_to_process[i * batch_size : end]
                current_crystals_indices = crystals_to_process_indices[
                    i * batch_size : end
                ]
            else:
                current_crystals = crystals_to_process[i * batch_size :]
                current_crystals_indices = crystals_to_process_indices[i * batch_size :]
                end = len(crystals_to_process)

            start_time = time.time()

            pool = NestablePool(processes=num_threads)  # keep one main thread

            handle = pool.map_async(
                Simulation.simulate_crystal,
                zip(current_crystals, itertools.repeat(test_crystallite_sizes)),
            )

            Simulation.__track_job(handle)

            results = handle.get()

            end_time = time.time()

            print(
                f"Finished batch of {batch_size} after {(end_time - start_time)/60} min"
            )

            for j, result in enumerate(results):
                (diffractograms, variations) = result
                index = current_crystals_indices[j]

                self.sim_patterns[index] = diffractograms
                self.sim_variations[index] = variations

            self.sim_batches_simulated += 1

            self.save(i)  # save after each batch to continue later

            del handle
            pool.close()
            del pool
            gc.collect()

    def simulate_crystal(
        arguments,
    ):  # keep this out of the class context to ensure thread safety
        # TODO: maybe add option for zero-point shifts

        crystal = arguments[0]
        test_crystallite_sizes = arguments[1]

        diffractograms = []
        variations = []

        # draw 5 crystallite sizes per crystal:
        for i in range(0, 5 if not test_crystallite_sizes else 6):

            if not test_crystallite_sizes:
                size_gauss = random.uniform(
                    crystallite_size_gauss_min, crystallite_size_gauss_max
                )
                size_lor = random.uniform(
                    crystallite_size_lor_min, crystallite_size_lor_max
                )

            else:

                # For comparing the different crystallite sizes
                if i == 0:
                    size_gauss = crystallite_size_gauss_max
                    size_lor = 3 * 10 ** 8
                elif i == 2:
                    size_gauss = crystallite_size_gauss_max
                    size_lor = crystallite_size_lor_max
                elif i == 1:
                    size_gauss = 3 * 10 ** 8
                    size_lor = crystallite_size_lor_max
                elif i == 3:
                    size_gauss = crystallite_size_gauss_min
                    size_lor = 3 * 10 ** 8
                elif i == 4:
                    size_gauss = 3 * 10 ** 8
                    size_lor = crystallite_size_lor_min
                elif i == 5:
                    size_gauss = crystallite_size_lor_min
                    size_lor = crystallite_size_gauss_min

            variations.append({"size_gauss": size_gauss, "size_lor": size_lor})

            try:

                powder = xu.simpack.Powder(
                    crystal,
                    1,
                    crystallite_size_lor=size_lor,  # default: 2e-07
                    crystallite_size_gauss=size_gauss,  # default: 2e-07
                    strain_lor=0,  # default
                    strain_gauss=0,  # default
                    preferred_orientation=(0, 0, 0),  # default
                    preferred_orientation_factor=1,  # default
                )

                # default parameters are in ~/.xrayutilities.conf
                # Alread set in config: Use one thread only
                # or use print(powder_model.pdiff[0].settings)
                # Further information on the settings can be found here: https://nvlpubs.nist.gov/nistpubs/jres/120/jres.120.014.c.py
                powder_model = xu.simpack.PowderModel(
                    powder,
                    I0=100,
                    fpsettings={
                        "classoptions": {
                            "anglemode": "twotheta",
                            "oversampling": 4,
                            "gaussian_smoother_bins_sigma": 1.0,
                            "window_width": 20.0,
                        },
                        "global": {
                            "geometry": "symmetric",
                            "geometry_incidence_angle": None,
                            "diffractometer_radius": 0.3,  # measured on experiment: 19.3 cm
                            "equatorial_divergence_deg": 0.5,
                            # "dominant_wavelength": 1.207930e-10, # this is a read-only setting!
                        },
                        "emission": {
                            "emiss_wavelengths": (1.207930e-10),
                            "emiss_intensities": (1.0),
                            "emiss_gauss_widths": (3e-14),
                            "emiss_lor_widths": (3e-14),
                            # "crystallite_size_lor": 2e-07,  # this needs to be set for the powder
                            # "crystallite_size_gauss": 2e-07,  # this needs to be set for the powder
                            # "strain_lor": 0,  # this needs to be set for the powder
                            # "strain_gauss": 0,  # this needs to be set for the powder
                            # "preferred_orientation": (0, 0, 0),  # this needs to be set for the powder
                            # "preferred_orientation_factor": 1,  # this needs to be set for the powder
                        },
                        "axial": {
                            "axDiv": "full",
                            "slit_length_source": 0.008001,
                            "slit_length_target": 0.008,
                            "length_sample": 0.01,
                            "angI_deg": 2.5,
                            "angD_deg": 2.5,
                            "n_integral_points": 10,
                        },
                        "absorption": {"absorption_coefficient": 100000.0},
                        "si_psd": {"si_psd_window_bounds": None},
                        "receiver_slit": {"slit_width": 5.5e-05},
                        "tube_tails": {
                            "main_width": 0.0002,
                            "tail_left": -0.001,
                            "tail_right": 0.001,
                            "tail_intens": 0.001,
                        },
                    },
                )

                # powder_model = xu.simpack.PowderModel(powder, I0=100,) # with default parameters
                # print(powder_model.pdiff[0].settings)

                xs = np.linspace(
                    angle_min, angle_max, angle_n
                )  # simulate a rather large range, we can still later use a smaller range for training

                diffractogram = powder_model.simulate(
                    xs, mode="local"
                )  # this also includes the Lorentzian + polarization correction

                rs = []
                for key, value in powder_model.pdiff[0].data.items():
                    rs.append(value["r"])
                # print("Max intensity: " + str(np.max(rs)))

                powder_model.close()

                diffractograms.append(diffractogram)

            except Exception as ex:

                diffractograms.append(None)

        return (diffractograms, variations)

        # return (np.random.random(size=9001), np.random.random(size=5))

    def read_icsd(self):

        pickle_file = os.path.join(self.icsd_cifs_dir, "icsd_meta")
        if not os.path.exists(pickle_file):
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

            print(f"{i} of {len(self.icsd_paths)}")

            with open(path) as f:
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

    def save(self, i=None):  # i specifies which batch to save

        if not os.path.exists(self.output_dir):
            os.system("mkdir -p " + self.output_dir)

        sim_batches_simulated_file = os.path.join(
            self.output_dir, "sim_batches_simulated"
        )
        with open(sim_batches_simulated_file, "w") as file:
            file.write(str(self.sim_batches_simulated))

        if i is not None:
            if ((i + 1) * batch_size) < len(self.sim_crystals):
                start = i * batch_size
                end = (i + 1) * batch_size
            else:
                start = i * batch_size
                end = len(self.sim_crystals)
        else:  # save all

            for j in range(0, math.ceil(len(self.sim_crystals) / batch_size)):
                self.save(i=j)

            return

        pickle_file = os.path.join(self.output_dir, f"data_{i}.lzma")

        with lzma.open(pickle_file, "wb") as file:
            pickle.dump(
                (
                    self.sim_crystals[start:end],
                    self.sim_labels[start:end],
                    self.sim_metas[start:end],
                    self.sim_patterns[start:end],
                    self.sim_variations[start:end],
                ),
                file,
            )

    def load(self):

        self.reset_simulation_status()

        sim_batches_simulated_file = os.path.join(
            self.output_dir, "sim_batches_simulated"
        )
        if os.path.exists(sim_batches_simulated_file):
            with open(sim_batches_simulated_file, "r") as file:
                self.sim_batches_simulated = int(file.readline().strip())

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
