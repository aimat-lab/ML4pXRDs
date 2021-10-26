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
import gzip

batch_size = 1000
num_threads = 8
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

        self.crystals = []
        self.patterns = []
        self.labels = []  # space group, etc.
        self.metas = []  # meta data: icsd id, ...

        self.icsd_info_file_path = icsd_info_file_path
        self.icsd_cifs_dir = icsd_cifs_dir
        self.read_icsd()

        self.output_dir = "patterns/default/"  # should be overwritten by child class

    def __track_job(job, update_interval=5):
        while job._number_left > 0:
            print(
                "Tasks remaining in this batch of 1000: {0}".format(
                    job._number_left * job._chunksize
                )
            )
            time.sleep(update_interval)

    def simulate_all(self, test_crystallite_sizes=False):

        self.crystals = self.crystals[0:16]  # only for testing

        os.system(f"mkdir -p {self.output_dir}")

        print(f"Simulating {len(self.crystals)} structures.")

        # put 1000 entries ("batch") into one file, process them at once:
        for i in range(0, math.ceil(len(self.crystals) / batch_size)):

            if os.path.exists(
                os.path.join(self.output_dir, "dataset_" + str(i) + ".csv")
            ):  # make it possible to continue later, skip already simulated patterns
                continue

            if ((i + 1) * batch_size) < len(self.crystals):
                end_index = (i + 1) * batch_size
                current_crystals = self.crystals[i * batch_size : end_index]
            else:
                current_crystals = self.crystals[i * batch_size :]
                end_index = len(self.crystals)

            start = time.time()

            pool = NestablePool(processes=num_threads)  # keep one main thread

            handle = pool.map_async(
                Simulation.simulate_crystal,
                zip(current_crystals, itertools.repeat(test_crystallite_sizes)),
            )

            Simulation.__track_job(handle)

            result = handle.get()

            # result = [
            #    Simulator.simulate_crystal(crystal) for crystal in current_crystals
            # ]

            end = time.time()

            print(
                "##### Calculated from cif {} to {} (total: {}) in {} s".format(
                    i * batch_size, end_index, len(self.crystals), end - start
                )
            )

            to_save = [
                np.append(item, [*self.labels[i], *self.metas[i]])
                for i, sublist in enumerate(result)
                for item in sublist
                if item is not None
            ]  # None indicates an error in the structure

            to_save = np.array(to_save)

            np.savetxt(
                os.path.join(self.output_dir, "dataset_" + str(i) + ".csv.gz"),
                to_save,
                delimiter=" ",
                fmt="%s",
                header=f"{len(self.labels[0])} {len(self.metas[0])}",
            )

        # load all already simulated patterns (including previous run)
        self.load_simulated_patterns_labels_metas()

    def simulate_crystal(arguments):
        # TODO: add option for zero-point shifts

        crystal = arguments[0]
        test_crystallite_sizes = arguments[1]

        diffractograms = []

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

                # diffractogram = powder_model.simulate(
                #    xs
                # )  # this also includes the Lorentzian + polarization correction

                powder_model.close()

                diffractograms.append(diffractogram)

            except Exception as ex:

                diffractograms.append(None)

        return diffractograms

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

            self.load_all_cif_paths()

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

    def load_all_cif_paths(self):

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

    def save_crystals_pickle(self):

        pickle_file = os.path.join(self.output_dir, "crystals_labels")

        with open(pickle_file, "wb") as file:
            pickle.dump(self.crystals, file)

    def load_crystals_pickle(self):

        pickle_file = os.path.join(self.output_dir, "crystals_labels")

        with open(pickle_file, "rb") as file:
            self.crystals = pickle.load(file)

    def load_simulated_patterns_labels_metas(
        self,
    ):  # loads patterns, labels and metadata from csv

        self.labels = []
        self.metas = []
        self.patterns = np.zeros((0, angle_n))

        csv_files = glob(os.path.join(self.output_dir, "*.csv.gz"))

        for csv_file in csv_files:
            with gzip.open(csv_file, "rt") as file:
                info = file.readline()[1:].strip().split(" ")
                n_labels = int(info[0])
                n_metas = int(info[1])

            data = np.genfromtxt(csv_file, delimiter=" ", skip_header=1)

            ys_simulated = data[:, :angle_n]
            labels = data[:, angle_n : angle_n + n_labels]
            metas = data[:, angle_n + n_labels :]

            self.labels.extend(labels.tolist())
            self.metas.extend(metas.tolist())
            self.patterns = np.append(self.patterns, ys_simulated, axis=0)
