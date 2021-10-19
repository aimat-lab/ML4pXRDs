import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as pl
import time
import math
import os
import multiprocessing
import pandas as pd
from glob import glob

batch_size = 1000
total_threads = 16
return_mode = "pattern"  # only full pattern supported at the moment
simulation_mode = "xrayutilities"  # only xrayutilities supported at the moment
output_dir = ""


class Simulator:
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        self.crystals = []
        self.labels = []  # space group, etc.
        self.icsd_info_file_path = icsd_info_file_path
        self.icsd_cifs_dir = icsd_cifs_dir
        self.read_icsd_info()
        self.load_all_cif_paths()

    def track_job(job, update_interval=5):
        while job._number_left > 0:
            print(
                "Tasks remaining in this batch of 1000: {0}".format(
                    job._number_left * job._chunksize
                )
            )
            time.sleep(update_interval)

    def simulate_all(self):

        output_dir = "patterns/smart/"
        os.system(f"mkdir -p {output_dir}")

        print(f"Processing {len(self.crystals)} structures.")

        # put 1000 entries ("batch") into one file, process them at once:
        for i in range(0, math.ceil(len(self.crystals) / batch_size)):

            if os.path.exists(
                os.path.join(output_dir, "dataset_" + str(i) + ".csv")
            ):  # make it possible to continue later
                continue

            if ((i + 1) * batch_size) < len(self.crystals):
                end_index = (i + 1) * batch_size
                current_crystals = self.crystals[i * batch_size : end_index]
            else:
                current_crystals = self.crystals[i * batch_size :]
                end_index = len(self.crystals)

            pool = multiprocessing.Pool(
                processes=total_threads - 1
            )  # keep one main thread

            start = time.time()

            handle = pool.map_async(Simulator.simulate_crystal, current_crystals)

            Simulator.track_job(handle)

            result = handle.get()

            end = time.time()

            result = [
                x for x in result if x is not None
            ]  # None indicates an error in the structure
            result = np.array(result)

            np.savetxt(
                os.path.join(output_dir, "dataset_" + str(i) + ".csv"),
                np.concatenate(result, self.labels[i]),
                delimiter=" ",
                fmt="%s",
            )

            print(
                "##### Calculated from cif {} to {} (total: {}) in {} s".format(
                    i * batch_size, end_index, len(self.crystals), end - start
                )
            )

    def simulate_crystal(crystal):  # TODO: Add option to augment crystallite size

        powder = xu.simpack.Powder(
            crystal,
            1,
            crystallite_size_lor=2e-07,  # default
            crystallite_size_gauss=2e-07,  # default
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
                    "diffractometer_radius": 0.3,
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
        # TODO: Include zero-point shift at some point
        # TODO: Can I reuse this for other samples? Caching?

        xs = np.linspace(
            0, 90, 9001
        )  # simulate a rather large range, we can still later use a smaller range for training
        diffractogram = powder_model.simulate(
            xs
        )  # this also includes the Lorentzian + polarization correction

        return diffractogram

    def read_icsd_info(self):

        icsd_info = pd.read_csv(self.icsd_info_file_path, sep=",", skiprows=1)

        self.icsd_ids = icsd_info["CollectionCode"]
        self.icsd_space_group_symbols = icsd_info["HMS"]
        self.icsd_formulas = icsd_info["SumFormula"]
        self.icsd_structure_types = icsd_info["StructureType"]
        self.icsd_standardised_cell_parameters = icsd_info["StandardisedCellParameter"]
        self.icsd_r_values = icsd_info["RValue"]

    def load_all_cif_paths(self):

        all_filenames = glob(os.path.join(self.icsd_cifs_dir, "*.cif"))
        all_ids = [
            int(os.path.splitext(os.path.basename(filename))[0])
            for filename in all_filenames
        ]

        self.icsd_paths = [
            (all_filenames[all_ids.index(i)] if i in all_ids else None)
            for i in self.icsd_ids
        ]

        pass
