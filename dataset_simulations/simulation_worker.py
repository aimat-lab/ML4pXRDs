# Since the xrayutilities package uses multiprocessing itself, having daemonic child processes turned out to be very slow and error-prune.
# Therefore, the main programs runs these worker scripts in parallel.

import sys
import random
import xrayutilities as xu
import numpy as np
import gc
import functools
import os
import pickle

sys.path.append("../")
from dataset_simulations.spectrum_generation.peak_broadening import BroadGen


# xrayutilities
xrayutil_crystallite_size_gauss_min = 15 * 10 ** -9
xrayutil_crystallite_size_gauss_max = 50 * 10 ** -9
xrayutil_crystallite_size_lor_min = 15 * 10 ** -9
xrayutil_crystallite_size_lor_max = 50 * 10 ** -9

# pymatgen, these are in nm
paymatgen_crystallite_size_gauss_min = 5  # TODO: Change these ranges? Look at them.
paymatgen_crystallite_size_gauss_max = 100

angle_min = 0
angle_max = 90
angle_n = 9018


def simulate_crystal(
    crystal, test_crystallite_sizes, simulation_software, id
):  # keep this out of the class context to ensure thread safety
    # TODO: maybe add option for zero-point shifts

    try:

        diffractograms = []
        variations = []

        angles = None
        intensities = None

        if simulation_software == "xrayutilities":

            # draw 5 crystallite sizes per crystal:
            for i in range(0, 5 if not test_crystallite_sizes else 6):

                if not test_crystallite_sizes:
                    size_gauss = random.uniform(
                        xrayutil_crystallite_size_gauss_min,
                        xrayutil_crystallite_size_gauss_max,
                    )
                    size_lor = random.uniform(
                        xrayutil_crystallite_size_lor_min,
                        xrayutil_crystallite_size_lor_max,
                    )

                else:

                    # For comparing the different crystallite sizes
                    if i == 0:
                        size_gauss = xrayutil_crystallite_size_gauss_max
                        size_lor = 3 * 10 ** 8
                    elif i == 2:
                        size_gauss = xrayutil_crystallite_size_gauss_max
                        size_lor = xrayutil_crystallite_size_lor_max
                    elif i == 1:
                        size_gauss = 3 * 10 ** 8
                        size_lor = xrayutil_crystallite_size_lor_max
                    elif i == 3:
                        size_gauss = xrayutil_crystallite_size_gauss_min
                        size_lor = 3 * 10 ** 8
                    elif i == 4:
                        size_gauss = 3 * 10 ** 8
                        size_lor = xrayutil_crystallite_size_lor_min
                    elif i == 5:
                        size_gauss = xrayutil_crystallite_size_lor_min
                        size_lor = xrayutil_crystallite_size_gauss_min

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

                if i == 0:  # since it is the same for all variations, only do it once
                    peak_positions = []
                    peak_sizes = []
                    for key, value in powder_model.pdiff[0].data.items():
                        if value["active"]:
                            peak_positions.append(value["ang"])
                            peak_sizes.append(value["r"])

                    peak_sizes = np.array(peak_sizes) / np.max(peak_sizes)

                    intensities = peak_sizes
                    angles = peak_positions

                powder_model.close()

                diffractograms.append(diffractogram)
                variations.append([size_gauss, size_lor])

                gc.collect()

        else:  # pymatgen

            broadener = BroadGen(
                crystal,
                min_domain_size=paymatgen_crystallite_size_gauss_min,
                max_domain_size=paymatgen_crystallite_size_gauss_max,
                min_angle=angle_min,
                max_angle=angle_max,
            )

            for i in range(0, 5 if not test_crystallite_sizes else 2):

                if not test_crystallite_sizes:

                    diffractogram, domain_size = broadener.broadened_spectrum(N=angle_n)

                else:

                    # For comparing the different crystallite sizes
                    if i == 0:
                        size_gauss = paymatgen_crystallite_size_gauss_min
                    elif i == 1:
                        size_gauss = paymatgen_crystallite_size_gauss_max

                    (
                        diffractogram,
                        domain_size,
                    ) = broadener.broadened_spectrum(domain_size=size_gauss, N=angle_n)

                if i == 0:
                    peak_positions = broadener.angles
                    peak_sizes = np.array(broadener.intensities) / np.max(
                        broadener.intensities
                    )

                    angles = peak_positions
                    intensities = peak_sizes

                diffractograms.append(diffractogram)
                variations.append([domain_size])

                gc.collect()

    except BaseException as ex:

        try:
            powder_model.close()
        except:
            pass

        print(f"Encountered error for cif id {id}.")
        print(ex)

        diffractograms = [[np.nan] * angle_n] * (
            5
            if not test_crystallite_sizes
            else (6 if simulation_software == "xrayutilities" else 2)
        )
        if simulation_software == "xrayutilities":
            variations = [[np.nan, np.nan]] * 5 if not test_crystallite_sizes else 6
        else:
            variations = [[np.nan]] * 5 if not test_crystallite_sizes else 2

        angles = [np.nan] * angle_n
        intensities = [np.nan] * angle_n

    return (diffractograms, variations, angles, intensities)


if __name__ == "__main__":

    # make print statement always flush
    print = functools.partial(print, flush=True)

    print("Worker started with args:")
    print(sys.argv)

    status_file = sys.argv[1]
    start_from_scratch = True if sys.argv[2] == "True" else False
    test_crystallite_sizes = True if sys.argv[3] == "True" else False
    simulation_software = sys.argv[4]
    files_to_process = sys.argv[5:]

    counter = 0

    for file in files_to_process:

        id_str = os.path.basename(file).replace("crystals_", "").replace(".npy", "")

        sim_crystals = np.load(file, allow_pickle=True)
        sim_metas = np.load(
            os.path.join(
                os.path.dirname(file),
                "metas_" + id_str + ".npy",
            ),
            allow_pickle=True,
        )  # just for printing the id when errors occurr

        sim_patterns_filepath = os.path.join(
            os.path.dirname(file),
            "patterns_" + id_str + ".npy",
        )
        sim_patterns = np.load(sim_patterns_filepath, allow_pickle=True)

        sim_variations_filepath = os.path.join(
            os.path.dirname(file),
            "variations_" + id_str + ".npy",
        )
        sim_variations = np.load(sim_variations_filepath, allow_pickle=True)

        sim_angles_filepath = os.path.join(
            os.path.dirname(file),
            "angles_" + id_str + ".npy",
        )
        with open(sim_angles_filepath, "rb") as pickle_file:
            sim_angles = pickle.load(pickle_file)

        sim_intensities_filepath = os.path.join(
            os.path.dirname(file),
            "intensities_" + id_str + ".npy",
        )
        with open(sim_intensities_filepath, "rb") as pickle_file:
            sim_intensities = pickle.load(pickle_file)

        save_points = range(0, len(sim_crystals), int(len(sim_crystals) / 10) + 1)

        for i, pattern in enumerate(sim_patterns):

            counter += 1

            if (i % 5) == 0:
                with open(status_file, "w") as write_file:
                    write_file.write(str(counter))

            if pattern is not None and not start_from_scratch:  # already processed
                continue

            crystal = sim_crystals[i]

            result = simulate_crystal(
                crystal, test_crystallite_sizes, simulation_software, sim_metas[i]
            )

            diffractograms, variations, angles, intensities = result

            sim_patterns[i] = diffractograms
            sim_variations[i] = variations
            sim_angles[i] = angles
            sim_intensities[i] = intensities

            if (i % 5) == 0:
                if os.path.exists(os.path.join(os.path.dirname(status_file), "STOP")):
                    print("Simulation stopped by user.")
                    exit()

            if i in save_points:
                np.save(sim_patterns_filepath, np.array(sim_patterns, dtype=object))
                np.save(sim_variations_filepath, np.array(sim_variations, dtype=object))

                # these have variable dimensions, so cannot save them as a numpy array
                with open(sim_angles_filepath, "wb") as pickle_file:
                    pickle.dump(sim_angles, pickle_file)
                with open(sim_intensities_filepath, "wb") as pickle_file:
                    pickle.dump(sim_intensities, pickle_file)

            gc.collect()

        # in the end save it as a proper numpy array
        np.save(sim_patterns_filepath, np.array(sim_patterns.tolist(), dtype=float))
        np.save(sim_variations_filepath, np.array(sim_variations.tolist(), dtype=float))
        with open(sim_angles_filepath, "wb") as pickle_file:
            pickle.dump(sim_angles, pickle_file)
        with open(sim_intensities_filepath, "wb") as pickle_file:
            pickle.dump(sim_intensities, pickle_file)
