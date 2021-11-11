# Since the xrayutilities package uses multiprocessing itself, having daemonic child processes turned out to be very slow and error-prune.
# Therefore, the main programs runs these worker scripts in parallel.

import sys
import random
import xrayutilities as xu
import numpy as np
import gc
import functools
import os
from dataset_simulations.spectrum_generation.peak_broadening import BroadGen
from train_dataset.generate_background_noise_utils import convert_to_discrete

sys.path.append("../")

# xrayutilities
xrayutil_crystallite_size_gauss_min = 15 * 10 ** -9
xrayutil_crystallite_size_gauss_max = 50 * 10 ** -9
xrayutil_crystallite_size_lor_min = 15 * 10 ** -9
xrayutil_crystallite_size_lor_max = 50 * 10 ** -9

# pymatgen
paymatgen_crystallite_size_gauss_min = 30 * 10 ** -9  # TODO: Change these ranges?
paymatgen_crystallite_size_gauss_max = 100 * 10 ** -9

angle_min = 0
angle_max = 90
angle_n = 9001


def simulate_crystal(
    crystal, test_crystallite_sizes, simulation_software, id
):  # keep this out of the class context to ensure thread safety
    # TODO: maybe add option for zero-point shifts

    try:

        diffractograms = []
        lines_lists = []
        variations = []
        peak_info_discs = []
        peak_size_discs = []

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

                variations.append({"size_gauss": size_gauss, "size_lor": size_lor})

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

                peak_positions = []
                peak_sizes = []
                for key, value in powder_model.pdiff[0].data.items():
                    if value["active"]:
                        peak_positions.append(value["ang"])
                        peak_sizes.append(value["r"])

                peak_sizes = np.array(peak_sizes) / np.max(peak_sizes)

                lines = list(zip(peak_positions, peak_sizes))
                lines_lists.append(lines)

                # lines_list = [None] * (5 if not test_crystallite_sizes else 6)

                peak_info_disc, peak_size_disc = convert_to_discrete(
                    peak_positions, peak_sizes, N=angle_n
                )

                peak_info_discs.append(peak_info_disc)
                peak_size_discs.append(peak_size_disc)

                powder_model.close()

                diffractograms.append(diffractogram)

                gc.collect()

            else:  # pymatgen

                broadener = BroadGen(
                    min_domain_size=paymatgen_crystallite_size_gauss_min,
                    max_domain_size=paymatgen_crystallite_size_gauss_max,
                    min_angle=angle_min,
                    max_angle=angle_max,
                )

                for i in range(0, 5 if not test_crystallite_sizes else 2):

                    if not test_crystallite_sizes:

                        diffractogram, domain_size = broadener.broadened_spectrum()

                    else:

                        # For comparing the different crystallite sizes
                        if i == 0:
                            size_gauss = paymatgen_crystallite_size_gauss_min
                        elif i == 1:
                            size_gauss = paymatgen_crystallite_size_gauss_max

                        (diffractogram, domain_size,) = broadener.broadened_spectrum(
                            domain_size=size_gauss
                        )

                    peak_positions = broadener.angles
                    peak_sizes = np.array(broadener.intensities) / np.max(
                        broadener.intensities
                    )
                    peak_info_disc, peak_size_disc = convert_to_discrete(
                        peak_positions, peak_sizes, N=angle_n
                    )
                    lines_list = list(zip(peak_positions, peak_sizes))

                    peak_info_discs.append(peak_info_disc)
                    peak_size_discs.append(peak_size_disc)
                    diffractograms.append(diffractogram)
                    variations.append(domain_size)
                    lines_lists.append(lines_list)

                    gc.collect()

    except BaseException as ex:

        try:
            powder_model.close()
        except:
            pass

        print(f"Encountered error for cif id {id}.")
        print(ex)

        diffractograms = [None] * (
            5
            if not test_crystallite_sizes
            else (6 if simulation_software == "xrayutilities" else 2)
        )
        variations = [None] * (
            5
            if not test_crystallite_sizes
            else (6 if simulation_software == "xrayutilities" else 2)
        )
        lines_lists = [None] * (
            5
            if not test_crystallite_sizes
            else (6 if simulation_software == "xrayutilities" else 2)
        )

    return (diffractograms, variations, lines_lists)


if __name__ == "__main__":

    # make print statement always flush
    print = functools.partial(print, flush=True)

    print("Worker started with args:")
    print(sys.argv)

    status_file = sys.argv[1]
    start_from_scratch = True if sys.argv[2] == "True" else False
    test_crystallite_sizes = True if sys.argv[3] == "True" else False
    simulation_software = sys.arv[4]
    files_to_process = sys.argv[5:]

    with open(status_file, "w") as file:
        file.write("0")

    counter = 0

    for file in files_to_process:

        id_str = os.path.basename(file).replace("crystals_", "").replace(".npy", "")

        sim_crystals = np.load(file)
        sim_metas = np.load(
            os.path.join(os.path.dirname(file), "metas_" + id_str + ".npy",)
        )  # just for printing the id when errors occurr

        sim_patterns_filepath = os.path.join(
            os.path.dirname(file), "patterns_" + id_str + ".npy",
        )
        sim_patterns = np.load(sim_patterns_filepath)

        sim_variations_filepath = os.path.join(
            os.path.dirname(file), "variations_" + id_str + ".npy",
        )
        sim_variations = np.load(sim_variations_filepath)

        sim_lines_lists_filepath = os.path.join(
            os.path.dirname(file), "lines_lists_" + id_str + ".npy",
        )
        sim_lines_lists = np.load(sim_lines_lists_filepath)

        save_points = range(0, len(sim_crystals), int(len(sim_crystals) / 10) + 1)

        for i, pattern in enumerate(sim_patterns):

            counter += 1

            if (i % 5) == 0:
                with open(status_file, "w") as write_file:
                    write_file.write(str(counter))

            if not len(pattern) == 0 and not start_from_scratch:  # already processed
                continue

            crystal = sim_crystals[i]

            result = simulate_crystal(
                crystal, test_crystallite_sizes, simulation_software, sim_metas[i]
            )

            diffractograms, variatons, lines_lists = result

            peak_info_disc, peak_size_disc = convert_to_discrete()

            sim_patterns[i] = diffractograms
            sim_variations[i] = variatons
            sim_lines_lists[i] = lines_lists

            if (i % 5) == 0:
                if os.path.exists(os.path.join(os.path.dirname(status_file), "STOP")):
                    print("Simulation stopped by user.")
                    exit()

            if i in save_points:
                np.save(sim_patterns, sim_patterns_filepath)
                np.save(sim_variations, sim_variations_filepath)
                np.save(sim_lines_lists, sim_lines_lists_filepath)

            gc.collect()
