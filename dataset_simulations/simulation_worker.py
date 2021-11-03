# Since the xrayutilities package uses multiprocessing itself, having daemonic child processes turned out to be very slow and error-prune.
# Therefore, the main programs runs these worker scripts in parallel.

import sys
import lzma
import pickle
import random
import xrayutilities as xu
import numpy as np
import gc
import functools
import os

crystallite_size_gauss_min = 15 * 10 ** -9
crystallite_size_gauss_max = 50 * 10 ** -9
crystallite_size_lor_min = 15 * 10 ** -9
crystallite_size_lor_max = 50 * 10 ** -9

angle_min = 0
angle_max = 90
angle_n = 9001


def simulate_crystal(
    crystal, test_crystallite_sizes,
):  # keep this out of the class context to ensure thread safety
    # TODO: maybe add option for zero-point shifts

    try:

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

            # rs = []
            # for key, value in powder_model.pdiff[0].data.items():
            #    rs.append(value["r"])
            # print("Max intensity: " + str(np.max(rs)))

            powder_model.close()

            diffractograms.append(diffractogram)

            gc.collect()

    except BaseException as ex:

        try:
            powder_model.close()
        except:
            pass

        return (None, ex.__str__())

    return (diffractograms, variations)


if __name__ == "__main__":

    # make print statement always flush
    print = functools.partial(print, flush=True)

    print("Worker started with args:")
    print(sys.argv)

    status_file = sys.argv[1]
    start_from_scratch = True if sys.argv[2] == "True" else False
    test_crystallite_sizes = True if sys.argv[3] == "True" else False
    files_to_process = sys.argv[4:]

    with open(status_file, "w") as file:
        file.write("0")

    counter = 0

    for file in files_to_process:

        with lzma.open(file, "rb") as read_file:
            additional = pickle.load(read_file)

        sim_crystals = additional[0]
        sim_labels = additional[1]
        sim_metas = additional[2]
        sim_patterns = additional[3]
        sim_variations = additional[4]

        save_points = range(0, len(sim_crystals), int(len(sim_crystals) / 10) + 1)

        for i, pattern in enumerate(sim_patterns):
            counter += 1

            with open(status_file, "w") as write_file:
                write_file.write(str(counter))

            if not len(pattern) == 0 and not start_from_scratch:  # already processed
                continue

            crystal = sim_crystals[i]

            result = simulate_crystal(crystal, test_crystallite_sizes)

            if result[0] is not None:
                diffractograms, variatons = result
            else:
                sim_variations[i] = [None] * (5 if not test_crystallite_sizes else 6)
                sim_patterns[i] = [None] * (5 if not test_crystallite_sizes else 6)

                print(f"Encountered error for cif id {sim_metas[i]}.")
                print(result[1])

                continue

            sim_patterns[i] = diffractograms
            sim_variations[i] = variatons

            if (i % 5) == 0:
                if os.path.exists(os.path.join(os.path.dirname(status_file), "STOP")):
                    print("Simulation stopped by user.")
                    exit()

            if i in save_points:
                with lzma.open(file, "wb") as pickle_file:
                    pickle.dump(
                        (
                            sim_crystals,
                            sim_labels,
                            sim_metas,
                            sim_patterns,
                            sim_variations,
                        ),
                        pickle_file,
                    )

            gc.collect()
