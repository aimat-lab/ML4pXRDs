"""The simulator class uses this script to simulate pXRD patterns.
It spawns multiple processes of this script in parallel. We have been using a
different simulation code in the past which did not work well with python's
multiprocessing.
"""

import sys
import numpy as np
import gc
import functools
import os
import pickle
import time
from ml4pxrd_tools.simulation.simulation_smeared import get_smeared_patterns

wavelength = 1.5406  # Cu-K line

# same as Vecsei et al.:
angle_min = 5
angle_max = 90
angle_n = 8501

# Skip crystals with a unit cell volume > max_volume.
max_volume = 20000  # None possible
NO_corn_sizes = 5  # How many patterns to simulate per crystal


def process_crystal(crystal, id):

    try:

        diffractograms = []
        variations = []

        angles = None
        intensities = None

        if max_volume is not None and crystal.volume > max_volume:
            raise Exception(f"Volume higher than max_volume ({max_volume})")

        xs = np.linspace(angle_min, angle_max, angle_n)

        diffractograms, corn_sizes, angles, intensities = get_smeared_patterns(
            structure=crystal,
            wavelength=wavelength,
            xs=xs,
            NO_corn_sizes=NO_corn_sizes,
            two_theta_range=(angle_min, angle_max),
            return_corn_sizes=True,
            return_angles_intensities=True,
            return_max_unscaled_intensity_angle=False,
        )

        for corn_size in corn_sizes:
            variations.append([corn_size])

        gc.collect()

    except BaseException as ex:

        print(f"Encountered error for cif id {id}.")
        print(ex)

        diffractograms = [[np.nan] * angle_n] * NO_corn_sizes
        variations = [[np.nan]] * NO_corn_sizes

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
    files_to_process = sys.argv[3:]

    counter = 0

    for file in files_to_process:

        # Load the current state of this chunk of the simulation
        # If it hasn't been processed yet, sim_patterns will be None and we need to simulate them
        # The patterns that haven't been simulated yet will still appear in the list sim_patterns,
        # but they will be None.

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

        save_points = range(0, len(sim_crystals), int(len(sim_crystals) / 20) + 1)

        timings = []

        for i, pattern in enumerate(sim_patterns):

            counter += 1

            # Update the status for the Simulator
            if (i % 5) == 0:
                with open(status_file, "w") as write_file:
                    write_file.write(str(counter))

            if (
                pattern is not None and not start_from_scratch
            ):  # already processed, go to next pattern
                continue

            crystal = sim_crystals[i]

            start = time.time()
            result = process_crystal(crystal, sim_metas[i])
            stop = time.time()
            timings.append(stop - start)

            diffractograms, variations, angles, intensities = result

            sim_patterns[i] = diffractograms
            sim_variations[i] = variations
            sim_angles[i] = angles
            sim_intensities[i] = intensities

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

        print(f"Timings:: {timings}")
        print(f"Average timing: {np.average(timings)}")
        print(f"Max timing: {np.max(timings)}")
        print(f"Min timing: {np.min(timings)}")
        print(f"Median timing: {np.median(timings)}")

        # in the end save it as a proper numpy array
        np.save(sim_patterns_filepath, np.array(sim_patterns.tolist(), dtype=float))
        np.save(sim_variations_filepath, np.array(sim_variations.tolist(), dtype=float))
        with open(sim_angles_filepath, "wb") as pickle_file:
            pickle.dump(sim_angles, pickle_file)
        with open(sim_intensities_filepath, "wb") as pickle_file:
            pickle.dump(sim_intensities, pickle_file)
