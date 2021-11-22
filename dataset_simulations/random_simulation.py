import sys
import os
import random_simulation_utils
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from dataset_simulations.simulation import Simulation

space_groups = [14, 104]
N_per_space_group = 10000


class RandomSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir, output_dir=None):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        if output_dir is None:
            self.output_dir = "patterns/random/"
        else:
            self.output_dir = output_dir

    def generate_structures(self):

        for spg in space_groups:

            print(f"Generating structures of space group #{spg}")

            start = time.time()

            structures = random_simulation_utils.generate_structures(
                spg, N_per_space_group
            )

            stop = time.time()
            print(
                f"Took {stop-start} s to generate {N_per_space_group} structures for space group #{spg}"
            )

            for structure in structures:
                self.add_crystal_to_be_simulated(
                    structure, [spg], [0]
                )  # meta not needed here


if __name__ == "__main__":

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        simulation = RandomSimulation(
            "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/kit/iti/la2559/Databases/ICSD/cif/",
        )
    else:
        simulation = RandomSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    """
    spg_numbers = []
    for i, id in enumerate(simulation.icsd_ids):
        if (i % 1000) == 0:
            print(i)

        spg_number = simulation.get_space_group_number(id)

        if spg_number is not None:
            spg_numbers.append(spg_number)

    plt.hist(spg_numbers, bins=np.arange(0, 230))
    plt.show()

    counts = np.bincount(spg_numbers)
    counts = counts[1:]

    most_index = np.argmax(counts)
    print()
    print(f"Most: {most_index + 1} with {counts[most_index]} entries")
    least_index = np.argmin(counts)
    print(f"Least: {least_index + 1} with {counts[least_index]} entries")

    # Let's pick these: 104 with 23 structures
    #                   14 with 20607 entries
    exit()
    """

    """
    lengths = []
    for i, id in enumerate(simulation.icsd_formulas):
        lengths.append(len(id.split(" ")))

    plt.hist(lengths, bins=np.arange(0, np.max(lengths)))
    plt.show()
    """

    if True:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()

    # simulation.save()

    if True:
        pass

    # simulation.plot(together=10)

    print()

    # simulation.simulate_all(start_from_scratch=True)

    # simulation.load()

    # simulation.plot(together=5)
