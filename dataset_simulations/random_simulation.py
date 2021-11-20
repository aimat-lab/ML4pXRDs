import sys
import os
import random_simulation_utils

sys.path.append("../")
from dataset_simulations.simulation import Simulation

space_groups = [3, 4]
N_per_space_group = 100


class RandomSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir, output_dir=None):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        if output_dir is None:
            self.output_dir = "patterns/random/"
        else:
            self.output_dir = output_dir

    def generate_structures(self):

        for spg in space_groups:

            structures = random_simulation_utils.generate_structures(
                spg, N_per_space_group
            )

            for structure in structures:
                self.add_crystal_to_be_simulated(structure, [spg], [0])


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

    if False:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()

    if True:

        simulation.simulate_all(start_from_scratch=True)

        # simulation.load()

    # simulation.plot(together=5)
