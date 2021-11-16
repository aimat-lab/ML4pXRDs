from simulation import Simulation
import functools
import os
import numpy as np
import matplotlib.pyplot as plt


class ICSDSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/icsd/"

    def generate_structures(self):

        self.reset_simulation_status()

        counter = 0

        for i, path in enumerate(self.icsd_paths):

            if i == 2500:
                break

            print(f"Generated {i} structures.", flush=True)

            if path is None:
                counter += 1
                continue

            space_group_number = self.get_space_group_number(self.icsd_ids[i])

            if space_group_number is not None:
                result = self.add_crystal_to_be_simulated(
                    path, [space_group_number], [self.icsd_ids[i]],
                )

                if result is None:
                    counter += 1
                    continue
            else:
                counter += 1
                continue

        print(f"Skipped {counter} structures due to errors or missing path.")


if __name__ == "__main__":
    # make print statement always flush
    print = functools.partial(print, flush=True)

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        simulation = ICSDSimulation(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
    else:
        simulation = ICSDSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    if True:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()

    if False:

        simulation.simulate_all(start_from_scratch=True)

        # simulation.load()

    # simulation.plot(together=5)
