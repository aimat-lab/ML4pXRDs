from simulation import Simulation
import functools
import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifParser
from dataset_simulations.core.quick_simulation import get_xy_patterns
import matplotlib_defaults


class ICSDSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        # self.output_dir = "patterns/icsd_park/"
        self.output_dir = "patterns/icsd_vecsei/"

    def generate_structures(self):

        self.reset_simulation_status()

        counter = 0

        for i, path in enumerate(self.icsd_paths):

            print(f"Generated {i} structures.", flush=True)

            if path is None:
                counter += 1
                continue

            space_group_number = self.get_space_group_number(self.icsd_ids[i])

            if space_group_number is not None:
                result = self.add_path_to_be_simulated(
                    path,
                    [space_group_number],
                    [self.icsd_ids[i]],
                )

                if result is None:
                    counter += 1
                    continue
            else:
                counter += 1
                continue

        print(f"Skipped {counter} structures due to errors or missing path.")

    def plot_histogram_of_spgs(self, do_show=True):

        spgs = []
        for i, id in enumerate(self.icsd_ids):
            if (i % 100) == 0:
                print(f"{i/len(self.icsd_ids)*100}%")

            spg = self.get_space_group_number(id)
            if spg is not None:
                spgs.append(spg)

        print(f"Number of ICSD entries with available spg number: {len(spgs)}")

        plt.figure(
            figsize=(
                matplotlib_defaults.pub_width * 0.65,
                matplotlib_defaults.pub_width * 0.40,
            )
        )
        plt.hist(spgs, bins=np.arange(1, 231) + 0.5)

        plt.xlabel("International space group number")
        plt.ylabel("count")

        plt.tight_layout()
        plt.savefig(f"distribution_spgs.pdf")
        plt.yscale("log")
        plt.savefig(f"distribution_spgs_logscale.pdf")

        if do_show:
            plt.show()


if __name__ == "__main__":

    # make print statement always flush
    print = functools.partial(print, flush=True)

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        simulation = ICSDSimulation(
            os.path.expanduser("~/Databases/ICSD/ICSD_data_from_API.csv"),
            os.path.expanduser("~/Databases/ICSD/cif/"),
        )
    else:  # local
        simulation = ICSDSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    print()

    if False:

        id = 259366

        path = simulation.icsd_paths[simulation.icsd_ids.index(id)]

        parser = CifParser(path)
        crystals = parser.get_structures()
        crystal = crystals[0]

        print(crystal.atomic_numbers)

        data = get_xy_patterns(
            crystal, 1.2, np.linspace(0, 90, 9000), 1, (0, 90), False, False, False
        )[0]

        plt.plot(data)
        plt.show()

        # So it turns out that this is not really a big problem and is OK to happen...

        pass

    if True:
        simulation.plot_histogram_of_spgs(do_show=False)

    if False:

        if False:  # toggle
            simulation.load(load_only=1)
        else:
            simulation.generate_structures()

        if True:

            simulation.simulate_all(start_from_scratch=True)

        # simulation.plot(together=5)
