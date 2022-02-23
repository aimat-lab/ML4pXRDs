from simulation import Simulation
import functools
import os
import numpy as np
import matplotlib.pyplot as plt


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

    def plot_histogram_of_spgs(self, do_show=True, logscale=True):

        spgs = [
            self.get_space_group_number(id) for id in self.icsd_ids
        ]  # TODO: Change back

        for i in reversed(range(0, len(spgs))):
            if spgs[i] is None:
                del spgs[i]

        print(f"Number of ICSD entries with available spg number: {len(spgs)}")

        plt.figure()
        plt.hist(spgs, bins=np.arange(1, 231) + 0.5)

        if logscale:
            plt.yscale("log")

        plt.xlabel("International space group number")
        plt.savefig(f"distribution_spgs{'_logscale' if logscale else ''}.png")

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
    else:
        simulation = ICSDSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    if True:

        simulation.plot_histogram_of_spgs(logscale=True)
        simulation.plot_histogram_of_spgs(logscale=False)

    if False:

        if False:  # toggle
            simulation.load(load_only=1)
        else:
            simulation.generate_structures()

        if True:

            simulation.simulate_all(start_from_scratch=True)

        # simulation.plot(together=5)
