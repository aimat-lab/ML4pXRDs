from tools.simulation.simulator import Simulator
import functools
import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifParser
from tools.simulation.simulation_core import get_xy_patterns
import tools.matplotlib_defaults as matplotlib_defaults


class ICSDSimulator(Simulator):
    def __init__(
        self, icsd_info_file_path, icsd_cifs_dir, output_dir="patterns/icsd_vecsei/"
    ):
        """This class can be used to simulate pXRD patterns for the crystals in the ICSD
        and also to access the results of this simulation.

        Args:
            icsd_info_file_path (str): path to the ICSD_data_from_API.csv file
            icsd_cifs_dir (str): path to the directory containing the ICSD cif files
            output_dir (str): In which directory to save the simulated patterns
        """

        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = output_dir

    def prepare_simulation(self):
        """Prepare the simulation of the ICSD crystals. Read the space group labels."""

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
        """Generate a histogram of the representation of the different spgs in the ICSD.

        Args:
            do_show (bool, optional): Whether or not to call plt.show(). If False,
            the histogram will only be saved to the disk. Defaults to True.
        """

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
        path_to_icsd_directory = os.path.expanduser("~/Databases/ICSD/")
    else:
        path_to_icsd_directory = os.path.expanduser("~/Dokumente/Big_Files/ICSD/")

    simulator = ICSDSimulator(
        os.path.join(path_to_icsd_directory, "ICSD_data_from_API.csv"),
        os.path.join(path_to_icsd_directory, "cif/"),
    )

    # simulation.load()
    simulator.prepare_simulation()
    simulator.save()
    simulator.simulate_all(start_from_scratch=True)
