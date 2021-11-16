import sys
import os

sys.path.append("../")
from dataset_simulations.simulation import Simulation

# Use a very narrow selection of ICSD entries

# Composition:
# Ce O2, Fm-3m (Fluorite)
# Y2 O3, bixbyite
# Lanthanum hydroxide La (O H)3, P63/m (not possible for HEOs)


class NarrowSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir, output_dir=None):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        if output_dir is None:
            self.output_dir = "patterns/narrow/"
        else:
            self.output_dir = output_dir

    def generate_structures(self):

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0

        for i, path in enumerate(self.icsd_paths):

            if (i % 1000) == 0:
                print(f"Processed {i} structures.")

            if self.icsd_structure_types[i] == "Fluorite#CaF2":
                label = 0
            elif self.icsd_structure_types[i] == "Bixbyite#(MnFe)O3":
                label = 1
            elif self.icsd_structure_types[i] == "UCl3":
                label = 2
            else:
                continue

            result = self.add_crystal_to_be_simulated(path, [label], [self.icsd_ids[i]])

            if result == 1:
                if self.icsd_structure_types[i] == "Fluorite#CaF2":
                    counter_0 += 1
                elif self.icsd_structure_types[i] == "Bixbyite#(MnFe)O3":
                    counter_1 += 1
                elif self.icsd_structure_types[i] == "UCl3":
                    counter_2 += 1

        print(f"Loaded {len(self.sim_crystals)} crystals")
        print(f"Fluorite#CaF2: {counter_0}")
        print(f"Bixbyite#(MnFe)O3: {counter_1}")
        print(f"UCl3: {counter_2}")


if __name__ == "__main__":

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        simulation = NarrowSimulation(
            "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/kit/iti/la2559/Databases/ICSD/cif/",
        )
    else:
        simulation = NarrowSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )

    if False:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()

    if True:

        simulation.simulate_all(start_from_scratch=True)

        simulation.load()

    # simulation.plot(together=5)
