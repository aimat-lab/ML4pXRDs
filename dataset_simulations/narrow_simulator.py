from dataset_simulations.simulator import Simulator

# Use a very narrow selection of ICSD entries, just for POC

# Composition:
# Ce O2, Fm-3m (Fluorite)
# Y2 O3, bixbyite
#


class NarrowSimulator(Simulator):
    def __init__(self):
        self.output_dir = "patterns/narrow/"

    def generate_structures(self):
        pass

        # fill self.structures + self.labels


"""
        crystal = xu.materials.Crystal.fromCIF(
            "/home/henrik/Dokumente/ICSD_cleaned/ICSD_1529.cif"
        )
"""
