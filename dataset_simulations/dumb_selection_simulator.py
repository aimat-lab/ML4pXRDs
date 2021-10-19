# At first:
# find common space groups in icsd (~20)
# select icsd entries from these space groups
# simulate

# Then:
# Try smarter selections:
# - only certain elements
# - only oxides, hydroxides, etc.
# - switch structure factors

"""
        crystal = xu.materials.Crystal.fromCIF(
            "/home/henrik/Dokumente/ICSD_cleaned/ICSD_1529.cif"
        )
"""

# Implement first: Narrow selection, only one structure per class, same selection as before!
# Train probabilistic network!? Train it with pure and then later also mixed samples.

from dataset_simulations.simulator import Simulator


class SmartSelectionSimulator(Simulator):
    def __init__():
        pass

    def generate_structures():
        pass

        # fill self.structures + self.labels
