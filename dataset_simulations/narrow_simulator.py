from simulator import Simulator
import xrayutilities as xu
import pickle
import os

# Use a very narrow selection of ICSD entries

# Composition:
# Ce O2, Fm-3m (Fluorite)
# Y2 O3, bixbyite
# Lanthanum hydroxide La (O H)3, P63/m (not possible for HEOs)


class NarrowSimulator(Simulator):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/narrow/"

    def generate_structures(self, read_from_pickle=False, write_to_pickle=False):

        if read_from_pickle:
            self.load_crystals_and_labels()
            return

        for i, path in enumerate(self.icsd_paths):
            """
            if self.icsd_sumformulas[i] == "Ce1 O2":
                label = 0
            elif self.icsd_sumformulas[i] == "O3 Y2":
                label = 1
            elif self.icsd_formulas[i] == "La (O H)3":
                label = 2
            else:
                continue
            """

            if self.icsd_structure_types[i] == "Fluorite#CaF2":
                label = 0
            elif self.icsd_structure_types[i] == "Bixbyite#(MnFe)O3":
                label = 1
            elif self.icsd_structure_types[i] == "UCl3":
                label = 2
            else:
                continue

            crystal = xu.materials.Crystal.fromCIF(path)

            self.crystals.append(crystal)
            self.labels.append(label)

        if write_to_pickle:
            self.save_crystals_and_labels()


if __name__ == "__main__":
    simulator = NarrowSimulator(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    simulator.generate_structures(read_from_pickle=True, write_to_pickle=False)

    print(len(simulator.crystals))
    print(simulator.labels.count(0))
    print(simulator.labels.count(1))
    print(simulator.labels.count(2))

    simulator.simulate_all()
