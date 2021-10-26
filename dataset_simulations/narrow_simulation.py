from simulation import Simulation
import xrayutilities as xu

# Use a very narrow selection of ICSD entries

# Composition:
# Ce O2, Fm-3m (Fluorite)
# Y2 O3, bixbyite
# Lanthanum hydroxide La (O H)3, P63/m (not possible for HEOs)


class NarrowSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/narrow/"

    def generate_structures(self, read_from_pickle=False, write_to_pickle=False):

        if read_from_pickle:
            self.load_crystals_pickle()
        else:

            for i, path in enumerate(self.icsd_paths):

                if self.icsd_structure_types[i] == "Fluorite#CaF2":
                    labels = [0]
                elif self.icsd_structure_types[i] == "Bixbyite#(MnFe)O3":
                    labels = [1]
                elif self.icsd_structure_types[i] == "UCl3":
                    labels = [2]
                else:
                    continue

                crystal = xu.materials.Crystal.fromCIF(path)

                self.crystals.append(crystal)
                self.labels.append(labels)
                self.metas.append([self.icsd_ids[i]])

        if write_to_pickle:
            self.save_crystals_pickle()

        print(f"Loaded {len(self.crystals)} crystals")
        print(f"Fluorite#CaF2: {self.labels.count(0)}")
        print(f"Bixbyite#(MnFe)O3: {self.labels.count(1)}")
        print(f"UCl3: {self.labels.count(2)}")


if __name__ == "__main__":

    simulation = NarrowSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    simulation.generate_structures(read_from_pickle=True, write_to_pickle=False)

    simulation.simulate_all()
