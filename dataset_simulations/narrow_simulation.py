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

    def generate_structures(self):

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0

        for i, path in enumerate(self.icsd_paths):

            if (i % 1000) == 0:
                print(f"Processed {i} structures.")

            if self.icsd_structure_types[i] == "Fluorite#CaF2":
                label = 0
                counter_0 += 1
            elif self.icsd_structure_types[i] == "Bixbyite#(MnFe)O3":
                label = 1
                counter_1 += 1
            elif self.icsd_structure_types[i] == "UCl3":
                label = 2
                counter_2 += 1
            else:
                continue

            crystal = xu.materials.Crystal.fromCIF(path)

            self.sim_crystals.append(crystal)
            self.sim_labels.append([label])
            self.sim_metas.append([self.icsd_ids[i]])

            self.sim_patterns.append([])
            self.sim_variations.append([])

        print(f"Loaded {len(self.sim_crystals)} crystals")
        print(f"Fluorite#CaF2: {counter_0}")
        print(f"Bixbyite#(MnFe)O3: {counter_1}")
        print(f"UCl3: {counter_2}")


if __name__ == "__main__":

    simulation = NarrowSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    if True:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()
        simulation.save()

    simulation.simulate_all(start_from_scratch=False)
