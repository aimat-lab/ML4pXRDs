from simulation import Simulation
import xrayutilities as xu
import warnings


class ICSDSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/icsd/"

    def generate_structures(self, read_from_pickle=False, write_to_pickle=False):

        # warnings.filterwarnings("ignore", message="/used instead of/")
        warnings.filterwarnings("ignore")

        if read_from_pickle:
            self.load_crystals_pickle()
        else:

            counter = 0

            for i, path in enumerate(self.icsd_paths):

                if (i % 1000) == 0:
                    print(f"Generated {i} structures.")

                if path is None:
                    counter += 1
                    continue

                try:
                    xu.materials
                    crystal = xu.materials.Crystal.fromCIF(path)

                except Exception as ex:
                    counter += 1
                    continue

                # Problem! Multiple grain sizes
                self.crystals.append(crystal)
                self.labels.append([self.get_space_group_number(self.icsd_ids[i])])
                self.metas.append([self.icsd_ids[i]])

        if write_to_pickle:
            self.save_crystals_pickle()

        print(f"Skipped {counter} structures due to errors.")


if __name__ == "__main__":

    simulation = ICSDSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    simulation.generate_structures(read_from_pickle=False, write_to_pickle=True)

    # simulator.simulate_all()
