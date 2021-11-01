from simulation import Simulation
import xrayutilities as xu

# make print statement always flush
print = functools.partial(print, flush=True)

class ICSDSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/icsd/"

    def generate_structures(self):

        self.reset_simulation_status()

        counter = 0

        for i, path in enumerate(self.icsd_paths):
            #print(path)

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

            self.sim_crystals.append(crystal)
            self.sim_labels.append([self.get_space_group_number(self.icsd_ids[i])])
            self.sim_metas.append([self.icsd_ids[i]])
            self.sim_variations.append(
                []
            )  # this will later be filled by the simulation, e.g. different corn sizes
            self.sim_patterns.append([])  # this will also be filled by the simulation

        print(f"Skipped {counter} structures due to errors.")


if __name__ == "__main__":

    simulation = ICSDSimulation(
        "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
        "/home/kit/iti/la2559/Databases/ICSD/cif/",
    )

    if True:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()
        simulation.save()

    simulation.simulate_all(start_from_scratch=True)
