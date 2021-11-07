from simulation import Simulation
import xrayutilities as xu
import functools
import multiprocessing
import os


class ICSDSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/icsd/"

    def generate_structures(self):

        self.reset_simulation_status()

        counter = 0

        for i, path in enumerate(self.icsd_paths):
            # print(path)

            if (i % 1000) == 0:
                print(f"Generated {i} structures.", flush=True)

            if path is None:
                counter += 1
                continue

            try:
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
            self.sim_lines_list.append([])  # this will also be filled by the simulation

        print(f"Skipped {counter} structures due to errors or missing path.")


if __name__ == "__main__":
    # make print statement always flush
    print = functools.partial(print, flush=True)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is not None and jobid != "":
        simulation = ICSDSimulation(
            "/home/kit/iti/la2559/Databases/ICSD/ICSD_data_from_API.csv",
            "/home/kit/iti/la2559/Databases/ICSD/cif/",
        )
    else:
        simulation = ICSDSimulation(
            "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
            "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
        )
    
    if False:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()
    simulation.simulate_all(start_from_scratch=False)

    #simulation.plot(together=5)
