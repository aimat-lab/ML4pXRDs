from simulation import Simulation

# Build the crystals for the already known HEOs (Rock salt, spinel, cubic perovskite, ortho perovskite, fluorite, bixbyite, pyrochlore, (O3 layered)) (Maybe more?)
# Vary all open parameters (only lattice parameters?) of the structures
# Also vary: crystallite size


class PhaseSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

    def generate_structures(self):
        pass

        # fill self.structures + self.labels
