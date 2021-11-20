from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


class ComparisonSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/comparison_pymatgen/"

    def generate_structures(self):

        path = self.icsd_paths[self.icsd_ids.index(238381)]
        self.add_path_to_be_simulated(path, [0], [238381])


if __name__ == "__main__":

    simulation = ComparisonSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    if False:  # toggle
        simulation.load()
    else:
        simulation.generate_structures()
        simulation.simulate_all(test_crystallite_sizes=True)
        simulation.load()

    data = np.array(simulation.sim_patterns[0])

    xs_simulated = np.linspace(0, 90, 9018)

    ys_simulated1 = data[0, :]
    ys_simulated2 = data[1, :]

    plt.plot(xs_simulated, ys_simulated1, label="gauss_min")
    plt.plot(xs_simulated, ys_simulated2, label="gauss_max")

    data_compare_to = np.genfromtxt("comparison_Ce_O2.csv", autostrip=True)
    xs_compare_to = data_compare_to[:, 0]
    ys_compare_to = data_compare_to[:, 1]
    plt.plot(
        np.array(xs_compare_to) - 0.2,
        ys_compare_to / np.max(ys_compare_to),
        label="exp. Ce O2",
    )

    plt.legend()

    plt.show()
