from simulation import Simulation
import xrayutilities as xu
import numpy as np
import os
import matplotlib.pyplot as plt


class ComparisonSimulation(Simulation):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/comparison/"

    def generate_structures(self):

        path = self.icsd_paths[self.icsd_ids.index(238381)]
        crystal = xu.materials.Crystal.fromCIF(path)
        self.crystals.append(crystal)
        self.labels.append(0)


if __name__ == "__main__":

    simulator = ComparisonSimulation(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    simulator.generate_structures()
    simulator.simulate_all()

    simulator.load_simulated_patterns()

    data = simulator.patterns

    xs_simulated = np.linspace(0, 90, 9001)

    ys_simulated1 = data[0, :]
    ys_simulated2 = data[1, :]
    ys_simulated3 = data[2, :]
    ys_simulated4 = data[3, :]
    ys_simulated5 = data[4, :]
    ys_simulated6 = data[5, :]

    plt.plot(
        xs_simulated, ys_simulated1 / np.max(ys_simulated1), label="gauss_max, lor_0"
    )
    plt.plot(
        xs_simulated, ys_simulated2 / np.max(ys_simulated2), label="gauss_0 lor_max"
    )
    plt.plot(
        xs_simulated, ys_simulated3 / np.max(ys_simulated3), label="gauss_max lor_max"
    )
    plt.plot(
        xs_simulated, ys_simulated4 / np.max(ys_simulated4), label="gauss_min, lor_0"
    )
    plt.plot(
        xs_simulated, ys_simulated5 / np.max(ys_simulated5), label="gauss_0, lor_min"
    )
    plt.plot(
        xs_simulated, ys_simulated6 / np.max(ys_simulated6), label="gauss_min, lor_min"
    )

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

