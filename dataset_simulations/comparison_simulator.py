from simulator import Simulator
import xrayutilities as xu
import numpy as np
import os
import matplotlib.pyplot as plt


class ComparisonSimulator(Simulator):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

        self.output_dir = "patterns/comparison/"

    def generate_structures(self):

        path = self.icsd_paths[self.icsd_ids.index(238381)]
        crystal = xu.materials.Crystal.fromCIF(path)
        self.crystals.append(crystal)
        self.labels.append(0)


if __name__ == "__main__":

    simulator = ComparisonSimulator(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    simulator.generate_structures()
    simulator.simulate_all()

    data = np.genfromtxt(
        os.path.join(simulator.output_dir, "dataset_0.csv"), delimiter=" "
    )

    xs_simulated = np.linspace(0, 90, 9001)

    ys_simulated1 = data[0, :-1]
    ys_simulated2 = data[1, :-1]
    ys_simulated3 = data[2, :-1]
    ys_simulated4 = data[3, :-1]
    ys_simulated5 = data[4, :-1]
    ys_simulated6 = data[5, :-1]

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

    plt.legend()

    data_compare_to = np.genfromtxt("comparison_Ce_O2.csv", autostrip=True)
    xs_compare_to = data_compare_to[:, 0]
    ys_compare_to = data_compare_to[:, 1]

    plt.plot(np.array(xs_compare_to) - 0.2, ys_compare_to / np.max(ys_compare_to))

    plt.show()

