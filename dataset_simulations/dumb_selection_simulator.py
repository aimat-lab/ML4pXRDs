# "Dumb" selection of structures: Just use all structures of the corresponding space group and use them for training
# This won't be able to distinguish between Rocksalt / Fluorite (Fm-3m) and also Spinel / Pyrochlore (Fd-3m)

from simulator import Simulator
import pandas as pd


class DumbSelectionSimulator(Simulator):
    def __init__(self, icsd_info_file_path, icsd_cifs_dir):
        super().__init__(icsd_info_file_path, icsd_cifs_dir)

    def fillStructures(self):
        for space_group in self.icsd_space_group_symbols:
            if space_group == "F m -3 m":  # 0
                pass
            elif space_group == "I a -3":  # 1
                pass
            elif space_group == "P 63/m":  # 2
                pass


if __name__ == "__main__":
    simulator = DumbSelectionSimulator(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        "/home/henrik/Dokumente/Big_Files/ICSD/cif/",
    )

    pass
