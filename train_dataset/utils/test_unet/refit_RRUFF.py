import numpy as np
from train_dataset.utils.test_unet.rruff_helpers import *
import pickle
import multiprocessing
import time
import os


def process_pattern(input):

    x, y, dif_file, raw_file = input

    data, wavelength, spg_number = dif_parser(dif_file)

    if data is not None:
        try:
            fit_parameters, score = fit_diffractogram(
                x,
                y / np.max(y),
                data[:, 0],  # angles
                data[:, 1] / np.max(data[:, 1]),  # intensities
                do_plot=False,
                only_plot_final=True,
                do_print=False,
            )
        except Exception as ex:
            print("Error fitting diffractogram to sample:", flush=True)
            print(ex)
            return None

        return (raw_file, fit_parameters, score)
    else:
        return None


class PoolProgress:
    def __init__(self, pool, update_interval=3):
        self.pool = pool
        self.update_interval = update_interval

    def track(self, job):
        task = self.pool._cache[job._job]
        while task._number_left > 0:
            print(
                "Tasks remaining = {0}".format(task._number_left * task._chunksize),
                flush=True,
            )
            time.sleep(self.update_interval)


if __name__ == "__main__":

    os.system("mkdir -p parameters")

    xs, ys, dif_files, raw_files = get_rruff_patterns(
        only_refitted_patterns=False,
        only_selected_patterns=True,
        start_angle=5,
        end_angle=90,
        reduced_resolution=False,
        only_if_dif_exists=True,  # skips patterns where no dif is file
    )

    if False:
        xs = xs[0:32]
        ys = ys[0:32]
        dif_files = dif_files[0:32]
        raw_files = raw_files[0:32]

    """ Example with bump in the beginning
    for i in range(len(raw_files)):
        if (
            raw_files[i]
            == "../../RRUFF_data/XY_RAW/FergusoniteYbeta__R080103-1__Powder__Xray_Data_XY_RAW__9738.txt"
        ):
            raw_files = raw_files[i : i + 1]
            xs = xs[i : i + 1]
            ys = ys[i : i + 1]
            dif_files = dif_files[i : i + 1]
            break
    print(raw_files)
    """

    # process_pattern((xs[0], ys[0], dif_files[0], raw_files[0]))
    # exit()

    for j in range(0, int(np.ceil(len(xs) / 200))):

        with multiprocessing.Pool(processes=32) as pool:

            progress = PoolProgress(pool, update_interval=30)

            map_results = pool.map_async(
                process_pattern,
                zip(
                    xs[j * 200 : (j + 1) * 200],
                    ys[j * 200 : (j + 1) * 200],
                    dif_files[j * 200 : (j + 1) * 200],
                    raw_files[j * 200 : (j + 1) * 200],
                ),
            )

            progress.track(map_results)

            map_results = map_results.get()
            results = [
                item for item in map_results if (item is not None and item[2] > 0.9)
            ]

        with open(f"parameters/rruff_refits_{j}.pickle", "wb") as file:
            pickle.dump(results, file)
