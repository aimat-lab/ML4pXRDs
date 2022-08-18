import numpy as np
from train_dataset.utils.test_unet.rruff_helpers import *
import pickle
import ray
from ray.util.multiprocessing import Pool
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
                do_plot=True,
                only_plot_final=True,
                do_print=False,
                save_index=0,
            )
        except Exception as ex:
            print("Error fitting diffractogram to sample:", flush=True)
            print(ex)
            return None

        return (raw_file, fit_parameters, score)
    else:
        return None


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

    if False:
        for i in range(len(raw_files)):
            if (
                raw_files[i]
                # == "../../RRUFF_data/XY_RAW/FergusoniteYbeta__R080103-1__Powder__Xray_Data_XY_RAW__9738.txt"
                == "../../RRUFF_data/XY_RAW/Cookeite__R061103-1__Powder__Xray_Data_XY_RAW__7829.txt"
            ):
                raw_files = raw_files[i : i + 1]
                xs = xs[i : i + 1]
                ys = ys[i : i + 1]
                dif_files = dif_files[i : i + 1]
                break
        process_pattern((xs[0], ys[0], dif_files[0], raw_files[0]))
        exit()

    if False:
        for i in range(len(raw_files)):
            if (
                "Petalite__R060365-1__Powder__Xray_Data_XY_RAW__2845.txt"
                in raw_files[i]
                or "Fluorite__R050115-1__Powder__Xray_Data_XY_RAW__916.txt"
                in raw_files[i]
                or "Hauyne__R050606-1__Powder__Xray_Data_XY_RAW__1990.txt"
                in raw_files[i]
                or "Cuprite__R050374-1__Powder__Xray_Data_XY_RAW__1594.txt"
                in raw_files[i]
            ):
                process_pattern((xs[i], ys[i], dif_files[i], raw_files[i]))
        exit()

    ray.init()

    chunk_size = 100
    N_workers = 32
    N_chunks = int(np.ceil(len(xs) / chunk_size))

    for j in range(0, N_chunks):

        print(f"Processing chunk {j+1} of {N_chunks}")

        with Pool(processes=N_workers) as pool:

            map_results = pool.map_async(
                process_pattern,
                zip(
                    xs[j * chunk_size : (j + 1) * chunk_size],
                    ys[j * chunk_size : (j + 1) * chunk_size],
                    dif_files[j * chunk_size : (j + 1) * chunk_size],
                    raw_files[j * chunk_size : (j + 1) * chunk_size],
                ),
            )

            map_results = map_results.get()
            results = [item for item in map_results if item is not None]

        with open(f"parameters/rruff_refits_{j}.pickle", "wb") as file:
            pickle.dump(results, file)
