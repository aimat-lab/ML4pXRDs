import numpy as np
from train_dataset.utils.test_unet.rruff_helpers import *
import pickle
import multiprocessing


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
            print("Error fitting diffractogram to sample:")
            print(ex)
            return None

        return (raw_file, fit_parameters, score)
    else:
        return None


if __name__ == "__main__":

    xs, ys, dif_files, raw_files = get_rruff_patterns(
        only_refitted_patterns=False,
        only_selected_patterns=True,
        start_angle=5,
        end_angle=90,
        reduced_resolution=False,
        only_if_dif_exists=True,  # skips patterns where no dif is file
    )

    if True:  # TODO: Change back
        xs = xs[0:8]
        ys = ys[0:8]
        dif_files = dif_files[0:8]
        raw_files = raw_files[0:8]

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

    pool = multiprocessing.Pool(processes=8)  # TODO: Change back

    map_results = pool.map(process_pattern, zip(xs, ys, dif_files, raw_files))
    results = [item for item in map_results if (item is not None and item[2] > 0.9)]

    # print(results)

    with open("rruff_refits.pickle", "wb") as file:
        pickle.dump(results, file)

    pool.close()
