import numpy as np
from train_dataset.utils.test_unet.rruff_helpers import *
import pickle

xs, ys, dif_files, raw_files = get_rruff_patterns(
    only_refitted_patterns=False,
    only_selected_patterns=True,
    start_angle=5,
    end_angle=90,
    reduced_resolution=False,
    only_if_dif_exists=True,  # skips patterns where no dif is file
)

if True:
    xs = xs[0:40]  # TODO: Change back
    ys = ys[0:40]
    dif_files = dif_files[0:40]
    raw_files = raw_files[0:40]

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

parameter_results = []

for i, x in enumerate(xs):

    print(f"{(i+1)/len(xs)*100:.2f}% processed")

    data, wavelength, spg_number = dif_parser(dif_files[i])

    if data is not None:

        try:
            fit_parameters, score = fit_diffractogram(
                x,
                ys[i] / np.max(ys[i]),
                data[:, 0],  # angles
                data[:, 1] / np.max(data[:, 1]),  # intensities
                do_plot=False,
                only_plot_final=True,
            )
        except Exception as ex:
            print("Error fitting diffractogram to sample:")
            print(ex)
            continue

        if score > 0.9:
            parameter_results.append((raw_files[i], fit_parameters))

with open("rruff_refits.pickle", "wb") as file:
    pickle.dump(parameter_results, file)
