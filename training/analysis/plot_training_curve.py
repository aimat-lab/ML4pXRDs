"""
This script can be used to plot a training curve as downloaded from Tensorboard.
"""

import ml4pxrd_tools.matplotlib_defaults
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import random
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


figure_double_width_pub = ml4pxrd_tools.matplotlib_defaults.pub_width

matplotlib.rcParams["lines.linewidth"] = 1


def plot_training_curve(
    files,
    labels,
    colors,
    linestyles,
    x_log=True,
    y_log=True,
    do_fix_step_continuity=False,
    start_epoch=0,
    plot_1_minus_acc=False,
):
    """Plot training curve from csv files (as downloaded from tensorboard).

    Args:
        files (list): List of csv files. This can also be a list of lists,
            where each sublist contains multiple csv files that belong to one run (concatenated)
        labels (list): Legend label for each item of `files`.
        colors (list): Color for each item of `files`.
        linestyles (list): Linestyle for each item of `files`.
        x_log (bool, optional): Use log scale for x axis. Defaults to False.
        y_log (bool, optional): Use log scale for y axis. Defaults to False.
        do_fix_step_continuity (bool, optional): If two parts of a run that should be concatenated start at epoch 0,
            this fixes the continuity of the epoch.
    """

    for i, file in enumerate(files):
        if isinstance(file, list):
            all_data = None

            current_step = 0
            for _subfile in file:
                if isinstance(_subfile, list):
                    subfile = _subfile[0]
                    prob = _subfile[1]
                else:
                    subfile = _subfile

                data = np.genfromtxt(subfile, skip_header=1, delimiter=",")[:, 1:3]

                if start_epoch != 0:
                    index = np.argmax(data[:, 0] >= start_epoch)
                    data = data[index:]

                # subsample with prob
                if isinstance(_subfile, list):
                    indices = []
                    for k in range(data.shape[0]):
                        if random.random() < prob:
                            indices.append(k)
                    data = data[indices, :]

                if do_fix_step_continuity:
                    data[:, 0] += current_step

                if all_data is None:
                    all_data = data
                else:
                    all_data = np.concatenate((all_data, data), axis=0)

                current_step = data[-1, 0]

        else:
            all_data = np.genfromtxt(file, skip_header=1, delimiter=",")[:, 1:3]

        x_data = all_data[:, 0]
        y_data = all_data[:, 1]
        plt.plot(
            x_data,
            1 - y_data if plot_1_minus_acc else y_data,
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
        )

    if x_log:
        plt.gca().set_xscale("log")

    if y_log:
        plt.gca().set_yscale("log")

    plt.ylim((0.0, 1.0))

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()


if __name__ == "__main__":
    plt.figure(
        figsize=(
            figure_double_width_pub * 0.7,
            figure_double_width_pub * 0.5,
        )
    )
    # plt.xlim([-500, 2000])
    plt.xlim([1, 2000])

    plt.ticklabel_format(style="plain")

    # Main training curve plot
    plot_training_curve(
        [
            "training_curves/resnet_10_training.csv",
            "training_curves/resnet_10_match.csv",
            "training_curves/resnet_10_top5.csv",
        ],
        [
            "ResNet-10 training",
            "ResNet-10 ICSD",
            "ResNet-10 ICSD top-5",
        ],
        ["r", "r", "r"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        x_log=True,
        y_log=True,
    )

    plot_training_curve(
        [
            [
                "training_curves/resnet_50_training_0.csv",
                ["training_curves/resnet_50_training_1.csv", 0.65263861055],
            ],
            [
                "training_curves/resnet_50_match_0.csv",
                ["training_curves/resnet_50_match_1.csv", 0.65263861055],
            ],
            [
                "training_curves/resnet_50_top5_0.csv",
                ["training_curves/resnet_50_top5_1.csv", 0.65263861055],
            ],
        ],
        [
            "ResNet-50 training",
            "ResNet-50 ICSD",
            "ResNet-50 ICSD top-5",
        ],
        ["g", "g", "g"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        x_log=True,
        y_log=True,
    )

    plot_training_curve(
        [
            [
                "training_curves/resnet_101_training_0.csv",
                ["training_curves/resnet_101_training_1.csv", 0.71439568899],
            ],
            [
                "training_curves/resnet_101_match_0.csv",
                ["training_curves/resnet_101_match_1.csv", 0.71439568899],
            ],
            [
                "training_curves/resnet_101_top5_0.csv",
                ["training_curves/resnet_101_top5_1.csv", 0.71439568899],
            ],
        ],
        [
            "ResNet-101 training",
            "ResNet-101 ICSD",
            "ResNet-101 ICSD top-5",
        ],
        ["b", "b", "b"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        x_log=True,
        y_log=True,
    )

    plt.gca().get_legend().remove()

    formatter = ticker.FuncFormatter(lambda x, pos: "{:.1f}".format(x))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_minor_formatter(formatter)
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    secax = plt.gca().secondary_xaxis(
        "top", functions=(lambda x: 130500 * x, lambda x: x / 130500)
    )
    secax.set_xlabel("Number of unique diffractograms")

    patches = [
        mpatches.Patch(color="b", label="ResNet-101"),
        mpatches.Patch(color="g", label="ResNet-50"),
        mpatches.Patch(color="r", label="ResNet-10"),
    ]
    plt.legend(handles=patches)

    plt.tight_layout()
    plt.savefig("training_curve_main.pdf", bbox_inches="tight")
    plt.show()

    """
    # Plots of 1-acc.
    plot_training_curve(
        [
            "training_curves/resnet_10_training.csv",
            "training_curves/resnet_10_match.csv",
            "training_curves/resnet_10_top5.csv",
        ],
        [
            "ResNet-10 training",
            "ResNet-10 ICSD",
            "ResNet-10 ICSD top-5",
        ],
        ["r", "r", "r"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        plot_1_minus_acc=True,
    )

    plot_training_curve(
        [
            [
                "training_curves/resnet_50_training_0.csv",
                ["training_curves/resnet_50_training_1.csv", 0.65263861055],
            ],
            [
                "training_curves/resnet_50_match_0.csv",
                ["training_curves/resnet_50_match_1.csv", 0.65263861055],
            ],
            [
                "training_curves/resnet_50_top5_0.csv",
                ["training_curves/resnet_50_top5_1.csv", 0.65263861055],
            ],
        ],
        [
            "ResNet-50 training",
            "ResNet-50 ICSD",
            "ResNet-50 ICSD top-5",
        ],
        ["g", "g", "g"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        plot_1_minus_acc=True,
    )

    plot_training_curve(
        [
            [
                "training_curves/resnet_101_match_0.csv",
                ["training_curves/resnet_101_match_1.csv", 0.71439568899],
            ],
            [
                "training_curves/resnet_101_training_0.csv",
                ["training_curves/resnet_101_training_1.csv", 0.71439568899],
            ],
            [
                "training_curves/resnet_101_top5_0.csv",
                ["training_curves/resnet_101_top5_1.csv", 0.71439568899],
            ],
        ],
        [
            "ResNet-101 ICSD",
            "ResNet-101 Train",
            "ResNet-101 ICSD top-5",
        ],
        ["r", "g", "b"],
        # ["solid", "--", (0, (1, 10))],
        ["solid", "solid", "solid"],
        start_epoch=5,
        plot_1_minus_acc=True,
    )

    # plt.gca().get_legend().remove()

    # patches = [
    #    mpatches.Patch(color="b", label="ResNet-101"),
    #    # mpatches.Patch(color="g", label="ResNet-50"),
    #    # mpatches.Patch(color="r", label="ResNet-10"),
    # ]
    # plt.legend(handles=patches)

    plt.ylabel("1 - acc.")
    plt.tight_layout()
    plt.savefig("training_curve_main_1_minus_acc.pdf", bbox_inches="tight")
    plt.show()

    """
