# This script can be used to plot a training curve as downloaded from Tensorboard.

import ml4pxrd_tools.matplotlib_defaults
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(files, labels, colors, linestyles, x_log=True, y_log=False):

    for i, file in enumerate(files):

        if isinstance(file, list):

            all_data = None

            current_step = 0
            for subfile in file:

                data = np.genfromtxt(subfile, skip_header=1, delimiter=",")[:, 1:3]
                data[:, 0] += current_step

                if all_data is None:
                    all_data = data
                else:
                    all_data = np.concatenate((all_data, data), axis=0)

                current_step = data[-1, 0]

        else:
            all_data = np.genfromtxt(file, skip_header=1, delimiter=",")[:, 1:3]

        plt.plot(
            all_data[:, 0],
            all_data[:, 1],
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
        )

        if x_log:
            plt.gca().set_xscale("log")

        if y_log:
            plt.gca().set_yscale("log")

    plt.xlabel("steps")
    plt.ylabel("Accuracy")

    plt.legend()


if __name__ == "__main__":

    plot_training_curve(
        [
            "training_curves/resnet_50_random.csv",
            "training_curves/resnet_50_match.csv",
            "training_curves/resnet_50_top5.csv",
        ],
        ["ResNet-50 training", "ResNet-50 ICSD", "ResNet-50 ICSD top-5"],
        ["r", "g", "b"],
        ["solid", "solid", "solid"],
    )

    plot_training_curve(
        [
            "training_curves/resnet_101_random.csv",
            "training_curves/resnet_101_match.csv",
            "training_curves/resnet_101_top5.csv",
        ],
        ["ResNet-101 training", "ResNet-101 ICSD", "ResNet-101 ICSD top-5"],
        ["r", "g", "b"],
        ["dashed", "dashed", "dashed"],
    )

    plot_training_curve(
        [
            [
                "training_curves/resnet_50_full_random_0.csv",
                "training_curves/resnet_50_full_random_1.csv",
            ],
            [
                "training_curves/resnet_50_full_match_0.csv",
                "training_curves/resnet_50_full_match_1.csv",
            ],
            [
                "training_curves/resnet_50_full_top5_0.csv",
                "training_curves/resnet_50_full_top5_1.csv",
            ],
        ],
        [
            "ResNet-50 with sqrt-scaling training",
            "ResNet-50 with sqrt-scaling ICSD",
            "ResNet-50 with sqrt-scaling ICSD top-5",
        ],
        ["r", "g", "b"],
        [(0, (1, 10)), (0, (1, 10)), (0, (1, 10))],
    )

    plt.vlines([160, 1000, 2000], 0, 1, colors=["k", "k", "k"])
    plt.show()
