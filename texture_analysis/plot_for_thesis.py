import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib_defaults

figure_double_width_pub = matplotlib_defaults.pub_width

if __name__ == "__main__":

    data_file_path = "data/XRD_6_component_systems_repeat.csv"
    df = pd.read_csv(data_file_path, sep=";")

    x = np.array(df.iloc[:, 0])
    xs = np.repeat(x[:, np.newaxis], len(df.columns.values) - 1, axis=1)
    ys = np.array(df.iloc[:, list(range(1, len(df.columns.values)))])

    names = df.columns.values[1::]

    i_to_plot = 88

    print(names[i_to_plot])

    plt.figure(
        figsize=(
            figure_double_width_pub * 0.95,
            figure_double_width_pub * 0.7,
        )
    )
    plt.plot(xs[:, i_to_plot], ys[:, i_to_plot] / np.max(ys[:, i_to_plot]))
    plt.xlabel(r"$2\theta$")
    plt.ylabel(r"Intensity / rel.")
    plt.tight_layout()
    plt.savefig(
        "example_xrd_thesis.pdf",
        bbox_inches="tight",
    )
    plt.show()
