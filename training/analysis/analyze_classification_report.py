""" 
This script can be used to analyze the classification reports generated and saved by 
the training script. It will plot the metric specified in `metric_name` (see below in the script)
over the space groups for both the synthetic and ICSD (match) test dataset.
A second plot will contain the gap between the synthetic and ICSD (match) test dataset.

You can call the script like this:

```
python analyze_classification_report.py <path_to_run_directory> <tag>
```

The tag will be included in the filename of the saved figures.
"""

from ml4pxrd_tools.manage_dataset import load_dataset_info
import pickle
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import ml4pxrd_tools.matplotlib_defaults

if __name__ == "__main__":

    include_NO_samples = False  # Whether or not to include the number of samples in the statistics dataset of the given space group in the plots.
    sort_by_NO_samples = False  # Whether or not to sort the plot containing the gap between synthetic and match by NO_samples.
    do_plot_random = (
        True  # Whether or not to include the synthetic dataset in the first plot.
    )
    metric_name = "recall"  # "precision", "recall", or "f1"

    if False:
        path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/21-08-2022_12-40-17"  # 2k epochs random resnet-50
        # path = "/home/henrik/Dokumente/Promotion/xrd_paper/ML4pXRDs/training/classifier_spgs/16-12-2022_08-37-44"  # direct training, big park model
    else:
        path = sys.argv[1]

    if len(sys.argv) > 2:
        tag = sys.argv[2]
    else:
        tag = "synthetic"

    with open(os.path.join(path, "classification_report_match.pickle"), "rb") as file:
        report_match = pickle.load(file)
    # print("Classification report match:")
    # print(report_match)

    with open(os.path.join(path, "classification_report_random.pickle"), "rb") as file:
        report_random = pickle.load(file)
    # print("Classification report random:")
    # print(report_random)

    def process_report(report):
        """Process a classification report as returned by
        sklearn.metrics.classification_report with output_dict=True.

        Args:
            report: Classification report (dict)

        Returns:
            tuple: (spgs, precisions, recalls, f1_scores, supports)
        """

        spgs = []
        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        for spg in report.keys():
            if not str.isnumeric(spg):
                continue

            precision = report[spg]["precision"]
            recall = report[spg]["recall"]
            f1_score = report[spg]["f1-score"]
            support = report[spg]["support"]

            spgs.append(int(spg))
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            supports.append(support)

        return spgs, precisions, recalls, f1_scores, supports

    (
        spgs_match,
        precisions_match,
        recalls_match,
        f1_scores_match,
        supports_match,
    ) = process_report(report_match)

    (
        spgs_random,
        precisions_random,
        recalls_random,
        f1_scores_random,
        supports_random,
    ) = process_report(report_random)

    assert spgs_match == spgs_random

    if metric_name == "recall":
        metrics_random = recalls_random
        metrics_match = recalls_match
    elif metric_name == "precision":
        metrics_random = precisions_random
        metrics_match = precisions_match
    elif metric_name == "f1":
        metrics_random = f1_scores_random
        metrics_match = f1_scores_match
    else:
        raise Exception(f"Metric name {metric_name} not recognized.")

    if include_NO_samples:
        (
            (
                probability_per_spg_per_element,
                probability_per_spg_per_element_per_wyckoff,
                NO_unique_elements_prob_per_spg,
                NO_repetitions_prob_per_spg_per_element,
                denseness_factors_density_per_spg,
                denseness_factors_conditional_sampler_seeds_per_spg,
                lattice_paras_density_per_lattice_type,
                per_element,
                represented_spgs,
                probability_per_spg,
            ),
            (
                statistics_metas,
                statistics_labels,
                statistics_crystals,
                statistics_match_metas,
                statistics_match_labels,
                test_metas,
                test_labels,
                test_crystals,
                corrected_labels,
                test_match_metas,
                test_match_pure_metas,
            ),
        ) = load_dataset_info()

        NO_samples = []

        statistics_match_labels_flat = np.array(
            [item[0] for item in statistics_match_labels]
        )
        for spg in spgs_random:
            NO_samples.append(np.sum(statistics_match_labels_flat == spg))

    # Plot absolute values:

    plt.figure()
    hd0 = plt.scatter(spgs_match, metrics_match, label="Match")
    if do_plot_random:
        hd1 = plt.scatter(spgs_random, metrics_random, label="Random")
    hd2 = plt.plot(spgs_match, np.zeros(len(spgs_match)))

    # for x in [1, 3, 16, 75, 143, 168, 195]:
    #    plt.axvline(x, color="r")

    plt.xlabel("spg")
    plt.ylabel(metric_name)

    if include_NO_samples:
        ax2 = plt.gca().twinx()
        hd3 = ax2.plot(spgs_match, NO_samples, label="NO samples", color="r")
        ax2.set_ylabel("NO samples")

        if do_plot_random:
            plt.legend(handles=[hd0, hd1] + hd3)
        else:
            plt.legend(handles=[hd0] + hd3)

        print(
            "Correlation between match metric and NO_samples in the spg:",
            np.corrcoef(metrics_match, NO_samples),
        )

        print(
            "Correlation between random metric and NO_samples in the spg:",
            np.corrcoef(metrics_random, NO_samples),
        )

    else:
        plt.legend()

    plt.savefig(f"{tag}.pdf")
    plt.show()

    # Plot difference between training (random) and match:

    diff = np.array(metrics_random) - np.array(metrics_match)

    if sort_by_NO_samples:
        NO_samples = np.array(NO_samples)
        sorting_indices = np.argsort(NO_samples)
        NO_samples = NO_samples[sorting_indices]
        diff = diff[sorting_indices]
        spgs_match = list(range(len(diff)))

    plt.figure()
    hd0 = plt.bar(
        spgs_match,
        diff,
        label="random - match",
    )
    hd1 = plt.plot(spgs_match, np.zeros(len(spgs_match)))
    # for x in [1, 3, 16, 75, 143, 168, 195]:
    #    plt.axvline(x, color="r")

    if not sort_by_NO_samples:
        plt.xlabel("spg")
    else:
        plt.xlabel("index")

    plt.ylabel("delta " + metric_name)

    if include_NO_samples:
        ax2 = plt.gca().twinx()
        hd3 = ax2.plot(spgs_match, NO_samples, label="NO samples", color="r")
        ax2.set_ylabel("NO samples")

        plt.legend(handles=[hd0] + hd3)

        print(
            "Correlation between (random metric - match metric) and NO_samples in the spg:",
            np.corrcoef(diff, NO_samples),
        )

    else:
        plt.legend()

    plt.savefig(f"{tag}_diff.pdf")
    plt.show()

    """
    metrics_per_crystal_system = {
        "triclinic": 0,
        "monoclinic": 0,
        "orthorhombic": 0,
        "tetragonal": 0,
        "trigonal": 0,
        "hexagonal": 0,
        "cubic": 0,
    }
    total_per_crystal_system = {
        "triclinic": 0,
        "monoclinic": 0,
        "orthorhombic": 0,
        "tetragonal": 0,
        "trigonal": 0,
        "hexagonal": 0,
        "cubic": 0,
    }

    for i, spg in enumerate(spgs_match):
        if spg < 3:
            crystal_system = "triclinic"
        elif spg < 16:
            crystal_system = "monoclinic"
        elif spg < 75:
            crystal_system = "orthorhombic"
        elif spg < 143:
            crystal_system = "tetragonal"
        elif spg < 168:
            crystal_system = "trigonal"
        elif spg < 195:
            crystal_system = "hexagonal"
        else:
            crystal_system = "cubic"

        total_per_crystal_system[crystal_system] += supports_match[i]
        metrics_per_crystal_system[crystal_system] += (
            metrics_match[i] * supports_match[i]
        )

    for key in metrics_per_crystal_system.keys():
        metrics_per_crystal_system[key] /= total_per_crystal_system[key]

    print(metrics_per_crystal_system)
    """
