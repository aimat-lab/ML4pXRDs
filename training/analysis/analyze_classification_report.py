from concurrent.futures import process
from xml.etree.ElementInclude import include
from sklearn.metrics import classification_report
from ml4pxrd_tools.manage_dataset import load_dataset_info
import pickle
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":

    include_NO_samples = True
    do_plot_random = False

    if True:
        # path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/21-08-2022_12-40-17"  # 2k epochs random resnet-50
        path = "/home/henrik/Dokumente/Promotion/xrd_paper/ML4pXRDs/training/classifier_spgs/16-12-2022_08-37-44"  # direct training, big park model
    else:
        path = sys.argv[1]

    tag = "direct"

    with open(os.path.join(path, "classification_report_match.pickle"), "rb") as file:
        report_match = pickle.load(file)
    # print("Classification report match:")
    # print(report_match)

    with open(os.path.join(path, "classification_report_random.pickle"), "rb") as file:
        report_random = pickle.load(file)
    # print("Classification report random:")
    # print(report_random)

    def process_report(report):

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

    # Switch between possible metrics:
    metrics_random = recalls_random
    metrics_match = recalls_match
    metric_name = "recall"

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

    else:
        plt.legend()

    plt.savefig(f"{tag}.pdf")
    plt.show()

    # Plot difference between training (random) and match:

    plt.figure()
    hd0 = plt.scatter(
        spgs_match,
        np.array(metrics_random) - np.array(metrics_match),
        label="random - match",
    )
    hd1 = plt.plot(spgs_match, np.zeros(len(spgs_match)))
    # for x in [1, 3, 16, 75, 143, 168, 195]:
    #    plt.axvline(x, color="r")
    plt.xlabel("spg")
    plt.ylabel("delta " + metric_name)

    if include_NO_samples:
        ax2 = plt.gca().twinx()
        hd3 = ax2.plot(spgs_match, NO_samples, label="NO samples", color="r")
        ax2.set_ylabel("NO samples")

        plt.legend(handles=[hd0] + hd3)
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
