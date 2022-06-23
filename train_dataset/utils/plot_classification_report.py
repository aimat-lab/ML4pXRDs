from concurrent.futures import process
from xml.etree.ElementInclude import include
from sklearn.metrics import classification_report
from dataset_simulations.random_simulation_utils import load_dataset_info
import pickle
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":

    include_NO_wyckoffs = False

    if True:

        # report = classification_report(
        #    [1, 2, 3, 4, 5, 6, 10, 2, 5, 6, 3, 6, 7, 1, 2, 1, 3, 6, 4],
        #    [1, 2, 3, 4, 3, 5, 10, 3, 5, 6, 2, 4, 2, 1, 2, 3, 1, 6, 4],
        #    output_dict=True,
        # )

        # path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/07-04-2022_14-55-52"  # 1-230
        # path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/09-04-2022_12-27-30"  # 100-230
        path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/07-06-2022_09-43-41"  # 50-230

    else:

        path = sys.argv[1]

    with open(os.path.join(path, "classification_report_match.pickle"), "rb") as file:
        report_match = pickle.load(file)
    print("Classification report match:")
    print(report_match)

    with open(os.path.join(path, "classification_report_random.pickle"), "rb") as file:
        report_random = pickle.load(file)
    print("Classification report random:")
    print(report_random)

    def process_report(report):

        spgs = []
        precisions = []
        recalls = []
        f1_scores = []

        for spg in report.keys():
            if not str.isnumeric(spg):
                continue

            precision = report[spg]["precision"]
            recall = report[spg]["recall"]
            f1_score = report[spg]["f1-score"]

            spgs.append(int(spg))
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        return spgs, precisions, recalls, f1_scores

    spgs_match, precisions_match, recalls_match, f1_scores_match = process_report(
        report_match
    )
    spgs_random, precisions_random, recalls_random, f1_scores_random = process_report(
        report_random
    )

    assert spgs_match == spgs_random

    # Switch between possible metrics:
    metrics_random = f1_scores_random
    metrics_match = f1_scores_match
    metric_name = "f1_scores"

    if include_NO_wyckoffs:
        (
            probability_per_spg_per_element,
            probability_per_spg_per_element_per_wyckoff,
            NO_wyckoffs_prob_per_spg,
            corrected_labels,
            files_to_use_for_test_set,
            represented_spgs,
            NO_unique_elements_prob_per_spg,
            NO_repetitions_prob_per_spg_per_element,
            denseness_factors_density_per_spg,
            kde_per_spg,
            all_data_per_spg_tmp,
        ) = load_dataset_info()

        average_NO_wyckoffs = []

        for spg in spgs_match:
            NO_wyckoffs_prob = NO_wyckoffs_prob_per_spg[spg]

            average = 0

            for i, NO_wyckoff in enumerate(range(1, len(NO_wyckoffs_prob) + 1)):
                average += NO_wyckoff * NO_wyckoffs_prob[i]

            average_NO_wyckoffs.append(average)

    plt.figure()
    hd0 = plt.plot(spgs_match, metrics_match, label="Match")
    hd1 = plt.plot(spgs_random, metrics_random, label="Random")
    hd2 = plt.plot(spgs_match, np.zeros(len(spgs_match)))
    for x in [1, 3, 16, 75, 143, 168, 195]:
        plt.axvline(x, color="r")
    plt.xlabel("spg")
    plt.ylabel(metric_name)

    if include_NO_wyckoffs:
        ax2 = plt.gca().twinx()
        hd3 = ax2.plot(
            spgs_match, average_NO_wyckoffs, label="Average NO_wyckoffs", color="r"
        )
        ax2.set_ylabel("average NO_wyckoffs")
        plt.legend(handles=hd0 + hd1 + hd3)
    else:
        plt.legend()
    plt.show()

    plt.figure()
    hd0 = plt.plot(
        spgs_match,
        np.array(metrics_random) - np.array(metrics_match),
        label="random - match",
    )
    hd1 = plt.plot(spgs_match, np.zeros(len(spgs_match)))
    for x in [1, 3, 16, 75, 143, 168, 195]:
        plt.axvline(x, color="r")
    plt.xlabel("spg")
    plt.ylabel("delta " + metric_name)

    if include_NO_wyckoffs:
        ax2 = plt.gca().twinx()
        hd2 = ax2.plot(
            spgs_match, average_NO_wyckoffs, label="Average NO_wyckoffs", color="r"
        )
        ax2.set_ylabel("average NO_wyckoffs")
        plt.legend(handles=hd0 + hd2)
    else:
        plt.legend()
    plt.show()

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

        total_per_crystal_system[crystal_system] += 1
        metrics_per_crystal_system[crystal_system] += metrics_match[i]

    for key in metrics_per_crystal_system.keys():
        metrics_per_crystal_system[key] /= total_per_crystal_system[key]

    print(metrics_per_crystal_system)

    # TODO: Maybe weight by number of samples
