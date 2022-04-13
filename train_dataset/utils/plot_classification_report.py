from concurrent.futures import process
from sklearn.metrics import classification_report
from dataset_simulations.random_simulation_utils import load_dataset_info
import pickle
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":

    if True:

        # report = classification_report(
        #    [1, 2, 3, 4, 5, 6, 10, 2, 5, 6, 3, 6, 7, 1, 2, 1, 3, 6, 4],
        #    [1, 2, 3, 4, 3, 5, 10, 3, 5, 6, 2, 4, 2, 1, 2, 3, 1, 6, 4],
        #    output_dict=True,
        # )

        path = "/home/henrik/Dokumente/Masterarbeit/HEOs_MSc/train_dataset/classifier_spgs/runs_from_cluster/continued_tests/07-04-2022_14-55-52"

    else:

        path = sys.argv[1]

    with open(os.path.join(path, "classification_report_match.pickle"), "rb") as file:
        report_match = pickle.load(file)

    with open(os.path.join(path, "classification_report_random.pickle"), "rb") as file:
        report_random = pickle.load(file)

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

        return [
            np.array(item)
            for item in zip(
                *sorted(zip(spgs, precisions, recalls, f1_scores), key=lambda x: x[0])
            )
        ]

    spgs_match, precisions_match, recalls_match, f1_scores_match = process_report(
        report_match
    )
    spgs_random, precisions_random, recalls_random, f1_scores_random = process_report(
        report_random
    )

    if False:
        plt.figure()
        plt.plot(spgs_match, f1_scores_match, label="Match")
        plt.plot(spgs_random, f1_scores_random, label="Random")
        plt.xlabel("spg")
        plt.ylabel("f1-score")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(spgs_match, f1_scores_random - f1_scores_match, label="random - match")
        plt.xlabel("spg")
        plt.ylabel("f1-score")
        plt.legend()
        plt.show()

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
    plt.plot(spgs_match, recalls_match, label="Match")
    plt.plot(spgs_random, recalls_random, label="Random")
    plt.plot(spgs_match, average_NO_wyckoffs, label="Average NO_wyckoffs")
    plt.plot(spgs_match, np.zeros(len(spgs_match)))
    for x in [1, 3, 16, 75, 143, 168, 195]:
        plt.axvline(x, color="r")
    plt.xlabel("spg")
    plt.ylabel("f1-score")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(spgs_match, recalls_random - recalls_match, label="random - match")
    plt.plot(spgs_match, average_NO_wyckoffs, label="Average NO_wyckoffs")
    plt.plot(spgs_match, np.zeros(len(spgs_match)))
    for x in [1, 3, 16, 75, 143, 168, 195]:
        plt.axvline(x, color="r")
    plt.xlabel("spg")
    plt.ylabel("delta f1-score")
    plt.legend()
    plt.show()
