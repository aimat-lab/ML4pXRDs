from sklearn.metrics import classification_report
import pickle
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if True:

        report = classification_report(
            [1, 2, 3, 4, 5, 6, 7, 2, 5, 6, 3, 6, 7, 1, 2, 1, 3, 6, 4],
            [1, 2, 3, 4, 3, 5, 7, 3, 5, 6, 2, 4, 2, 1, 2, 3, 1, 6, 4],
            output_dict=True,
        )

    else:

        with open(sys.argv[1], "rb") as file:
            report = pickle.load(file)

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

    plt.figure()
    plt.plot(spgs, precisions, label="precision")
    plt.plot(spgs, recalls, label="recall")
    plt.plot(spgs, f1_scores, label="f1-score")
    plt.gca().set_xticks(list(range(min(spgs), max(spgs) + 1)))
    plt.legend()
    plt.show()
