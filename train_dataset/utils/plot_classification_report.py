from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

report = classification_report(
    [1, 2, 3, 4, 5, 6, 7, 2, 5, 6, 3, 6, 7, 1, 2, 1, 3, 6, 4],
    [1, 2, 3, 4, 3, 5, 7, 3, 5, 6, 2, 4, 2, 1, 2, 3, 1, 6, 4],
    output_dict=True,
)

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

    spgs.append(spg)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

print()
