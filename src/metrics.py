"""Metrics used for the final evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calculate_metrics(probabilities, targets, class_names):
    probabilities = np.array(probabilities)
    targets = np.array(targets)
    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=range(len(class_names)),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )

    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }

    matrix = confusion_matrix(targets, predictions)
    ranked_classes = sorted(per_class.items(), key=lambda item: item[1]["f1"])

    confusion_counts = matrix.copy()
    np.fill_diagonal(confusion_counts, 0)
    common_confusions = []
    for position in np.argsort(confusion_counts, axis=None)[-3:][::-1]:
        true_index, predicted_index = np.unravel_index(position, confusion_counts.shape)
        common_confusions.append({
            "true_label": class_names[true_index],
            "predicted_label": class_names[predicted_index],
            "count": int(confusion_counts[true_index, predicted_index]),
        })

    return {
        "top1_accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_roc_auc_ovr": float(
            roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
        ),
        "per_class": per_class,
        "five_highest_f1": [dict(class_name=name, **values) for name, values in ranked_classes[-5:][::-1]],
        "five_lowest_f1": [dict(class_name=name, **values) for name, values in ranked_classes[:5]],
        "common_confusions": common_confusions,
        "confusion_matrix": matrix.tolist(),
    }
