"""
Evaluation metrics for classification on clinical time series.

Wraps sklearn metrics with consistent handling for binary classification
from softmax probabilities.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_classification(probs, labels):
    """
    Compute standard classification metrics from probabilities.

    Args:
        probs: (N, 2) binary classification probabilities
        labels: (N,) integer labels in {0, 1}

    Returns:
        dict with accuracy, macro_f1, auroc
    """
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
    }
