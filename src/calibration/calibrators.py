"""
Probability calibration methods for R* estimation.

Implements three calibration algorithms, all with 5-fold cross-validation
to avoid using the test set for fitting:
- Isotonic regression (non-parametric, monotonic)
- Temperature scaling (single-parameter sigmoid)
- Platt scaling (logistic regression on logits)

References:
    Zadrozny & Elkan (2002). Transforming classifier scores into accurate
    multiclass probability estimates. KDD.
    Guo et al. (2017). On calibration of modern neural networks. ICML.
    Ushio, Ishida, Sugiyama (2026). Practical estimation of the optimal
    classification error with soft labels and calibration. ICLR.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def isotonic_calibrate_cv(probs, labels, n_classes, n_splits=5):
    """
    Isotonic regression calibration with cross-validation.

    For each fold, fit isotonic regression on the training portion and
    apply to the validation portion. Final probabilities are renormalized
    to sum to 1 per sample.

    Args:
        probs: (N, C) raw softmax probabilities
        labels: (N,) integer class labels
        n_classes: number of classes C
        n_splits: number of CV folds

    Returns:
        calibrated: (N, C) calibrated probabilities
    """
    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(probs, labels):
        for c in range(n_classes):
            y_bin = (labels[train_idx] == c).astype(float)
            ir = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip"
            )
            ir.fit(probs[train_idx, c], y_bin)
            calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.maximum(row_sums, 1e-10)


def temperature_scale_cv(probs, labels, n_splits=5):
    """
    Temperature scaling with cross-validation.

    Uses grid search over T in [0.1, 5.0] with step 0.05 to minimize
    cross-entropy on each training fold.

    Args:
        probs: (N, C) raw softmax probabilities
        labels: (N,) integer class labels
        n_splits: number of CV folds

    Returns:
        calibrated: (N, C) calibrated probabilities
    """
    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(probs, labels):
        logits = np.log(np.clip(probs[train_idx], 1e-10, 1.0))
        best_T, best_loss = 1.0, float("inf")
        for T in np.arange(0.1, 5.0, 0.05):
            scaled = np.exp(logits / T)
            scaled = scaled / scaled.sum(axis=1, keepdims=True)
            loss = log_loss(
                labels[train_idx], scaled, labels=list(range(probs.shape[1]))
            )
            if loss < best_loss:
                best_T, best_loss = T, loss
        logits_val = np.log(np.clip(probs[val_idx], 1e-10, 1.0))
        scaled_val = np.exp(logits_val / best_T)
        calibrated[val_idx] = scaled_val / scaled_val.sum(axis=1, keepdims=True)
    return calibrated


def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE).

    Partitions the confidence range into n_bins and computes the weighted
    absolute difference between accuracy and confidence in each bin.

    Args:
        probs: (N, C) probabilities
        labels: (N,) integer labels
        n_bins: number of bins

    Returns:
        ece: scalar ECE value
    """
    confidences = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    correct = (pred_labels == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(
                confidences[mask].mean() - correct[mask].mean()
            )
    return ece / len(labels)
