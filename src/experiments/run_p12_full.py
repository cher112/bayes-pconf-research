#!/usr/bin/env python3
"""
Run 6 baseline models on PhysioNet 2012 (P12).
100 epochs each, sequential single-GPU execution.
Saves softmax probabilities, computes R* with isotonic/temperature calibration.
"""
import sys
import os
import json
import time
import traceback
import numpy as np
from datetime import datetime

sys.path.insert(0, "/root/autodl-tmp/IrregularlySampledTimeSeriesLibrary")

# ---- Monkey-patch pypots device issue ----
import pypots.base

_orig_setup = pypots.base.BaseModel._setup_device


def _patched_setup(self, d):
    if isinstance(d, str) and d.lower() == "best":
        d = None
    _orig_setup(self, d)


pypots.base.BaseModel._setup_device = _patched_setup

from benchpots.datasets import preprocess_physionet2012
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


# ---- Calibration utilities ----


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
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


def isotonic_calibrate_cv(probs, labels, n_classes, n_splits=5):
    """Cross-validated isotonic regression calibration."""
    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(probs, labels):
        for c in range(n_classes):
            y_bin = (labels[train_idx] == c).astype(float)
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            ir.fit(probs[train_idx, c], y_bin)
            calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.maximum(row_sums, 1e-10)


def temperature_scale_cv(probs, labels, n_splits=5):
    """Cross-validated temperature scaling via grid search."""
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


def compute_rstar_binary(eta):
    """R* = 0.5 - 0.5 * mean(|2*eta - 1|) for binary case."""
    return 0.5 - 0.5 * np.mean(np.abs(2 * eta - 1))


def evaluate_and_calibrate(probs, labels, n_classes):
    """Full evaluation: metrics + calibration + R* estimation."""
    results = {}

    # Classification metrics
    pred_labels = np.argmax(probs, axis=1)
    results["accuracy"] = float(accuracy_score(labels, pred_labels))
    results["macro_f1"] = float(f1_score(labels, pred_labels, average="macro"))
    if n_classes == 2:
        results["auroc"] = float(roc_auc_score(labels, probs[:, 1]))

    # ECE (raw)
    results["ece_raw"] = float(compute_ece(probs, labels))

    # Isotonic calibration
    calib_iso = isotonic_calibrate_cv(probs, labels, n_classes)
    results["ece_isotonic"] = float(compute_ece(calib_iso, labels))

    # Temperature scaling
    calib_temp = temperature_scale_cv(probs, labels)
    results["ece_temperature"] = float(compute_ece(calib_temp, labels))

    # R* estimation for each calibration method
    for tag, cal_probs in [
        ("raw", probs),
        ("isotonic", calib_iso),
        ("temperature", calib_temp),
    ]:
        if n_classes == 2:
            eta = cal_probs[:, 1]
            rstar = compute_rstar_binary(eta)
        else:
            rstar = float(np.mean(1 - np.max(cal_probs, axis=1)))
        results[f"rstar_{tag}"] = float(rstar)
        results[f"bayes_ceiling_{tag}"] = float(1 - rstar)

    return results


# ---- Model configs ----

MODELS = {
    "GRUD": {"rnn_hidden_size": 128},
    "BRITS": {"rnn_hidden_size": 128},
    "iTransformer": {
        "n_layers": 2,
        "d_model": 64,
        "d_ffn": 128,
        "n_heads": 4,
        "d_k": 16,
        "d_v": 16,
        "dropout": 0.1,
    },
    "SAITS": {
        "n_layers": 2,
        "d_model": 64,
        "d_ffn": 128,
        "n_heads": 4,
        "d_k": 16,
        "d_v": 16,
        "dropout": 0.1,
    },
    "TimesNet": {
        "n_layers": 2,
        "d_model": 64,
        "d_ffn": 128,
        "top_k": 3,
        "n_kernels": 3,
    },
    "SeFT": {
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "d_ffn": 128,
    },
}

OUTPUT_DIR = "/root/bayes-pconf-research/experiments"


def run_single_model(model_name, data, epochs, batch_size, patience):
    """Train one model, predict, calibrate, save probs + metrics."""
    from pypots.classification import (
        GRUD,
        BRITS,
        iTransformer,
        SAITS,
        TimesNet,
        SeFT,
    )

    model_map = {
        "GRUD": GRUD,
        "BRITS": BRITS,
        "iTransformer": iTransformer,
        "SAITS": SAITS,
        "TimesNet": TimesNet,
        "SeFT": SeFT,
    }

    ns = data["n_steps"]
    nf = data["n_features"]
    nc = data["n_classes"]

    common = {
        "n_steps": ns,
        "n_features": nf,
        "n_classes": nc,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "saving_path": None,  # No checkpoint saving to conserve disk
    }

    kwargs = {**MODELS[model_name], **common}
    model_cls = model_map[model_name]

    print(f"\n[{model_name}] Creating model (steps={ns}, features={nf}, classes={nc})")
    model = model_cls(**kwargs)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[{model_name}] Parameters: {n_params:,}")

    # Train
    t0 = time.time()
    model.fit(
        {"X": data["train_X"], "y": data["train_y"]},
        {"X": data["val_X"], "y": data["val_y"]},
    )
    train_time = time.time() - t0
    print(f"[{model_name}] Training time: {train_time:.1f}s")

    # Predict
    results = model.predict({"X": data["test_X"]})
    probs = results.get("classification_proba")
    if probs is None:
        print(f"[{model_name}] WARNING: no proba output, constructing from labels")
        preds = results["classification"].ravel().astype(int)
        probs = np.full((len(preds), nc), 0.1 / nc)
        for i, p in enumerate(preds):
            probs[i, p] = 0.9

    test_labels = data["test_y"]

    # Save probs
    probs_path = os.path.join(OUTPUT_DIR, f"{model_name}_probs.npy")
    np.save(probs_path, probs)
    print(f"[{model_name}] Saved probs to {probs_path}")

    # Calibrate + R*
    metrics = evaluate_and_calibrate(probs, test_labels, nc)
    metrics["model"] = model_name
    metrics["dataset"] = "p12"
    metrics["epochs"] = epochs
    metrics["n_params"] = n_params
    metrics["train_time_s"] = round(train_time, 1)
    metrics["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_name}] Saved metrics to {metrics_path}")

    return metrics


def print_metrics(m):
    """Pretty-print one model's metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {m['model']} on P12  ({m['n_params']:,} params, {m['train_time_s']}s)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:      {m['accuracy']:.4f}")
    print(f"  Macro-F1:      {m['macro_f1']:.4f}")
    if "auroc" in m:
        print(f"  AUROC:         {m['auroc']:.4f}")
    print(f"  ECE (raw):     {m['ece_raw']:.4f}")
    print(f"  ECE (iso):     {m['ece_isotonic']:.4f}")
    print(f"  ECE (temp):    {m['ece_temperature']:.4f}")
    print(f"  R* (raw):      {m['rstar_raw']:.4f}  -> ceiling {m['bayes_ceiling_raw']:.4f}")
    print(f"  R* (iso):      {m['rstar_isotonic']:.4f}  -> ceiling {m['bayes_ceiling_isotonic']:.4f}")
    print(f"  R* (temp):     {m['rstar_temperature']:.4f}  -> ceiling {m['bayes_ceiling_temperature']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run baseline models on PhysioNet 2012"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Models to run (default: all 6)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  PhysioNet 2012 - Baseline Evaluation + R* Estimation")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, Patience: {args.patience}")
    print("=" * 60)

    # Load data (cached)
    print("\n[Step 1] Loading P12 data...")
    data = preprocess_physionet2012("all", rate=0.1)
    ns, nf, nc = data["n_steps"], data["n_features"], data["n_classes"]
    print(
        f"  Data loaded: steps={ns}, features={nf}, classes={nc}"
    )
    print(
        f"  Shapes: train={data['train_X'].shape}, "
        f"val={data['val_X'].shape}, test={data['test_X'].shape}"
    )

    # Save test labels once
    labels_path = os.path.join(OUTPUT_DIR, "p12_test_labels.npy")
    np.save(labels_path, data["test_y"])
    print(f"  Test labels saved to {labels_path}")

    # Run models sequentially
    all_metrics = []
    for model_name in args.models:
        print(f"\n{'#' * 60}")
        print(f"# Running {model_name}")
        print(f"{'#' * 60}")

        try:
            metrics = run_single_model(
                model_name, data, args.epochs, args.batch_size, args.patience
            )
            print_metrics(metrics)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            traceback.print_exc()
            all_metrics.append({"model": model_name, "error": str(e)})

    # Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY: All P12 Baselines")
    print("=" * 80)
    header = (
        f"{'Model':15s} {'Acc':>7s} {'F1':>7s} {'AUROC':>7s} "
        f"{'R*(iso)':>8s} {'R*(temp)':>8s} {'ECE(raw)':>8s} {'ECE(iso)':>8s}"
    )
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        if "error" in m:
            print(f"{m['model']:15s} FAILED: {m['error'][:50]}")
        else:
            print(
                f"{m['model']:15s} {m['accuracy']:7.4f} {m['macro_f1']:7.4f} "
                f"{m.get('auroc', 0):7.4f} {m['rstar_isotonic']:8.4f} "
                f"{m['rstar_temperature']:8.4f} {m['ece_raw']:8.4f} "
                f"{m['ece_isotonic']:8.4f}"
            )

    # Save combined summary
    summary_path = os.path.join(OUTPUT_DIR, "p12_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
