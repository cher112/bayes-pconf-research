#!/usr/bin/env python3
"""
Run all 6 baseline models on PhysioNet 2019 Sepsis data.
Sequential execution on single GPU, 100 epochs each.
Outputs: per-model metrics.json + softmax probs for R* estimation.
"""
import sys
import os
import json
import time
import traceback
import numpy as np
from datetime import datetime

sys.path.insert(0, "/root/autodl-tmp/IrregularlySampledTimeSeriesLibrary")
sys.path.insert(0, "/root/bayes-pconf-research/src")

# ---- Monkey-patch pypots device issue ----
import pypots.base

_orig = pypots.base.BaseModel._setup_device


def _p(self, d):
    if isinstance(d, str) and d.lower() == "best":
        d = None
    _orig(self, d)


pypots.base.BaseModel._setup_device = _p

from preprocess_p19 import preprocess_physionet2019
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---- Calibration utilities (from run_baseline.py) ----


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_labels == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(
                confidences[mask].mean() - accuracies[mask].mean()
            )
    return ece / len(labels)


def isotonic_calibrate_cv(probs, labels, n_classes, n_splits=5):
    """Cross-validated isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(probs, labels):
        for c in range(n_classes):
            y_bin = (labels[train_idx] == c).astype(float)
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            ir.fit(probs[train_idx, c], y_bin)
            calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-10)
    return calibrated


def temperature_scale_cv(probs, labels, n_splits=5):
    """Cross-validated temperature scaling."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss

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
        scaled = np.exp(logits_val / best_T)
        calibrated[val_idx] = scaled / scaled.sum(axis=1, keepdims=True)
    return calibrated


def spline_calibrate_cv(probs, labels, n_classes, n_splits=5):
    """Cross-validated spline calibration using UnivariateSpline."""
    from sklearn.model_selection import StratifiedKFold
    from scipy.interpolate import UnivariateSpline

    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(probs, labels):
        for c in range(n_classes):
            y_bin = (labels[train_idx] == c).astype(float)
            p_train = probs[train_idx, c]
            # Sort for spline fitting
            order = np.argsort(p_train)
            p_sorted = p_train[order]
            y_sorted = y_bin[order]
            # Bin to smooth
            n_bins = min(50, len(p_sorted) // 20)
            if n_bins < 5:
                # Fall back to isotonic if too few samples
                from sklearn.isotonic import IsotonicRegression

                ir = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds="clip"
                )
                ir.fit(p_train, y_bin)
                calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
                continue
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_means = []
            for b in range(n_bins):
                mask = (p_sorted >= bin_edges[b]) & (p_sorted < bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_centers.append(p_sorted[mask].mean())
                    bin_means.append(y_sorted[mask].mean())
            if len(bin_centers) < 4:
                from sklearn.isotonic import IsotonicRegression

                ir = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds="clip"
                )
                ir.fit(p_train, y_bin)
                calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
                continue
            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            try:
                spl = UnivariateSpline(bin_centers, bin_means, k=3, s=0.1)
                cal_vals = spl(probs[val_idx, c])
                cal_vals = np.clip(cal_vals, 0.01, 0.99)
                calibrated[val_idx, c] = cal_vals
            except Exception:
                from sklearn.isotonic import IsotonicRegression

                ir = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds="clip"
                )
                ir.fit(p_train, y_bin)
                calibrated[val_idx, c] = ir.predict(probs[val_idx, c])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-10)
    return calibrated


def calibrate_and_estimate_rstar(probs, labels, n_classes):
    """Full calibration + R* estimation with isotonic, temperature, spline."""
    results = {}

    # Basic metrics
    pred_labels = np.argmax(probs, axis=1)
    results["accuracy"] = float(accuracy_score(labels, pred_labels))
    results["macro_f1"] = float(f1_score(labels, pred_labels, average="macro"))
    if n_classes == 2:
        results["auroc"] = float(roc_auc_score(labels, probs[:, 1]))

    # ECE
    results["ece_raw"] = float(compute_ece(probs, labels))

    # Calibrations
    calib_iso = isotonic_calibrate_cv(probs, labels, n_classes)
    results["ece_isotonic"] = float(compute_ece(calib_iso, labels))

    calib_temp = temperature_scale_cv(probs, labels)
    results["ece_temperature"] = float(compute_ece(calib_temp, labels))

    calib_spl = spline_calibrate_cv(probs, labels, n_classes)
    results["ece_spline"] = float(compute_ece(calib_spl, labels))

    # R* estimation
    for name, cal_probs in [
        ("raw", probs),
        ("isotonic", calib_iso),
        ("temperature", calib_temp),
        ("spline", calib_spl),
    ]:
        if n_classes == 2:
            eta = cal_probs[:, 1]
            rstar = 0.5 - 0.5 * np.mean(np.abs(2 * eta - 1))
        else:
            rstar = np.mean(1 - np.max(cal_probs, axis=1))
        results[f"rstar_{name}"] = float(rstar)
        results[f"bayes_ceiling_{name}"] = float(1 - rstar)

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

EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
OUTPUT_DIR = "/root/bayes-pconf-research/experiments"


def run_single_model(model_name, data, exp_dir):
    """Train one model, predict, calibrate, save results."""
    from pypots.classification import (
        GRUD,
        BRITS,
        iTransformer,
        SAITS,
        TimesNet,
        SeFT,
    )

    model_classes = {
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
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "saving_path": os.path.join(exp_dir, "checkpoints"),
        "model_saving_strategy": "best",
    }

    kwargs = {**MODELS[model_name], **common}
    model_cls = model_classes[model_name]

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
    probs = results.get("classification_proba", None)
    if probs is None:
        print(f"[{model_name}] WARNING: no proba output, using labels")
        preds = results["classification"]
        probs = np.zeros((len(preds), nc))
        for i, p in enumerate(preds.ravel().astype(int)):
            probs[i, p] = 0.9
            probs[i, :] += 0.1 / nc

    test_labels = data["test_y"]

    # Save raw predictions
    np.save(os.path.join(exp_dir, "probs.npy"), probs)
    np.save(os.path.join(exp_dir, "labels.npy"), test_labels)

    # Calibrate + R*
    metrics = calibrate_and_estimate_rstar(probs, test_labels, nc)
    metrics["model"] = model_name
    metrics["dataset"] = "p19"
    metrics["epochs"] = EPOCHS
    metrics["n_params"] = n_params
    metrics["train_time_s"] = round(train_time, 1)
    metrics["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def print_metrics(m):
    """Pretty-print one model's metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {m['model']} on P19  ({m['n_params']:,} params, {m['train_time_s']}s)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {m['accuracy']:.4f}")
    print(f"  Macro-F1:     {m['macro_f1']:.4f}")
    if "auroc" in m:
        print(f"  AUROC:        {m['auroc']:.4f}")
    print(f"  ECE (raw):    {m['ece_raw']:.4f}")
    print(f"  ECE (iso):    {m['ece_isotonic']:.4f}")
    print(f"  ECE (temp):   {m['ece_temperature']:.4f}")
    print(f"  ECE (spline): {m['ece_spline']:.4f}")
    print(f"  R* (raw):     {m['rstar_raw']:.4f}  -> ceiling {m['bayes_ceiling_raw']:.4f}")
    print(f"  R* (iso):     {m['rstar_isotonic']:.4f}  -> ceiling {m['bayes_ceiling_isotonic']:.4f}")
    print(f"  R* (temp):    {m['rstar_temperature']:.4f}  -> ceiling {m['bayes_ceiling_temperature']:.4f}")
    print(f"  R* (spline):  {m['rstar_spline']:.4f}  -> ceiling {m['bayes_ceiling_spline']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Models to run (default: all 6)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PhysioNet 2019 Sepsis - Baseline Evaluation")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Patience: {PATIENCE}")
    print("=" * 60)

    # Step 1: Preprocess
    print("\n[Step 1] Preprocessing P19 data...")
    data = preprocess_physionet2019()
    print(
        f"  Data loaded: train={data['train_X'].shape}, "
        f"val={data['val_X'].shape}, test={data['test_X'].shape}"
    )

    # Step 2: Run models sequentially
    all_metrics = []
    for model_name in args.models:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(OUTPUT_DIR, f"p19_{model_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"# Running {model_name}")
        print(f"# Output: {exp_dir}")
        print(f"{'#' * 60}")

        try:
            metrics = run_single_model(model_name, data, exp_dir)
            print_metrics(metrics)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            traceback.print_exc()
            all_metrics.append({"model": model_name, "error": str(e)})

    # Step 3: Summary
    print("\n\n" + "=" * 80)
    print("  SUMMARY: All P19 Baselines")
    print("=" * 80)
    header = f"{'Model':15s} {'Acc':>7s} {'F1':>7s} {'AUROC':>7s} {'R*(iso)':>8s} {'R*(temp)':>8s} {'R*(spl)':>8s} {'ECE(iso)':>8s}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        if "error" in m:
            print(f"{m['model']:15s} FAILED: {m['error'][:50]}")
        else:
            print(
                f"{m['model']:15s} {m['accuracy']:7.4f} {m['macro_f1']:7.4f} "
                f"{m.get('auroc', 0):7.4f} {m['rstar_isotonic']:8.4f} "
                f"{m['rstar_temperature']:8.4f} {m['rstar_spline']:8.4f} "
                f"{m['ece_isotonic']:8.4f}"
            )

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "p19_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
