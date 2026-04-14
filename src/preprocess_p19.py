#!/usr/bin/env python3
"""
Preprocess PhysioNet 2019 Sepsis data from raw .psv files.
Produces a dict compatible with benchpots preprocess_physionet2012 output:
  train_X, val_X, test_X  (N, T, D) with NaN for missing
  train_y, val_y, test_y  (N,) int labels
  n_steps, n_features, n_classes
"""
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

# 34 time-varying features (columns 0-33 of .psv)
N_TIME_FEATURES = 34
# Max time steps to keep (first 72 hours, like DualDynamics)
MAX_STEPS = 72

CACHE_PATH = "/root/data/p19_preprocessed.npz"


def read_psv_file(filepath):
    """Read a single .psv file, return (time_series, label).
    time_series: list of lists (each row = 34 time-varying features)
    label: 1 if any SepsisLabel==1, else 0
    """
    rows = []
    label = 0
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        for line in reader:
            if len(line) < 41:
                continue  # skip malformed rows
            iculos = int(line[39])  # ICULOS column
            if iculos > MAX_STEPS:
                break
            time_values = [
                float(v) if v != "NaN" else float("nan")
                for v in line[:N_TIME_FEATURES]
            ]
            rows.append(time_values)
            if int(float(line[40])) == 1:  # SepsisLabel column
                label = 1
    return rows, label


def preprocess_physionet2019(
    dir_a="/root/data/training_setA",
    dir_b="/root/data/training_setB",
    max_steps=MAX_STEPS,
    cache_path=CACHE_PATH,
):
    """Read all .psv files, pad to max_steps, split 64/16/20, return dict."""
    if os.path.exists(cache_path):
        print(f"[P19] Loading cached data from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files}

    print("[P19] Reading raw .psv files...")
    all_series = []
    all_labels = []
    skipped = 0

    for directory in [dir_a, dir_b]:
        files = sorted([f for f in os.listdir(directory) if f.endswith(".psv") and not f.startswith("._")])
        for i, fname in enumerate(files):
            ts, lab = read_psv_file(os.path.join(directory, fname))
            if len(ts) < 3:  # skip very short sequences
                skipped += 1
                continue
            all_series.append(ts)
            all_labels.append(lab)
            if (i + 1) % 5000 == 0:
                print(f"  {directory}: {i+1}/{len(files)}")

    print(f"[P19] Read {len(all_series)} patients, skipped {skipped} (too short)")

    # Pad all to max_steps with NaN
    n_patients = len(all_series)
    X = np.full((n_patients, max_steps, N_TIME_FEATURES), np.nan, dtype=np.float64)
    for i, ts in enumerate(all_series):
        T = min(len(ts), max_steps)
        X[i, :T, :] = np.array(ts[:T])

    y = np.array(all_labels, dtype=np.int64)

    # Split: 64/16/20 stratified
    idx = np.arange(n_patients)
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.36, stratify=y, random_state=42
    )
    val_frac = 0.16 / 0.36
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=(1 - val_frac), stratify=y[tmp_idx], random_state=42
    )

    # Normalize using training stats
    train_flat = X[tr_idx].reshape(-1, N_TIME_FEATURES)
    feat_mean = np.nanmean(train_flat, axis=0)
    feat_std = np.nanstd(train_flat, axis=0)
    feat_std[feat_std < 1e-6] = 1.0  # avoid div by zero

    X = (X - feat_mean) / feat_std

    result = {
        "n_classes": int(len(np.unique(y))),
        "n_steps": int(max_steps),
        "n_features": int(N_TIME_FEATURES),
        "train_X": X[tr_idx],
        "train_y": y[tr_idx],
        "val_X": X[va_idx],
        "val_y": y[va_idx],
        "test_X": X[te_idx],
        "test_y": y[te_idx],
    }

    # Print stats
    for split in ["train", "val", "test"]:
        sx = result[f"{split}_X"]
        sy = result[f"{split}_y"]
        print(
            f"  {split}: X={sx.shape}, y={sy.shape}, "
            f"pos_rate={sy.mean():.4f}, missing={np.isnan(sx).mean():.4f}"
        )

    # Cache
    print(f"[P19] Saving cache to {cache_path}")
    np.savez(cache_path, **result)
    print("[P19] Done.")
    return result


if __name__ == "__main__":
    data = preprocess_physionet2019()
    print(
        f"\nSummary: {data['n_steps']} steps, "
        f"{data['n_features']} features, {data['n_classes']} classes"
    )
