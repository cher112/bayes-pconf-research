"""
Run R*(t) truncation experiments on PhysioNet 2019.
Truncate records at t=6, 12, 24, 48, 72h and estimate R* at each horizon.
Uses GRU-D (fastest encoder).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/root/bayes-pconf-research/src")
from preprocess_p19 import preprocess_physionet2019
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pypots.classification import GRUD

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(labels)

def isotonic_calibrate_cv(probs, labels, n_classes, n_splits=5):
    cal_probs = np.copy(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, cal_idx in skf.split(probs, labels):
        for c in range(n_classes):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[train_idx, c], (labels[train_idx] == c).astype(float))
            cal_probs[cal_idx, c] = ir.predict(probs[cal_idx, c])
    row_sums = cal_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cal_probs /= row_sums
    return cal_probs

def estimate_rstar(cal_probs):
    eta = cal_probs[:, 1]
    rstar = 0.5 - 0.5 * np.mean(np.abs(2 * eta - 1))
    return max(0, rstar)

print("Loading PhysioNet 2019...")
data = preprocess_physionet2019()
ns, nf, nc = data["n_steps"], data["n_features"], data["n_classes"]
print(f"Full data: {ns} steps, {nf} features, {nc} classes")
print(f"Shapes: train={data['train_X'].shape}, val={data['val_X'].shape}, test={data['test_X'].shape}")

OUTPUT_DIR = "/root/bayes-pconf-research/experiments/truncated_p19"
os.makedirs(OUTPUT_DIR, exist_ok=True)

horizons = [6, 12, 24, 48, 72]
all_results = []

for t in horizons:
    t_steps = min(t, ns)
    print(f"\n{'='*60}")
    print(f"  Truncation: t={t}h ({t_steps} steps)")
    print(f"{'='*60}")

    train_X = data["train_X"][:, :t_steps, :]
    val_X = data["val_X"][:, :t_steps, :]
    test_X = data["test_X"][:, :t_steps, :]

    exp_dir = os.path.join(OUTPUT_DIR, f"t{t}")
    os.makedirs(exp_dir, exist_ok=True)

    model = GRUD(
        n_steps=t_steps, n_features=nf, n_classes=nc,
        rnn_hidden_size=64,
        batch_size=64, epochs=100, patience=10,
        saving_path=os.path.join(exp_dir, "checkpoints"),
        model_saving_strategy="best",
    )

    t0 = time.time()
    model.fit(
        {"X": train_X, "y": data["train_y"]},
        {"X": val_X, "y": data["val_y"]},
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    results = model.predict({"X": test_X})
    probs = results.get("classification_proba", None)
    if probs is None:
        preds = results["classification"]
        probs = np.zeros((len(preds), nc))
        for i, p in enumerate(preds.ravel().astype(int)):
            probs[i, p] = 0.9
            probs[i, :] += 0.1 / nc

    test_labels = data["test_y"]
    pred_labels = np.argmax(probs, axis=1)
    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="macro")
    auroc = roc_auc_score(test_labels, probs[:, 1])

    cal_probs = isotonic_calibrate_cv(probs, test_labels, nc)
    rstar = estimate_rstar(cal_probs)
    ceiling = 1 - rstar
    ece = compute_ece(cal_probs[:, 1], (test_labels == 1).astype(float))

    metrics = {
        "truncation_hours": int(t),
        "n_steps_used": int(t_steps),
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(f1, 4)),
        "auroc": float(round(auroc, 4)),
        "rstar_isotonic": float(round(rstar, 4)),
        "bayes_ceiling": float(round(ceiling, 4)),
        "ece_isotonic": float(round(ece, 4)),
        "train_time_s": float(round(train_time, 1)),
        "model": "GRUD",
        "dataset": "p19",
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  AUROC={auroc:.4f}, Acc={acc:.4f}, R*={rstar:.4f}, Ceiling={ceiling:.4f}")
    all_results.append(metrics)

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\n\nR*(t) Summary for P19:")
print(f"{'t':>4s}  {'AUROC':>7s}  {'Acc':>7s}  {'R*':>7s}  {'Ceiling':>7s}  {'Gap':>7s}")
for r in all_results:
    gap = r['bayes_ceiling'] - r['auroc']
    print(f"{r['truncation_hours']:4d}  {r['auroc']:7.4f}  {r['accuracy']:7.4f}  {r['rstar_isotonic']:7.4f}  {r['bayes_ceiling']:7.4f}  {gap:7.4f}")
