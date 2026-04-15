#!/usr/bin/env python3
"""
Baseline training + R* estimation pipeline.
Supports: GRUD, BRITS, Raindrop, Transformer on P12/P19.
Outputs: trained model, softmax predictions, calibration metrics, R* estimates.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# ---- Data loading ----
def load_p12(missing_rate=0.1):
    """Load PhysioNet 2012 via benchpots."""
    from benchpots.datasets import preprocess_physionet2012
    return preprocess_physionet2012('all', rate=missing_rate)

def load_p19(missing_rate=0.1):
    """Load PhysioNet 2019 via benchpots."""
    from benchpots.datasets import preprocess_physionet2019
    return preprocess_physionet2019('all', rate=missing_rate)

def load_p12_local():
    """Load P12 from local pickle (fallback if benchpots download fails)."""
    import pickle
    from sklearn.model_selection import train_test_split
    
    d = pickle.load(open('/root/data/PhysioNet2012_fixed.pkl', 'rb'))
    ds = d['dataset']
    
    # Extract features and labels
    X = np.array([s['features'] for s in ds])  # (N, T, D)
    y = np.array([s['label'] for s in ds])      # (N,)
    mask = np.array([s['mask'] for s in ds]) if 'mask' in ds[0] else np.ones_like(X)
    
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    
    # Split: 70/15/15
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    va_idx, te_idx = train_test_split(te_idx, test_size=0.5, stratify=y[te_idx], random_state=42)
    
    return {
        'n_classes': len(np.unique(y)),
        'n_steps': X.shape[1],
        'n_features': X.shape[2],
        'train_X': X[tr_idx], 'train_y': y[tr_idx],
        'val_X': X[va_idx], 'val_y': y[va_idx],
        'test_X': X[te_idx], 'test_y': y[te_idx],
        'train_missing_mask': mask[tr_idx] if mask is not None else None,
        'val_missing_mask': mask[va_idx] if mask is not None else None,
        'test_missing_mask': mask[te_idx] if mask is not None else None,
    }

# ---- Model training ----
def train_model(model_name, data, epochs=100, batch_size=64, patience=10, device='cuda', save_dir=None):
    """Train a pypots classification model and return test predictions."""
    from pypots.classification import GRUD, BRITS, Raindrop, iTransformer
    
    n_steps = data['n_steps']
    n_features = data['n_features']
    n_classes = data['n_classes']
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model_cls = {
        'GRUD': GRUD,
        'BRITS': BRITS, 
        'Raindrop': Raindrop,
        'iTransformer': iTransformer,
    }[model_name]
    
    # Model-specific kwargs
    kwargs = {
        'n_steps': n_steps,
        'n_features': n_features,
        'n_classes': n_classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'device': torch.device(device),
        'saving_path': save_dir,
        'model_saving_strategy': 'best',
    }
    
    # Add model-specific params
    if model_name == 'GRUD':
        kwargs['rnn_hidden_size'] = 128
    elif model_name == 'BRITS':
        kwargs['rnn_hidden_size'] = 128
    elif model_name == 'Raindrop':
        kwargs['n_layers'] = 2
        kwargs['d_model'] = 128
        kwargs['d_inner'] = 256
        kwargs['n_heads'] = 4
        kwargs['dropout'] = 0.1
    elif model_name == 'iTransformer':
        kwargs['n_layers'] = 2
        kwargs['d_model'] = 128
        kwargs['d_ffn'] = 256
        kwargs['n_heads'] = 4
        kwargs['d_k'] = 32
        kwargs['d_v'] = 32
        kwargs['dropout'] = 0.1
    
    print(f"\n[{model_name}] Training with {n_steps} steps, {n_features} features, {n_classes} classes...")
    model = model_cls(**kwargs)
    
    # Prepare data dicts
    train_set = {"X": data['train_X'], "y": data['train_y']}
    val_set = {"X": data['val_X'], "y": data['val_y']}
    
    model.fit(train_set, val_set)
    
    # Get predictions on test set
    test_set = {"X": data['test_X']}
    results = model.predict(test_set)
    
    return results, model

# ---- Calibration + R* estimation ----
def calibrate_and_estimate_rstar(probs, labels, n_classes):
    """Run isotonic/temperature calibration and estimate R*."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
    
    results = {}
    
    # Basic metrics
    pred_labels = np.argmax(probs, axis=1)
    results['accuracy'] = float(accuracy_score(labels, pred_labels))
    results['macro_f1'] = float(f1_score(labels, pred_labels, average='macro'))
    if n_classes == 2:
        results['auroc'] = float(roc_auc_score(labels, probs[:, 1]))
    
    # ECE
    results['ece_raw'] = float(compute_ece(probs, labels))
    
    # Isotonic calibration (cross-validated)
    calib_iso = isotonic_calibrate_cv(probs, labels, n_classes)
    results['ece_isotonic'] = float(compute_ece(calib_iso, labels))
    
    # Temperature scaling (cross-validated)
    calib_temp = temperature_scale_cv(probs, labels)
    results['ece_temperature'] = float(compute_ece(calib_temp, labels))
    
    # R* estimation
    for name, cal_probs in [('raw', probs), ('isotonic', calib_iso), ('temperature', calib_temp)]:
        if n_classes == 2:
            eta = cal_probs[:, 1]
            rstar = 0.5 - 0.5 * np.mean(np.abs(2 * eta - 1))
        else:
            rstar = np.mean(1 - np.max(cal_probs, axis=1))
        results[f'rstar_{name}'] = float(rstar)
        results[f'bayes_ceiling_{name}'] = float(1 - rstar)
    
    return results

def isotonic_calibrate_cv(probs, labels, n_classes, n_splits=5):
    """Cross-validated isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    
    calibrated = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(probs, labels):
        for c in range(n_classes):
            y_bin = (labels[train_idx] == c).astype(float)
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
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
        best_T, best_loss = 1.0, float('inf')
        for T in np.arange(0.1, 5.0, 0.05):
            scaled = np.exp(logits / T)
            scaled = scaled / scaled.sum(axis=1, keepdims=True)
            loss = log_loss(labels[train_idx], scaled, labels=list(range(probs.shape[1])))
            if loss < best_loss:
                best_T, best_loss = T, loss
        
        logits_val = np.log(np.clip(probs[val_idx], 1e-10, 1.0))
        scaled = np.exp(logits_val / best_T)
        calibrated[val_idx] = scaled / scaled.sum(axis=1, keepdims=True)
    
    return calibrated

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (pred_labels == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(confidences[mask].mean() - accuracies[mask].mean())
    return ece / len(labels)

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='p12', choices=['p12', 'p19'])
    parser.add_argument('--model', type=str, default='GRUD', choices=['GRUD', 'BRITS', 'Raindrop', 'iTransformer', 'TimesNet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='/root/bayes-pconf-research/experiments')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_{args.model}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"=== Experiment: {exp_name} ===")
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    
    # Load data
    print("\n[1/4] Loading data...")
    if args.dataset == 'p12':
        try:
            data = load_p12()
        except Exception as e:
            print(f"  benchpots failed ({e}), using local fallback...")
            data = load_p12_local()
    elif args.dataset == 'p19':
        data = load_p19()
    
    print(f"  Train: {data['train_X'].shape}, Val: {data['val_X'].shape}, Test: {data['test_X'].shape}")
    print(f"  Classes: {data['n_classes']}, Steps: {data['n_steps']}, Features: {data['n_features']}")
    
    # Train model
    print("\n[2/4] Training model...")
    save_dir = os.path.join(exp_dir, 'checkpoints')
    results, model = train_model(
        args.model, data,
        epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, save_dir=save_dir
    )
    
    # Extract predictions
    print("\n[3/4] Extracting predictions...")
    predictions = results.get('classification_proba', results.get('classification'))
    probs = predictions  # pypots returns probabilities
    
    # If probs is 1D (just predicted labels), we need to get actual probabilities
    if probs.ndim == 1:
        print("  WARNING: model returned labels not probabilities. Using predict_proba workaround.")
        # Fallback: use one-hot as proxy (not ideal)
        n_classes = data['n_classes']
        probs_onehot = np.zeros((len(probs.ravel()), n_classes))
        for i, p in enumerate(probs.ravel().astype(int)):
            probs_onehot[i, p] = 0.9
            probs_onehot[i, :] += 0.1 / n_classes
        probs = probs_onehot
    
    test_labels = data['test_y']
    
    # Save raw predictions
    np.save(os.path.join(exp_dir, 'probs.npy'), probs)
    np.save(os.path.join(exp_dir, 'labels.npy'), test_labels)
    
    # Calibrate and estimate R*
    print("\n[4/4] Calibrating and estimating R*...")
    metrics = calibrate_and_estimate_rstar(probs, test_labels, data['n_classes'])
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  Results: {args.model} on {args.dataset}")
    print(f"{'='*60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
    if 'auroc' in metrics:
        print(f"  AUROC:        {metrics['auroc']:.4f}")
    print(f"  ECE (raw):    {metrics['ece_raw']:.4f}")
    print(f"  ECE (iso):    {metrics['ece_isotonic']:.4f}")
    print(f"  ECE (temp):   {metrics['ece_temperature']:.4f}")
    print(f"  R* (raw):     {metrics['rstar_raw']:.4f}  → ceiling {metrics['bayes_ceiling_raw']:.4f}")
    print(f"  R* (iso):     {metrics['rstar_isotonic']:.4f}  → ceiling {metrics['bayes_ceiling_isotonic']:.4f}")
    print(f"  R* (temp):    {metrics['rstar_temperature']:.4f}  → ceiling {metrics['bayes_ceiling_temperature']:.4f}")
    
    # Save metrics
    metrics['model'] = args.model
    metrics['dataset'] = args.dataset
    metrics['epochs'] = args.epochs
    metrics['timestamp'] = timestamp
    
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  Results saved to {exp_dir}/metrics.json")
    return metrics

if __name__ == '__main__':
    main()
