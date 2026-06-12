"""
Shared OSR metrics module.
Provides standard OSR metrics (AUROC, AUPR, FPR95) plus paper-defined metrics.

Binary label convention:
    known samples → label 0 (positive/normal)
    unknown samples → label 1 (negative/anomaly)

Score convention:
    Higher score → more likely to be unknown (anomaly).
    Callers must ensure score direction matches this convention.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_standard_metrics(scores_known, scores_unknown):
    y_true = np.concatenate([
        np.zeros(len(scores_known), dtype=int),
        np.ones(len(scores_unknown), dtype=int),
    ])
    y_score = np.concatenate([scores_known, scores_unknown])
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx_tpr95 = np.searchsorted(tpr, 0.95)
    if idx_tpr95 >= len(fpr):
        idx_tpr95 = len(fpr) - 1
    fpr95 = fpr[idx_tpr95]
    return dict(auroc=auroc, aupr=aupr, fpr95=fpr95)


def compute_paper_metrics(score_k, pred_k, label_k, score_u, threshold):
    import torch
    correct_k_closed = (pred_k == label_k)
    acc_known_closed = float(correct_k_closed.float().mean())
    accepted_mask_k = score_k <= threshold
    rejected_mask_u = score_u > threshold
    correct_k_th = correct_k_closed & accepted_mask_k
    n_k = len(label_k)
    acc_known_th = float(correct_k_th.float().sum()) / n_k
    n_u = len(score_u)
    acc_unknown_th = float(rejected_mask_u.float().sum()) / n_u
    total_correct = float(correct_k_th.float().sum()) + float(rejected_mask_u.float().sum())
    acc_open_set = total_correct / (n_k + n_u)
    tp = float(correct_k_th.float().sum())
    fp = float((~rejected_mask_u).float().sum())
    fn = n_k - tp
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f_score = 2 * precision * recall / (precision + recall + eps)
    return acc_known_closed, acc_known_th, acc_unknown_th, acc_open_set, f_score


def search_best_threshold(score_k, pred_k, label_k, score_u, n_steps=2000):
    import torch
    score_k_np = np.asarray(score_k)
    score_u_np = np.asarray(score_u)
    all_scores = np.concatenate([score_k_np, score_u_np])
    thresholds = np.unique(all_scores)
    if len(thresholds) > n_steps:
        thresholds = np.linspace(all_scores.min(), all_scores.max(), n_steps)
    best_f = -1.0
    best_th = 0.0
    best_metrics = None
    for th in thresholds:
        metrics = compute_paper_metrics(
            torch.as_tensor(score_k_np), torch.as_tensor(np.asarray(pred_k)),
            torch.as_tensor(np.asarray(label_k)), torch.as_tensor(score_u_np), th)
        if metrics[4] > best_f:
            best_f = metrics[4]
            best_th = th
            best_metrics = metrics
    return best_th, best_metrics


def compute_tau_curve(score_k, pred_k, label_k, score_u, n_steps=500):
    import torch
    score_k_np = np.asarray(score_k)
    score_u_np = np.asarray(score_u)
    all_scores = np.concatenate([score_k_np, score_u_np])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_steps)
    curves = dict(thresholds=thresholds, known_closed=[], known_th=[],
                  unknown_th=[], os_acc=[], f_score=[])
    for th in thresholds:
        metrics = compute_paper_metrics(
            torch.as_tensor(score_k_np), torch.as_tensor(np.asarray(pred_k)),
            torch.as_tensor(np.asarray(label_k)), torch.as_tensor(score_u_np), th)
        curves['known_closed'].append(metrics[0])
        curves['known_th'].append(metrics[1])
        curves['unknown_th'].append(metrics[2])
        curves['os_acc'].append(metrics[3])
        curves['f_score'].append(metrics[4])
    for k in curves:
        if k != 'thresholds':
            curves[k] = np.array(curves[k])
    return curves
