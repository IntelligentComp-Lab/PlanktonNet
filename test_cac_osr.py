#!/usr/bin/env python3
"""
CAC Open-Set Recognition (OSR) Evaluation Script

Implements 8 OSR metrics:
  1. Known Class Accuracy (Closed-set)
  2. Known Class Accuracy (Thresholded)
  3. Unknown Class Accuracy (Thresholded)
  4. Open-Set Accuracy
  5. Open-Set F-score
  6. AUROC (Area Under ROC Curve)
  7. AUPR (Area Under Precision-Recall Curve)
  8. FPR95 (False Positive Rate at 95% TPR)

Rejection score: gamma = d * (1 - softmin(d))
Lower score = Known, Higher score = Unknown.
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from timm.data import create_dataset, create_loader
from timm.utils import setup_default_logging

from PlanktonNet import PlanktonNet
from cacUtils import CACModel


def parse_args():
    parser = argparse.ArgumentParser(description='CAC OSR Test - Paper Metrics')
    parser.add_argument('--data-dir', required=True, metavar='DIR')
    parser.add_argument('--dataset', default='ImageFolder')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-classes', type=int, default=50)
    parser.add_argument('--feature-dim', type=int, default=448)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--train-split', default='train', help='Gallery set')
    parser.add_argument('--test-known-split', default='test_know', help='Known-class test set')
    parser.add_argument('--test-unknown-split', default='test_unknow', help='Unknown-class test set')
    args = parser.parse_args()
    return args


def get_features_and_targets(model, loader, device):
    """Extract projected features and labels from a data loader."""
    feats = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader, desc="Extracting")):
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            # Handle tuple/list outputs: use the last element as features
            if isinstance(output, (tuple, list)):
                f = output[-1]
            else:
                f = output

            feats.append(f.cpu())
            targets.append(target.cpu())

    feats = torch.cat(feats, dim=0)
    targets = torch.cat(targets, dim=0)
    return feats, targets


def calculate_cac_score(features, centers):
    """Compute CAC rejection score: gamma = d * (1 - softmin(d))

    Args:
        features: [N, Dim] feature matrix
        centers: [C, Dim] class center matrix
    Returns:
        min_scores: [N] per-sample minimum rejection score
        preds: [N] predicted class indices
    """
    # 1. Euclidean distance: [N, C]
    dist = torch.cdist(features, centers, p=2)

    # 2. Softmin (softmax over negative distances): [N, C]
    softmin_score = F.softmax(-dist, dim=1)

    # 3. Rejection score: d * (1 - softmin(d)) -> [N, C]
    score_map = dist * (1.0 - softmin_score)

    # 4. Per-sample minimum score and corresponding class
    min_scores, preds = torch.min(score_map, dim=1)

    return min_scores, preds


def calculate_paper_metrics_cac(pred_k, label_k, score_k, score_u, threshold):
    """Compute the 5 paper OSR metrics for a given threshold.

    CAC convention: lower score = known, higher score = unknown.
    """
    # 1. Known Class Accuracy (Closed-set)
    correct_k_closed = (pred_k == label_k)
    acc_known_closed = correct_k_closed.float().mean().item()

    # Threshold decisions
    accepted_mask_k = score_k <= threshold       # known samples accepted
    rejected_mask_u = score_u > threshold         # unknown samples rejected

    # 2. Known Class Accuracy (Thresholded): correct AND accepted
    correct_k_th = correct_k_closed & accepted_mask_k
    acc_known_th = correct_k_th.float().sum().item() / len(label_k)

    # 3. Unknown Class Accuracy (Thresholded)
    acc_unknown_th = rejected_mask_u.float().sum().item() / len(score_u)

    # 4. Open-Set Accuracy
    total_correct = correct_k_th.float().sum().item() + rejected_mask_u.float().sum().item()
    total_samples = len(label_k) + len(score_u)
    acc_open_set = total_correct / total_samples

    # 5. Open-Set F-score
    # TP: known, correct, accepted | FP: unknown, wrongly accepted | FN: rejected or misclassified
    tp = correct_k_th.float().sum().item()
    fp = (~rejected_mask_u).float().sum().item()
    fn = len(label_k) - tp

    epsilon = 1e-10
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return acc_known_closed, acc_known_th, acc_unknown_th, acc_open_set, f_score


def main():
    setup_default_logging()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- 1. Load model ---
    print("Creating PlanktonNet + CACModel...")
    backbone = PlanktonNet(pretrained=False)
    model = CACModel(backbone, feature_dim=args.feature_dim, num_classes=args.num_classes)

    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # --- 2. Prepare data ---
    root_path = os.path.join(args.data_dir, args.dataset)
    data_cfg = {'input_size': (3, 224, 224), 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}

    # Gallery (training set) — for computing class centers
    ds_g = create_dataset(name='', root=os.path.join(root_path, args.train_split), is_training=False,
                          batch_size=args.batch_size)
    loader_g = create_loader(ds_g, data_cfg['input_size'], args.batch_size, is_training=False, mean=data_cfg['mean'],
                             std=data_cfg['std'], num_workers=args.workers, device=device)
    class_map = ds_g.reader.class_to_idx if hasattr(ds_g, 'reader') else ds_g.class_to_idx

    # Known-class test set
    ds_k = create_dataset(name='', root=os.path.join(root_path, args.test_known_split), is_training=False,
                          batch_size=args.batch_size, class_map=class_map)
    loader_k = create_loader(ds_k, data_cfg['input_size'], args.batch_size, is_training=False, mean=data_cfg['mean'],
                             std=data_cfg['std'], num_workers=args.workers, device=device)

    # Unknown-class test set
    ds_u = create_dataset(name='', root=os.path.join(root_path, args.test_unknown_split), is_training=False,
                          batch_size=args.batch_size)
    loader_u = create_loader(ds_u, data_cfg['input_size'], args.batch_size, is_training=False, mean=data_cfg['mean'],
                             std=data_cfg['std'], num_workers=args.workers, device=device)

    # --- 3. Compute class centers from gallery ---
    # Using gallery feature means as centers is more robust than fixed training anchors.
    print("Calculating Class Centers from Gallery...")
    feats_g, targets_g = get_features_and_targets(model, loader_g, device)

    unique_classes = torch.unique(targets_g).sort()[0]
    class_centers = []
    for cls_idx in unique_classes:
        mask = targets_g == cls_idx
        center = feats_g[mask].mean(dim=0)
        class_centers.append(center)

    # [C, Dim]
    class_centers = torch.stack(class_centers).to(device)
    print(f"Class Centers Shape: {class_centers.shape}")

    # --- 4. Extract features and compute scores ---
    print("Processing Known Test Set...")
    feats_k, targets_k = get_features_and_targets(model, loader_k, device)
    scores_k, preds_k = calculate_cac_score(feats_k.to(device), class_centers)
    scores_k = scores_k.cpu()
    preds_k = preds_k.cpu()

    print("Processing Unknown Test Set...")
    feats_u, _ = get_features_and_targets(model, loader_u, device)
    scores_u, _ = calculate_cac_score(feats_u.to(device), class_centers)
    scores_u = scores_u.cpu()

    print("\n[Debug] CAC Score Statistics (Lower is Known, Higher is Unknown):")
    print(f"Known:   Min={scores_k.min():.4f}, Mean={scores_k.mean():.4f}, Max={scores_k.max():.4f}")
    print(f"Unknown: Min={scores_u.min():.4f}, Mean={scores_u.mean():.4f}, Max={scores_u.max():.4f}")

    # --- 5. Threshold search (maximize Open-Set F-score) ---
    print("\nFinding Optimal Threshold based on Open-Set F-score...")

    all_scores = torch.cat([scores_k, scores_u]).numpy()
    th_range = np.linspace(all_scores.min(), all_scores.max(), num=500)

    best_metrics = {}
    max_f_score = -1.0
    best_th = 0.0

    for th in th_range:
        m1, m2, m3, m4, m5 = calculate_paper_metrics_cac(preds_k, targets_k, scores_k, scores_u, th)

        if m5 > max_f_score:
            max_f_score = m5
            best_th = th
            best_metrics = {
                "Known Acc": m1,
                "Known Acc (Th)": m2,
                "Unknown Acc (Th)": m3,
                "Open-Set Acc": m4,
                "Open-Set F-score": m5
            }

    # --- 6. Threshold-independent metrics ---
    # Binary labels: 0 = known (negative), 1 = unknown (positive)
    y_true = np.concatenate([np.zeros(len(scores_k)), np.ones(len(scores_u))])
    y_score = np.concatenate([scores_k.numpy(), scores_u.numpy()])

    # AUROC (higher score = unknown, so use raw scores)
    auroc = roc_auc_score(y_true, y_score)

    # AUPR (average precision)
    aupr = average_precision_score(y_true, y_score)

    # FPR95: false positive rate at 95% true positive rate
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find FPR where TPR >= 0.95
    idx_tpr95 = np.argmax(tpr >= 0.95)
    fpr95 = fpr[idx_tpr95] if idx_tpr95 > 0 else 1.0

    best_metrics["AUROC"] = auroc
    best_metrics["AUPR"] = aupr
    best_metrics["FPR95"] = fpr95

    # --- 7. Print results ---
    print("\n" + "=" * 60)
    print(f"CAC RESULTS (Best Threshold = {best_th:.4f})")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value (%)':<10}")
    print("-" * 45)
    print(f"{'Known Class Accuracy':<30} | {best_metrics['Known Acc'] * 100:.2f}%")
    print(f"{'Known Class Accuracy (Th)':<30} | {best_metrics['Known Acc (Th)'] * 100:.2f}%")
    print(f"{'Unknown Class Accuracy (Th)':<30} | {best_metrics['Unknown Acc (Th)'] * 100:.2f}%")
    print(f"{'Open-Set Accuracy':<30} | {best_metrics['Open-Set Acc'] * 100:.2f}%")
    print(f"{'Open-Set F-score':<30} | {best_metrics['Open-Set F-score'] * 100:.2f}%")
    print("-" * 45)
    print(f"{'AUROC':<30} | {best_metrics['AUROC'] * 100:.2f}%")
    print(f"{'AUPR':<30} | {best_metrics['AUPR'] * 100:.2f}%")
    print(f"{'FPR95':<30} | {best_metrics['FPR95'] * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
