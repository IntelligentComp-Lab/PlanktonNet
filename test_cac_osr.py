#!/usr/bin/env python3
"""
CAC Open-Set Recognition Evaluation — Paper Protocol
=====================================================
- Projection matrix W and training anchors are DISCARDED at inference.
- Class centers mu_k are per-class means of BACKBONE features F(x) (448-dim).
- Anomaly score: S = min_k ||F(x) - mu_k||_2  (pure Euclidean distance).
- Threshold tau is selected on the VALIDATION set (maximizing Open-Set F1).
- Final metrics are computed on the held-out TEST set.

Reports 8 metrics:
  1. Known Acc (Closed)   5. Open-Set F-score
  2. Known Acc (Th)       6. AUROC
  3. Unknown Acc (Th)     7. AUPR
  4. Open-Set Acc         8. FPR95

Usage:
    python test_cac_osr.py \\
        --data-dir ./ --dataset ZooplanktonBench50_padding \\
        --checkpoint checkpoint-261.pth.tar \\
        --num-classes 50 --feature-dim 448 \\
        --device cuda:0
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse, logging
import torch, torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.data import create_dataset, create_loader
from timm.utils import setup_default_logging
import timm

from PlanktonNet import PlanktonNet
from cacUtils import CACModel
from osr_metrics import (
    compute_standard_metrics, compute_paper_metrics, search_best_threshold,
)

_logger = logging.getLogger('test_cac')


# ── Helpers ────────────────────────────────────────────────────────────

def get_class_map(dataset):
    for attr in ['class_to_idx', 'parser', 'reader']:
        obj = getattr(dataset, attr, None)
        if obj is not None:
            cm = getattr(obj, 'class_to_idx', None) if attr != 'class_to_idx' else obj
            if cm is not None: return cm
    return None


def make_loader(root, split, img_size, batch_size, workers, device, class_map=None):
    cfg = {'input_size': (3, img_size, img_size),
           'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
    path = os.path.join(root, split)
    ds = create_dataset(name='', root=path, is_training=False,
                        class_map=class_map, batch_size=batch_size)
    loader = create_loader(ds, cfg['input_size'], batch_size, is_training=False,
                           mean=cfg['mean'], std=cfg['std'],
                           num_workers=workers, device=device,
                           persistent_workers=workers > 0)
    return loader, ds


def load_cac_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _logger.info(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CAC OSR Test — Paper Protocol')
    p.add_argument('--data-dir', required=True, metavar='DIR')
    p.add_argument('--dataset', default='ImageFolder')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-classes', type=int, default=50)
    p.add_argument('--feature-dim', type=int, default=448)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--train-split', default='train')
    p.add_argument('--val-split', default='validation')
    p.add_argument('--val-unknown-split', default='validation_unknow')
    p.add_argument('--test-known-split', default='test_know')
    p.add_argument('--test-unknown-split', default='test_unknow')
    return p.parse_args()


def main():
    setup_default_logging()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1. Build model — keep CACModel wrapper (needed for checkpoint keys)
    _logger.info("Building CAC + PlanktonNet model...")
    backbone = PlanktonNet(pretrained=False)
    model = CACModel(backbone, feature_dim=args.feature_dim, num_classes=args.num_classes).to(device)
    load_cac_checkpoint(model, args.checkpoint)
    model.eval()

    # 2. Data loaders
    root = os.path.join(args.data_dir, args.dataset)
    bs, ws, isize = args.batch_size, args.workers, args.img_size

    loader_train, ds_train = make_loader(root, args.train_split, isize, bs, ws, device)
    class_map = get_class_map(ds_train)
    loader_val, _ = make_loader(root, args.val_split, isize, bs, ws, device, class_map)
    loader_val_u, _ = make_loader(root, args.val_unknown_split, isize, bs, ws, device)
    loader_k, _ = make_loader(root, args.test_known_split, isize, bs, ws, device, class_map)
    loader_u, _ = make_loader(root, args.test_unknown_split, isize, bs, ws, device)

    # 3. Compute class centers in BACKBONE feature space
    _logger.info("Computing class centers from training set (backbone features)...")
    feats_g_list, tgts_g_list = [], []
    with torch.no_grad():
        for inp, tgt in tqdm(loader_train, desc="Gallery"):
            # Use backbone directly — discard projection W
            feats_g_list.append(model.backbone(inp.to(device)).cpu())
            tgts_g_list.append(tgt.cpu())
    feats_g = torch.cat(feats_g_list)
    targets_g = torch.cat(tgts_g_list)
    unique = torch.unique(targets_g).sort()[0]
    centers = torch.stack([feats_g[targets_g == c].mean(0) for c in unique]).to(device)
    _logger.info(f"Centers: {centers.shape}")

    # 4. Scoring: backbone feature Euclidean distance
    def extract(loader, desc=""):
        scores, preds, tgts = [], [], []
        with torch.no_grad():
            for inp, tgt in tqdm(loader, desc=desc):
                feats = model.backbone(inp.to(device))
                dist = torch.cdist(feats.float(), centers.float(), p=2)
                min_dists, cls_preds = torch.min(dist, dim=1)
                scores.append(min_dists.cpu())
                preds.append(cls_preds.cpu())
                tgts.append(tgt.cpu())
        return torch.cat(scores), torch.cat(preds), torch.cat(tgts)

    _logger.info("Scoring...")
    s_k, p_k, t_k = extract(loader_k, "Known")
    s_u, _, _     = extract(loader_u, "Unknown")
    s_v, p_v, t_v = extract(loader_val, "Validation")
    s_vu, _, _    = extract(loader_val_u, "Val-Unknown")

    _logger.info(f"Known score:   mean={s_k.mean():.4f}  std={s_k.std():.4f}")
    _logger.info(f"Unknown score: mean={s_u.mean():.4f}  std={s_u.std():.4f}")

    # 5. Select tau on validation set (maximize Open-Set F1)
    _logger.info("Selecting tau on validation set...")
    best_th, _ = search_best_threshold(
        s_v.numpy(), p_v.numpy(), t_v.numpy(), s_vu.numpy())

    # 6. Compute metrics on test set using the fixed tau
    paper = compute_paper_metrics(
        torch.as_tensor(s_k.numpy()), torch.as_tensor(p_k.numpy()),
        torch.as_tensor(t_k.numpy()), torch.as_tensor(s_u.numpy()), best_th)
    std = compute_standard_metrics(s_k.numpy(), s_u.numpy())

    # 7. Print
    print("\n" + "=" * 60)
    print(f"CAC + PlanktonNet  OSR Results  (tau = {best_th:.4f})")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value (%)':<10}")
    print("-" * 45)
    labels = ['Known Acc (Closed)', 'Known Acc (Th)',  'Unknown Acc (Th)',
              'Open-Set Acc',        'Open-Set F-score']
    for i, name in enumerate(labels[:5]):
        print(f"{name:<30} | {paper[i] * 100:.2f}%")
    print("-" * 45)
    for k, v in std.items():
        print(f"{k.upper():<30} | {v * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
