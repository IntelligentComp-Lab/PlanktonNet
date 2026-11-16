#!/usr/bin/env python3
"""Evaluate PMAL with the feature-center Euclidean score used in the May paper table.

The unknown score is the distance from a backbone feature to its nearest
training-class feature centroid.  The decision threshold is selected only on
validation/validation_unknow and then frozen for the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation_common import (
    PAPER_METRIC_NAMES,
    REVISION_ROOT,
    compute_paper_metrics,
    compute_standard_metrics,
    get_class_map,
    make_loader,
    metrics_for_validation,
    sha256_file,
)
from osr_metrics import compute_tau_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                        choices=("plankton_net", "tf_efficientnetv2_s.in1k"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REVISION_ROOT / "data")
    parser.add_argument("--dataset", default="ZooplanktonBench50_OpenSet")
    parser.add_argument("--val-unknown-split", default="validation_unknow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def extract_features(backbone, loader, device, description):
    from tqdm import tqdm

    feature_batches, target_batches = [], []
    for images, targets in tqdm(loader, desc=description):
        features = backbone(images.to(device))
        if features.ndim > 2:
            features = features.flatten(1)
        feature_batches.append(features.cpu())
        target_batches.append(targets.cpu())
    return torch.cat(feature_batches), torch.cat(target_batches)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from pmal.pmalUtils import PMALModel, PMALoss, PMALWrapper, create_pmal_backbone

    backbone, _ = create_pmal_backbone(args.arch, 448, img_size=224)
    core = PMALModel(backbone, feature_dim=448, num_classes=50)
    model = PMALWrapper(core, PMALoss(50, device=device)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value
             for key, value in checkpoint.get("state_dict", checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    root = args.data_dir / args.dataset
    train_loader, train_dataset = make_loader(
        root, "train", args.batch_size, args.workers, device)
    class_map = get_class_map(train_dataset)
    loaders = {
        "train": train_loader,
        "validation": make_loader(root, "validation", args.batch_size,
                                  args.workers, device, class_map)[0],
        "validation_unknow": make_loader(root, args.val_unknown_split, args.batch_size,
                                         args.workers, device)[0],
        "test_know": make_loader(root, "test_know", args.batch_size,
                                 args.workers, device, class_map)[0],
        "test_unknow": make_loader(root, "test_unknow", args.batch_size,
                                   args.workers, device)[0],
    }

    feature_sets = {
        name: extract_features(model.model.backbone, loader, device, name)
        for name, loader in loaders.items()
    }
    train_features, train_targets = feature_sets["train"]
    classes = torch.unique(train_targets).sort().values
    centers = torch.stack([
        train_features[train_targets == class_id].mean(0) for class_id in classes
    ])

    def nearest_center(features):
        distances = torch.cdist(features.float(), centers.float(), p=2)
        scores, predictions = distances.min(1)
        return scores, predictions

    val_scores, val_predictions = nearest_center(feature_sets["validation"][0])
    val_unknown_scores, _ = nearest_center(feature_sets["validation_unknow"][0])
    test_scores, test_predictions = nearest_center(feature_sets["test_know"][0])
    test_unknown_scores, _ = nearest_center(feature_sets["test_unknow"][0])

    selection = metrics_for_validation(
        val_scores.numpy(), val_predictions.numpy(),
        feature_sets["validation"][1].numpy(), val_unknown_scores.numpy())
    threshold = float(selection["best_threshold"])
    paper = compute_paper_metrics(
        test_scores, test_predictions, feature_sets["test_know"][1],
        test_unknown_scores, threshold)
    standard = {
        key: float(value) for key, value in
        compute_standard_metrics(test_scores.numpy(), test_unknown_scores.numpy()).items()
    }

    output = {
        "method": "PMAL feature-center Euclidean distance",
        "backbone": args.arch,
        "score_formula": "min_c ||f(x)-mean_{i:y_i=c} f(x_i)||_2",
        "prediction_formula": "argmin_c ||f(x)-mean_{i:y_i=c} f(x_i)||_2",
        "threshold_selection": "validation + class-disjoint validation_unknow only",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "best_threshold": threshold,
        "paper_metrics": dict(zip(PAPER_METRIC_NAMES, map(float, paper))),
        "standard_metrics": standard,
        "validation_selection": selection,
        "scores": {
            "known": test_scores.numpy().tolist(),
            "unknown": test_unknown_scores.numpy().tolist(),
            "val": val_scores.numpy().tolist(),
            "val_unknown": val_unknown_scores.numpy().tolist(),
        },
        "tau_curve": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in compute_tau_curve(
                val_scores.numpy(), val_predictions.numpy(),
                feature_sets["validation"][1].numpy(),
                val_unknown_scores.numpy(), n_steps=500).items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Results written to {args.output}', flush=True)


if __name__ == "__main__":
    main()
