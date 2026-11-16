#!/usr/bin/env python3
"""Evaluate the paper's IsPlanktonCLR checkpoints on the class-disjoint split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import create_dataset, create_loader
from timm.utils import setup_default_logging
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parent

from osr_metrics import (  # noqa: E402
    compute_paper_metrics,
    compute_standard_metrics,
    compute_tau_curve,
    search_best_threshold,
)
from isplanktonir import CosineClassifier  # noqa: E402


CONFIGS = ("plankton_net", "tf_efficientnetv2_s.in1k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "ZooplanktonBench50_OpenSet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=EXPERIMENT_ROOT / "results" / "isplanktonclr"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--val-unknown-split", default="validation_unknow")
    parser.add_argument("--feature-dim", type=int, default=512)
    return parser.parse_args()


def make_loader(root, split, img_size, batch_size, workers, device, class_map=None):
    data_config = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bicubic",
    }
    dataset = create_dataset(
        name="ImageFolder",
        root=os.path.join(root, split),
        is_training=False,
        class_map=class_map,
        batch_size=batch_size,
    )
    loader = create_loader(
        dataset,
        (3, img_size, img_size),
        batch_size,
        is_training=False,
        mean=data_config["mean"],
        std=data_config["std"],
        interpolation=data_config["interpolation"],
        num_workers=workers,
        persistent_workers=workers > 0,
        device=device,
    )
    return loader, dataset


def get_class_map(dataset):
    for attr in ("class_to_idx", "parser", "reader"):
        obj = getattr(dataset, attr, None)
        if obj is not None:
            class_map = getattr(obj, "class_to_idx", None) if attr != "class_to_idx" else obj
            if class_map is not None:
                return class_map
    return None


@torch.no_grad()
def extract_scores(model, loader, device):
    model.eval()
    scores, labels, predictions = [], [], []
    prototype_norm = F.normalize(model.prototype.weight, p=2, dim=1)
    for inputs, targets in tqdm(loader, desc="Score"):
        features = model.backbone(inputs.to(device))
        if features.dim() > 2:
            features = features.flatten(1)
        features = model.bn(model.proj(features))
        normalized = F.normalize(features, p=2, dim=1)
        similarities = F.linear(normalized, prototype_norm)
        max_similarity, preds = torch.max(similarities, dim=1)
        scores.append((1.0 - max_similarity).cpu())
        labels.append(targets.cpu())
        predictions.append(preds.cpu())
    return (
        torch.cat(scores).numpy(),
        torch.cat(labels),
        torch.cat(predictions),
    )


def main() -> None:
    setup_default_logging()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    if args.arch == "plankton_net":
        from PlanktonNet import PlanktonNet

        backbone = PlanktonNet(pretrained=False)
    else:
        backbone = timm.create_model(args.arch, pretrained=False, num_classes=0)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        features = backbone(dummy)
        backbone_dim = features.shape[1] if features.dim() <= 2 else features.flatten(1).shape[1]

    model = CosineClassifier(
        backbone,
        backbone_dim,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {args.arch}: missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.to(device).eval()

    data_root = str(args.data_root.resolve())
    train_loader, train_dataset = make_loader(
        data_root, "train", args.img_size, args.batch_size, args.workers, device
    )
    class_map = get_class_map(train_dataset)
    known_loader, _ = make_loader(
        data_root, "test_know", args.img_size, args.batch_size, args.workers, device, class_map
    )
    unknown_loader, _ = make_loader(
        data_root, "test_unknow", args.img_size, args.batch_size, args.workers, device
    )
    val_loader, _ = make_loader(
        data_root, "validation", args.img_size, args.batch_size, args.workers, device, class_map
    )
    val_unknown_loader, _ = make_loader(
        data_root, args.val_unknown_split, args.img_size, args.batch_size, args.workers, device
    )

    scores_known, labels_known, preds_known = extract_scores(model, known_loader, device)
    scores_unknown, _, _ = extract_scores(model, unknown_loader, device)
    scores_val, labels_val, preds_val = extract_scores(model, val_loader, device)
    scores_val_unknown, _, _ = extract_scores(model, val_unknown_loader, device)

    scores_known = np.asarray(scores_known, dtype=np.float64)
    scores_unknown = np.asarray(scores_unknown, dtype=np.float64)
    scores_val = np.asarray(scores_val, dtype=np.float64)
    scores_val_unknown = np.asarray(scores_val_unknown, dtype=np.float64)

    standard_metrics = compute_standard_metrics(scores_known, scores_unknown)
    best_threshold, _ = search_best_threshold(
        scores_val, preds_val, labels_val, scores_val_unknown
    )
    paper_metrics = compute_paper_metrics(
        torch.as_tensor(scores_known),
        preds_known,
        labels_known,
        torch.as_tensor(scores_unknown),
        best_threshold,
    )
    tau_curve = compute_tau_curve(
        scores_val, preds_val, labels_val, scores_val_unknown, n_steps=50
    )

    result = {
        "method": "IsPlanktonIR",
        "backbone": args.arch,
        "checkpoint": str(checkpoint_path),
        "best_threshold": float(best_threshold),
        "paper_metrics": dict(
            zip(
                ("known_closed", "known_th", "unknown_th", "os_acc", "f_score"),
                (float(value) for value in paper_metrics),
            )
        ),
        "standard_metrics": {key: float(value) for key, value in standard_metrics.items()},
        "scores": {
            "known": scores_known.tolist(),
            "unknown": scores_unknown.tolist(),
            "val": scores_val.tolist(),
            "val_unknown": scores_val_unknown.tolist(),
        },
        "tau_curve": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in tau_curve.items()
        },
    }

    safe_arch = args.arch.replace(".", "_")
    output_path = args.output_dir / f"scores_isplanktonclr_{safe_arch}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"output": str(output_path), **result["paper_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
