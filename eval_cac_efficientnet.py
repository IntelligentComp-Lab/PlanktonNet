#!/usr/bin/env python3
"""Evaluate CAC + EfficientNetV2-S with training class means and Euclidean distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from eval_all_methods import (
    BackboneWithProjection,
    CACModelWrapper,
    create_backbone,
    detect_backbone_dim,
    make_loader,
)
from evaluation_common import (
    PAPER_METRIC_NAMES, check_new_result_path, check_unknown_classes,
    checkpoint_provenance, get_class_map, score_cac_features, training_class_centers,
)
from osr_metrics import (
    compute_paper_metrics,
    compute_standard_metrics,
    compute_tau_curve,
    search_best_threshold,
)

ARCHIVED_CHECKPOINT_SHA256 = "61797c1adc428ed2bfc6d321c3aa78dd1f113922532a84d8e1be5de1f07a416e"


@torch.no_grad()
def extract(model, loader, centers, class_ids, device, description):
    scores, predictions, targets = [], [], []
    for images, labels in tqdm(loader, desc=description):
        features = model.model.backbone(images.to(device))
        batch_scores, batch_predictions = score_cac_features(features, centers, class_ids)
        scores.append(batch_scores.cpu())
        predictions.append(batch_predictions.cpu())
        targets.append(labels.cpu())
    return torch.cat(scores), torch.cat(predictions), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        check_new_result_path(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device.index is None or device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        print(f"Using {device}: {torch.cuda.get_device_name(device)}", flush=True)
    data_root = args.data_root
    checkpoint = args.checkpoint

    backbone = create_backbone("tf_efficientnetv2_s.in1k", 224, device=device)
    real_dim = detect_backbone_dim(backbone, 224)
    if real_dim != 448:
        backbone = BackboneWithProjection(backbone, real_dim, 448)
    model = CACModelWrapper(backbone, 448, 50).to(device)
    missing, unexpected = model.load_checkpoint(str(checkpoint))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    train_loader, train_dataset = make_loader(
        str(data_root), "train", 224, args.batch_size, args.workers, device)
    class_map = get_class_map(train_dataset)
    validation, _ = make_loader(
        str(data_root), "validation", 224, args.batch_size, args.workers, device, class_map)
    validation_unknown, validation_unknown_data = make_loader(
        str(data_root), "validation_unknow", 224, args.batch_size, args.workers, device)
    test, _ = make_loader(
        str(data_root), "test_know", 224, args.batch_size, args.workers, device, class_map)
    test_unknown, test_unknown_data = make_loader(
        str(data_root), "test_unknow", 224, args.batch_size, args.workers, device)
    check_unknown_classes(class_map, validation_unknown_data, test_unknown_data)

    train_features, train_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="train centers"):
            train_features.append(model.model.backbone(images.to(device)).cpu())
            train_targets.append(labels.cpu())
    train_features = torch.cat(train_features)
    train_targets = torch.cat(train_targets)
    classes, centers = training_class_centers(train_features, train_targets)
    centers = centers.to(device)
    class_ids = classes.to(device)
    val_scores, val_predictions, val_targets = extract(
        model, validation, centers, class_ids, device, "validation")
    val_unknown_scores = extract(model, validation_unknown, centers, class_ids, device, "validation_unknow")[0]
    threshold, _ = search_best_threshold(
        val_scores.numpy(), val_predictions.numpy(), val_targets.numpy(),
        val_unknown_scores.numpy(), n_steps=None)

    test_known_scores, test_predictions, test_targets = extract(
        model, test, centers, class_ids, device, "test_know")
    test_unknown_scores = extract(model, test_unknown, centers, class_ids, device, "test_unknow")[0]
    standard = {
        key: float(value) for key, value in compute_standard_metrics(
            test_known_scores.numpy(), test_unknown_scores.numpy()).items()
    }

    paper = compute_paper_metrics(
        test_known_scores, test_predictions, test_targets, test_unknown_scores, threshold)
    metrics = {
        "unknown_validation_images": int(val_unknown_scores.numel()),
        "threshold": float(threshold),
        "paper_metrics": dict(zip(PAPER_METRIC_NAMES, map(float, paper))),
        "standard_metrics": standard,
        "scores": {
            "known": test_known_scores.numpy().tolist(),
            "unknown": test_unknown_scores.numpy().tolist(),
            "val": val_scores.numpy().tolist(),
            "val_unknown": val_unknown_scores.numpy().tolist(),
        },
        "tau_curve": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in compute_tau_curve(
                val_scores.numpy(), val_predictions.numpy(), val_targets.numpy(),
                val_unknown_scores.numpy(), n_steps=500).items()
        },
    }

    output = {
        "method": "CAC",
        "backbone": "tf_efficientnetv2_s.in1k",
        "cac_protocol": "paper",
        "score_formula": "min distance_to_class_centers",
        "prediction_formula": "argmin distance_to_class_centers",
        "centers": {"source": "train", "class_ids": classes.tolist()},
        "threshold_source": {
            "known_split": "validation", "unknown_split": "validation_unknow",
            "objective": "open_set_f1", "search": "exact",
        },
        **checkpoint_provenance(checkpoint, ARCHIVED_CHECKPOINT_SHA256),
        **metrics,
    }
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results written to {destination}")


if __name__ == "__main__":
    main()
