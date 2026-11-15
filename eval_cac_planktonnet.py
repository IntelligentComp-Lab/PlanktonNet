#!/usr/bin/env python3
"""Evaluate CAC + PlanktonNet with training class means and Euclidean distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation_common import (
    PAPER_METRIC_NAMES, REVISION_ROOT, check_new_result_path,
    check_unknown_classes, checkpoint_provenance, compute_paper_metrics,
    compute_standard_metrics, extract_features, get_class_map, load_model, make_loader,
    metrics_for_validation, score_cac_features, training_class_centers,
)
from osr_metrics import compute_tau_curve

ARCHIVED_CHECKPOINT_SHA256 = "ff076aa7d9344cc5926f7696fa008123fa6024348dfeb21d348cea69b48fefdf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backbone-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=REVISION_ROOT / "data")
    p.add_argument("--dataset", default="ZooplanktonBench50_OpenSet")
    p.add_argument("--val-unknown-split", default="validation_unknow")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    try:
        check_new_result_path(args.output)
    except ValueError as exc:
        p.error(str(exc))
    return args


def calculate(features, centers, class_ids):
    scores, predictions = score_cac_features(features, centers, class_ids)
    return scores.numpy().astype(np.float64), predictions.numpy().astype(np.int64)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = args.data_dir / args.dataset
    train_loader, train_dataset = make_loader(root, "train", args.batch_size, args.workers, device)
    class_map = get_class_map(train_dataset)
    validation, _ = make_loader(root, "validation", args.batch_size, args.workers, device, class_map)
    validation_unknown, validation_unknown_data = make_loader(
        root, args.val_unknown_split, args.batch_size, args.workers, device)
    test, _ = make_loader(root, "test_know", args.batch_size, args.workers, device, class_map)
    test_unknown, test_unknown_data = make_loader(root, "test_unknow", args.batch_size, args.workers, device)
    check_unknown_classes(class_map, validation_unknown_data, test_unknown_data)

    model = load_model(args.checkpoint, args.backbone_source, device)
    train_features, train_targets = extract_features(model, train_loader, device, "train centers")
    class_ids, centers = training_class_centers(train_features, train_targets)

    vk, vt = extract_features(model, validation, device, "validation")
    vu, _ = extract_features(model, validation_unknown, device, args.val_unknown_split)
    vks, vpred = calculate(vk, centers, class_ids)
    vus, _ = calculate(vu, centers, class_ids)
    selected = metrics_for_validation(vks, vpred, vt.numpy(), vus, n_steps=None)
    threshold = float(selected["best_threshold"])

    tk, tt = extract_features(model, test, device, "test_know")
    tu, _ = extract_features(model, test_unknown, device, "test_unknow")
    tks, tpred = calculate(tk, centers, class_ids)
    tus, _ = calculate(tu, centers, class_ids)
    paper = compute_paper_metrics(torch.as_tensor(tks), torch.as_tensor(tpred), tt, torch.as_tensor(tus), threshold)
    standard = {k: float(v) for k, v in compute_standard_metrics(tks, tus).items()}
    output = {
        "method": "CAC",
        "backbone": "plankton_net",
        "protocol": f"{args.val_unknown_split}; fixed score; validation-selected threshold",
        "cac_protocol": "paper",
        **checkpoint_provenance(args.checkpoint, ARCHIVED_CHECKPOINT_SHA256),
        "score": "euclidean",
        "score_formula": "min distance_to_class_centers",
        "centers": {"source": "train", "class_ids": class_ids.tolist()},
        "threshold_source": {
            "known_split": "validation", "unknown_split": args.val_unknown_split,
            "objective": "open_set_f1", "search": "exact",
        },
        "threshold": threshold,
        "paper_metrics": dict(zip(PAPER_METRIC_NAMES, map(float, paper))),
        "standard_metrics": standard,
        "scores": {
            "known": tks.tolist(),
            "unknown": tus.tolist(),
            "val": vks.tolist(),
            "val_unknown": vus.tolist(),
        },
        "tau_curve": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in compute_tau_curve(
                vks, vpred, vt.numpy(), vus, n_steps=500).items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f'Results written to {args.output}')


if __name__ == "__main__":
    main()
