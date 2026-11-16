"""Data loading, checkpoint validation, and scoring for open-set evaluation."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from timm.data import create_dataset, create_loader
from tqdm import tqdm
from cacUtils import CACModel
from osr_metrics import compute_paper_metrics, compute_standard_metrics, search_best_threshold

REVISION_ROOT = Path(__file__).resolve().parent
PAPER_METRIC_NAMES = ("known_closed", "known_th", "unknown_th", "os_acc", "f_score")


def make_loader(
    root: Path,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    class_map: dict[str, int] | None = None,
):
    dataset = create_dataset(
        name="ImageFolder",
        root=str(root / split),
        is_training=False,
        class_map=class_map,
        batch_size=batch_size,
    )
    loader = create_loader(
        dataset,
        (3, 224, 224),
        batch_size,
        is_training=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bicubic",
        num_workers=workers,
        device=device,
        persistent_workers=workers > 0,
    )
    return loader, dataset


def get_class_map(dataset) -> dict[str, int]:
    direct = getattr(dataset, "class_to_idx", None)
    if direct is not None:
        return direct
    for attribute in ("parser", "reader"):
        obj = getattr(dataset, attribute, None)
        mapping = getattr(obj, "class_to_idx", None) if obj is not None else None
        if mapping is not None:
            return mapping
    raise RuntimeError("Could not obtain ImageFolder class_to_idx")


def check_unknown_classes(class_map, validation_dataset, test_dataset):
    """Require disjoint known, unknown validation, and unknown test classes."""
    validation = set(get_class_map(validation_dataset))
    test = set(get_class_map(test_dataset))
    overlap = (validation & test) | ((validation | test) & set(class_map))
    if overlap:
        raise ValueError(f"Open-set class splits overlap: {', '.join(sorted(overlap))}")


def training_class_centers(features, targets):
    """Return the mean training embedding for each known class, ordered by label."""
    class_ids = targets.unique(sorted=True)
    centers = torch.stack([features[targets == label].mean(dim=0) for label in class_ids])
    return class_ids, centers


def score_cac_features(features, centers, class_ids):
    """Return the Euclidean distance and class label of the nearest training center."""
    distances = torch.cdist(features.float(), centers.float(), p=2)
    scores, nearest = distances.min(dim=1)
    return scores, class_ids[nearest]


def check_new_result_path(path):
    destination = path.resolve()
    for relative_path in ("results/reference", "results/new_run/archived"):
        if destination.is_relative_to((REVISION_ROOT / relative_path).resolve()):
            raise ValueError(f"Choose an output outside {relative_path}")


def create_exact_planktonnet(backbone_source: Path):
    """Instantiate the PlanktonNet backbone from the specified Python file."""
    source_hash = hashlib.sha256(str(backbone_source.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"cac_exact_checkpoint_planktonnet_{source_hash}"
    spec = importlib.util.spec_from_file_location(module_name, backbone_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(backbone_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.PlanktonNet(pretrained=False)


def load_model(
    checkpoint: Path, backbone_source: Path, device: torch.device
) -> CACModel:
    """Load a CAC checkpoint and require every model parameter and buffer to match."""
    backbone = create_exact_planktonnet(backbone_source)
    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_object.get("state_dict", checkpoint_object)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    num_classes, feature_dim = state_dict["classifier.weight"].shape
    model = CACModel(backbone, feature_dim=feature_dim, num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is not an exact match: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: CACModel,
    loader,
    device: torch.device,
    description: str,
    feature_space: str = "backbone",
):
    """Collect backbone embeddings or CAC logits and labels on the CPU."""
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for inputs, targets in tqdm(loader, desc=description):
        if feature_space == "cac_projection":
            features = model(inputs.to(device))
        else:
            features = model.backbone(inputs.to(device))
        if features.ndim > 2:
            features = features.flatten(1)
        feature_batches.append(features.float().cpu())
        label_batches.append(targets.cpu())
    return torch.cat(feature_batches), torch.cat(label_batches)


def metrics_for_validation(
    known_scores: np.ndarray,
    known_predictions: np.ndarray,
    known_targets: np.ndarray,
    unknown_scores: np.ndarray,
    *,
    n_steps: int | None = 2000,
) -> dict[str, Any]:
    """Select the rejection threshold that maximizes open-set F1 on validation data."""
    threshold, paper_tuple = search_best_threshold(
        known_scores, known_predictions, known_targets, unknown_scores, n_steps=n_steps
    )
    standard = compute_standard_metrics(known_scores, unknown_scores)
    return {
        "best_threshold": float(threshold),
        "paper_metrics": dict(
            zip(PAPER_METRIC_NAMES, [float(value) for value in paper_tuple])
        ),
        "standard_metrics": {key: float(value) for key, value in standard.items()},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(path, archived_sha256):
    digest = sha256_file(path)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_source": (
            "bundled_archived_checkpoint" if digest == archived_sha256 else "provided_checkpoint"
        ),
    }
