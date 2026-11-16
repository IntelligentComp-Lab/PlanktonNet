#!/usr/bin/env python3
"""Redraw archived Figures 20 and 21 or plot a new paper-protocol run.

Figure 20 uses the frozen test scores. Figure 21 uses validation scores only.
The script rejects incomplete or mismatched result files before plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


SCRIPT = Path(__file__).resolve()
REVISION_ROOT = SCRIPT.parent


METHOD_ORDER = ("CAC", "OpenMax", "PMAL", "IsPlanktonCLR")
BACKBONE_ORDER = ("PlanktonNet", "EfficientNetV2-S")
DISPLAY_NAMES = {"IsPlanktonCLR": "IsPlanktonIR"}
COLORS = {
    "CAC": "#0072B2",
    "OpenMax": "#E69F00",
    "PMAL": "#D55E00",
    "IsPlanktonCLR": "#009E73",
}
EXPECTED_COUNTS = {"known": 1777, "unknown": 483, "val": 1772, "val_unknown": 320}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path,
        default=REVISION_ROOT / "results" / "reference")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REVISION_ROOT / "figures" / "open_set")
    return parser.parse_args()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data, *keys):
    for key in keys:
        data = data[key]
    return data


def load_entries(root):
    openmax = read_json(root / "openmax" / "all_results.json")
    paper_planktonnet = root / "cac_planktonnet.json"
    paper_efficientnet = root / "cac_efficientnetv2_s.json"
    paper = paper_planktonnet.is_file() or paper_efficientnet.is_file()
    sources = {
        ("CAC", "PlanktonNet"): (paper_planktonnet if paper else root / "cac_planktonnet_seed71.json", ()),
        ("CAC", "EfficientNetV2-S"): (
            (paper_efficientnet, ()) if paper else (root / "cac_effnet_msp_t100.json", ("results", "10"))),
        ("OpenMax", "PlanktonNet"): (
            root / "openmax" / "all_results.json", ("openmax_plankton_net",)),
        ("OpenMax", "EfficientNetV2-S"): (
            root / "openmax" / "all_results.json", ("openmax_efficientnetv2_s",)),
        ("PMAL", "PlanktonNet"): (root / "pmal_planktonnet.json", ()),
        ("PMAL", "EfficientNetV2-S"): (root / "pmal_efficientnet.json", ()),
        ("IsPlanktonCLR", "PlanktonNet"): (
            root / "isplanktonclr_plankton" / "scores_isplanktonclr_plankton_net.json", ()),
        ("IsPlanktonCLR", "EfficientNetV2-S"): (
            root / "isplanktonclr_efficientnet" /
            "scores_isplanktonclr_tf_efficientnetv2_s_in1k.json", ()),
    }
    entries = {}
    for key, (path, nested) in sources.items():
        data = openmax if path.name == "all_results.json" else read_json(path)
        entry = get_nested(data, *nested) if nested else data
        scores = entry.get("scores")
        tau_curve = entry.get("tau_curve")
        if not scores or not tau_curve:
            raise ValueError(f"Missing scores/tau_curve: {path} {nested}")
        arrays = {name: np.asarray(scores[name], dtype=np.float64) for name in EXPECTED_COUNTS}
        for split, expected in EXPECTED_COUNTS.items():
            if arrays[split].size != expected:
                raise ValueError(
                    f"{key} {split}: expected {expected}, got {arrays[split].size}")
            if not np.isfinite(arrays[split]).all():
                raise ValueError(f"Non-finite scores: {key} {split}")
        labels = np.r_[np.zeros(arrays["known"].size), np.ones(arrays["unknown"].size)]
        values = np.r_[arrays["known"], arrays["unknown"]]
        auc = float(roc_auc_score(labels, values))
        aupr = float(average_precision_score(labels, values))
        stored = entry.get("standard_metrics", {})
        if "auroc" in stored and not np.isclose(auc, float(stored["auroc"]), atol=2e-6):
            raise ValueError(f"AUROC mismatch for {key}: computed={auc}, stored={stored['auroc']}")
        if "aupr" in stored and not np.isclose(aupr, float(stored["aupr"]), atol=2e-6):
            raise ValueError(f"AUPR mismatch for {key}: computed={aupr}, stored={stored['aupr']}")
        entries[key] = {
            "path": str(path.resolve()), "scores": arrays, "tau_curve": tau_curve,
            "auc": auc, "aupr": aupr,
            "threshold": float(entry.get("best_threshold", entry.get("threshold"))),
        }
        if key[0] == "CAC":
            cac_protocol = data.get("cac_protocol", "archived")
            expected = "paper" if paper else "archived"
            if cac_protocol != expected:
                raise ValueError(f"Unexpected CAC protocol in {path}: {cac_protocol}")
            entries[key]["cac_protocol"] = cac_protocol
    return entries


def fpr95_from_curve(fpr, tpr):
    candidates = np.flatnonzero(tpr >= 0.95)
    return float(fpr[candidates[0]]) if candidates.size else 1.0


def draw_figure20(entries, output_dir, filename="Figure_20"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), sharex=True, sharey=True)
    for ax, backbone in zip(axes, BACKBONE_ORDER):
        for method in METHOD_ORDER:
            record = entries[(method, backbone)]
            known = record["scores"]["known"]
            unknown = record["scores"]["unknown"]
            labels = np.r_[np.zeros(known.size), np.ones(unknown.size)]
            values = np.r_[known, unknown]
            fpr, tpr, _ = roc_curve(labels, values)
            fpr95 = fpr95_from_curve(fpr, tpr)
            name = "CAC (paper score)" if record.get("cac_protocol") == "paper" else DISPLAY_NAMES.get(method, method)
            ax.plot(
                fpr, tpr, lw=2.5, color=COLORS[method],
                label=f"{name} (AUROC={record['auc']:.4f})")
            ax.scatter([fpr95], [0.95], s=28, color=COLORS[method], zorder=4)
        ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1.2, label="Random")
        ax.set_title(backbone, fontsize=20, fontweight="bold", pad=10)
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, color="#d9d9d9", lw=0.8, alpha=0.7)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            frameon=True, fontsize=14, ncol=2, borderaxespad=0.0)
    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(-0.01, 1.01)
    fig.tight_layout(rect=(0.02, 0.18, 1, 0.98), w_pad=2.5)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{filename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_figure21(entries, output_dir, filename="Figure_21"):
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8))
    metrics = (
        ("known_th", "Known (Th)", "#0072B2"),
        ("unknown_th", "Unknown (Th)", "#D55E00"),
        ("os_acc", "Open-Set Acc", "#009E73"),
        ("f_score", "Open-Set F1", "#CC79A7"),
    )
    shared_handles = None
    shared_labels = None
    for ax, method in zip(axes.flat, METHOD_ORDER):
        record = entries[(method, "PlanktonNet")]
        curve = record["tau_curve"]
        thresholds = np.asarray(curve["thresholds"], dtype=np.float64)
        if thresholds.ndim != 1 or thresholds.size < 2:
            raise ValueError(f"Invalid tau curve for {method}")
        panel_handles = []
        panel_labels = []
        for key, label, color in metrics:
            values = np.asarray(curve[key], dtype=np.float64)
            if values.shape != thresholds.shape:
                raise ValueError(f"Tau curve length mismatch for {method}: {key}")
            line, = ax.plot(
                thresholds, 100.0 * values, label=label, color=color, lw=2.1)
            panel_handles.append(line)
            panel_labels.append(label)
        tau = record["threshold"]
        ax.axvline(tau, color="#333333", ls="--", lw=1.5)
        display_name = DISPLAY_NAMES.get(method, method)
        if record.get("cac_protocol") == "paper":
            display_name = "CAC (paper score)"
        ax.set_title(f"{display_name} (τ={tau:.4g})", fontsize=20, fontweight="bold")
        ax.set_xlabel("Rejection threshold τ", fontsize=20)
        ax.set_ylabel("Metric (%)", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylim(0, 101)
        ax.grid(True, color="#d9d9d9", lw=0.8, alpha=0.7)
        if shared_handles is None:
            shared_handles = panel_handles
            shared_labels = panel_labels
    fig.legend(
        shared_handles, shared_labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.008), ncol=4, fontsize=15, frameon=True)
    fig.tight_layout(rect=(0.02, 0.075, 1, 0.98), h_pad=2.3, w_pad=2.0)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{filename}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to((REVISION_ROOT / "results/reference").resolve()):
        raise ValueError("Choose an output outside results/reference")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.result_root.resolve())
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cac_protocol = entries[("CAC", "PlanktonNet")]["cac_protocol"]
    if cac_protocol == "paper":
        roc_name, threshold_name, audit_name = "paper_cac_roc", "paper_cac_thresholds", "paper_cac_plot_audit.json"
    else:
        roc_name, threshold_name, audit_name = "Figure_20", "Figure_21", "figure_data_audit.json"
    draw_figure20(entries, output_dir, roc_name)
    draw_figure21(entries, output_dir, threshold_name)
    audit = {
        "protocol": "10 class-disjoint unknown validation categories",
        "cac_protocol": cac_protocol,
        "counts": EXPECTED_COUNTS,
        "figure_20_data": "frozen test known/test unknown scores",
        "figure_21_data": "validation known/validation unknown scores only",
        "entries": {
            f"{DISPLAY_NAMES.get(method, method)}+{backbone}": {
                "source": record["path"], "threshold": record["threshold"],
                "auroc": record["auc"], "aupr": record["aupr"],
            }
            for (method, backbone), record in entries.items()
        },
    }
    (output_dir / audit_name).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created: {output_dir / (roc_name + '.pdf')}")
    print(f"Created: {output_dir / (threshold_name + '.pdf')}")
    print(f"Audit:   {output_dir / audit_name}")


if __name__ == "__main__":
    main()
