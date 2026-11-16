#!/usr/bin/env python3
"""Summarize the eight configurations of an open-set run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    for relative_path in ("results/reference", "results/new_run/archived"):
        if root.is_relative_to((Path(__file__).resolve().parent / relative_path).resolve()):
            parser.error(f"Choose a result root outside {relative_path}")
    openmax = load(root / "openmax" / "all_results.json")
    eff = load(root / "cac_efficientnetv2_s.json")
    planktonnet = load(root / "cac_planktonnet.json")
    if any(result.get("cac_protocol") != "paper" for result in (eff, planktonnet)):
        parser.error("Expected CAC results from the paper's inference method")
    rows = [
        ("CAC", "EfficientNetV2-S", eff),
        ("OpenMax", "PlanktonNet", openmax["openmax_plankton_net"]),
        ("OpenMax", "EfficientNetV2-S", openmax["openmax_efficientnetv2_s"]),
        ("PMAL", "PlanktonNet", load(root / "pmal_planktonnet.json")),
        ("PMAL", "EfficientNetV2-S", load(root / "pmal_efficientnet.json")),
        ("IsPlanktonIR", "PlanktonNet", load(root / "isplanktonclr_plankton" / "scores_isplanktonclr_plankton_net.json")),
        ("IsPlanktonIR", "EfficientNetV2-S", load(root / "isplanktonclr_efficientnet" / "scores_isplanktonclr_tf_efficientnetv2_s_in1k.json")),
        ("CAC", "PlanktonNet", planktonnet),
    ]
    headers = ["Method", "Backbone", "Known (Closed)", "Known (Th)", "Unknown (Th)",
               "Open-Set Acc", "Open-Set F1", "AUROC", "AUPR", "FPR95"]
    table = ["| " + " | ".join(headers) + " |", "|---|---|" + "---:|" * 8]
    machine_rows = []
    for method, backbone, result in rows:
        paper = result["paper_metrics"]
        standard = result["standard_metrics"]
        values = [paper[key] * 100 for key in
                  ("known_closed", "known_th", "unknown_th", "os_acc", "f_score")]
        values += [standard[key] * 100 for key in ("auroc", "aupr", "fpr95")]
        table.append("| " + " | ".join([method, backbone] + [f"{x:.2f}" for x in values]) + " |")
        row = {"method": method, "backbone": backbone, "paper_metrics": paper, "standard_metrics": standard}
        if method == "CAC":
            row["cac_protocol"] = "paper"
        machine_rows.append(row)
    (root / "OPEN_SET_TABLE.md").write_text(
        "# Open-set run with paper CAC inference\n\n" + "\n".join(table) + "\n",
        encoding="utf-8",
    )
    (root / "open_set_results.json").write_text(
        json.dumps(machine_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
