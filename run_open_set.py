"""Run the ZooplanktonBench50 open-set evaluation from images."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from evaluation_common import check_new_result_path


ROOT = Path(__file__).resolve().parent
ARCHITECTURES = {"planktonnet": "plankton_net", "efficientnetv2_s": "tf_efficientnetv2_s.in1k"}
CASES = tuple(f"{method}_{backbone}" for method in ("cac", "openmax", "pmal", "isplanktonir")
              for backbone in ARCHITECTURES)
CAC_CASES = ("cac_planktonnet", "cac_efficientnetv2_s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", help="Repeat to select multiple runs")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/ZooplanktonBench50_OpenSet")
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint for a single CAC case")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/new_run/paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cases = list(dict.fromkeys(args.case or CASES))
    if args.checkpoint is not None and (len(cases) != 1 or cases[0] not in CAC_CASES):
        parser.error("--checkpoint requires one CAC --case")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cuda" and (not torch.cuda.is_available() or
                                  (target.index or 0) >= torch.cuda.device_count()):
        parser.error(f"Requested CUDA device is unavailable: {device}")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers must be nonnegative")
    data = args.data_root.resolve()
    split_names = ("train", "validation", "validation_unknow", "test_know", "test_unknow")
    for split in split_names:
        if not (data / split).is_dir():
            parser.error(f"Dataset split not found: {data / split}")
    output = args.output_dir.resolve()
    try:
        check_new_result_path(output)
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)

    def destination(relative_path):
        path = output / relative_path
        check_new_result_path(path)
        return path

    def run(script, *options):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, options)],
                       cwd=ROOT, check=True)

    batch = ["--device", device, "--batch-size", args.batch_size, "--workers", args.workers]
    dataset = ["--data-dir", data.parent, "--dataset", data.name]
    weights = ROOT / "weights"
    openmax_groups = {}
    for name in cases:
        method, backbone = name.split("_", 1)
        arch = ARCHITECTURES[backbone]
        checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else weights / f"{name}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"Evaluating {name} on {device}", flush=True)
        if name == "cac_planktonnet":
            run("eval_cac_planktonnet.py", *dataset, *batch, "--checkpoint", checkpoint,
                "--backbone-source", ROOT / "PlanktonNet.py",
                "--output", destination(f"{name}.json"))
        elif name == "cac_efficientnetv2_s":
            run("eval_cac_efficientnet.py", "--data-root", data, *batch, "--checkpoint", checkpoint,
                "--output", destination(f"{name}.json"))
        elif method == "openmax":
            openmax_output = destination(Path("openmax") / backbone)
            run("eval_all_methods.py", *dataset, *batch, "--methods", "openmax",
                "--om-model", arch, "--om-checkpoint", checkpoint, "--output-dir", openmax_output)
            rows = json.loads((openmax_output / "all_results.json").read_text(encoding="utf-8"))
            row = next(iter(rows.values()))
            if "error" in row:
                raise RuntimeError(f"{name}: {row['error']}")
            key = "openmax_plankton_net" if backbone == "planktonnet" else "openmax_efficientnetv2_s"
            openmax_groups[key] = row
        elif method == "pmal":
            filename = "pmal_planktonnet.json" if backbone == "planktonnet" else "pmal_efficientnet.json"
            run("eval_pmal.py", *dataset, *batch, "--arch", arch, "--checkpoint", checkpoint,
                "--output", destination(filename))
        else:
            folder = "isplanktonclr_plankton" if backbone == "planktonnet" else "isplanktonclr_efficientnet"
            run("eval_isplanktonir.py", "--data-root", data, *batch, "--arch", arch,
                "--checkpoint", checkpoint, "--output-dir", destination(folder))
    if openmax_groups:
        path = destination("openmax/all_results.json")
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(json.dumps(prior | openmax_groups, indent=2) + "\n", encoding="utf-8")
    if set(cases) == set(CASES):
        run("summarize_open_set.py", "--result-root", output)


if __name__ == "__main__":
    main()
