"""Estimate the FP16 state size reported for Table 9."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Bundled .pth file with a state_dict")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(tensor) for tensor in state.values()):
        parser.error("Expected a model state dictionary with tensors")

    element_count = sum(tensor.numel() for tensor in state.values())
    mebibyte = 1024**2
    print(f"Checkpoint file size: {args.checkpoint.stat().st_size / mebibyte:.2f} MiB")
    print(f"FP16 tensor size (two bytes per element): {element_count * 2 / mebibyte:.2f} MiB")


if __name__ == "__main__":
    main()
