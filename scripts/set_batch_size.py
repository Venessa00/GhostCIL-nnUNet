#!/usr/bin/env python3
"""Reduce the batch size chosen automatically by nnU-Net when GPU memory is insufficient.

    python scripts/set_batch_size.py --halve        # halve
    python scripts/set_batch_size.py --value 2      # set explicitly
    python scripts/set_batch_size.py                # show the current value

Both nnUNetPlans.json and (if present) nnUNetPlans_half.json are updated so that all
configurations share the same batch size. Change it before training starts: models
trained with different batch sizes are not comparable. The paper used batch size 2
for every configuration (11.6 GB GPU, 512x448 patches).
"""
import argparse
import json
import os
import sys
from pathlib import Path


def set_bs(p: Path, configuration: str, new):
    plans = json.loads(p.read_text(encoding="utf-8"))
    cfg = plans["configurations"][configuration]
    old = int(cfg["batch_size"])
    print(f"{p.name}: patch_size = {cfg['patch_size']}, batch_size = {old}")
    if new is None:
        return old
    target = max(2, old // 2) if new == "halve" else int(new)
    if target == old:
        print("  unchanged")
        return old
    backup = p.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  original saved as {backup.name}")
    cfg["batch_size"] = target
    p.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    print(f"  batch_size: {old} -> {target}")
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset501_CTSCOPE")
    ap.add_argument("--configuration", default="2d")
    ap.add_argument("--halve", action="store_true")
    ap.add_argument("--value", type=int)
    args = ap.parse_args()

    pre = os.environ.get("nnUNet_preprocessed", "")
    if not pre:
        sys.exit("nnUNet_preprocessed is not set")
    main_plans = Path(pre) / args.dataset / "nnUNetPlans.json"
    if not main_plans.exists():
        sys.exit(f"{main_plans} not found; run preprocessing first")

    new = args.value if args.value else ("halve" if args.halve else None)
    final = set_bs(main_plans, args.configuration, new)
    half = main_plans.with_name("nnUNetPlans_half.json")
    if half.exists():
        # keep the half-width plans aligned with the main plans
        set_bs(half, args.configuration, final if new is not None else None)
    if new is not None:
        print("NOTE: configurations trained with the previous batch size must be retrained.")


if __name__ == "__main__":
    main()
