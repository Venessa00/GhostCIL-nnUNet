#!/usr/bin/env python3
"""Read the radial gains learned by the frequency branch of the attention module
(16 values per encoder stage) directly from trained checkpoints.

No network is built and nnU-Net is not needed: only state_dict keys ending in
``freq.filt.gain`` are read. Output: results/radial_gain.json.

Usage: python scripts/dump_radial_gain.py            # scans $nnUNet_results for all *GMA* models
       python scripts/dump_radial_gain.py --ckpt a.pth b.pth
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch


def read_gains(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("network_weights", obj.get("state_dict", obj))
    gains = {}
    for k, v in sd.items():
        k2 = k.replace("_orig_mod.", "").replace("module.", "")
        # nnU-Net's decoder holds a reference to the encoder: decoder.encoder.* duplicates encoder.*
        if k2.startswith("decoder.encoder."):
            continue
        if k2.endswith("freq.filt.gain"):
            stage = k2.replace(".attention.freq.filt.gain", "")
            gains[stage] = [round(float(x), 4) for x in v.flatten()]
    return gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*")
    ap.add_argument("--out", default="results/radial_gain.json")
    args = ap.parse_args()

    ckpts = args.ckpt or sorted(glob.glob(os.path.join(
        os.environ.get("nnUNet_results", ""), "Dataset501_CTSCOPE",
        "*GMA*", "fold_*", "checkpoint_final.pth")))
    if not ckpts:
        print("no checkpoints found (nnUNet_results not set or nothing trained yet)")
        return

    result = {}
    for c in ckpts:
        parts = Path(c).parts
        name = f"{parts[-3]}/{parts[-2]}"
        g = read_gains(c)
        result[name] = g
        print(name)
        for stage, vec in g.items():
            hi = sum(vec[len(vec) // 2:]) / (len(vec) - len(vec) // 2)
            lo = sum(vec[: len(vec) // 2]) / (len(vec) // 2)
            print(f"  {stage:24s} low-frequency mean {lo:.3f}  high-frequency mean {hi:.3f}  "
                  f"{'high-pass' if hi > lo else 'low-pass'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} (x axis: normalised frequency 0 -> 1 in 16 bins)")


if __name__ == "__main__":
    main()
