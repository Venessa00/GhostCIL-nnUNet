#!/usr/bin/env python3
"""Generate a small synthetic dataset with the folder layout of CT-SCOPE, so that the whole
pipeline (run/01 -> run/05) can be exercised on a machine without the real data or a GPU.
The results are meaningless; the purpose is to check that every step runs.

Layout (identical to BYO-UPM/Craneal_CT):
    <out>/Dataset/Labeled Data PNG/Internal Dataset/P01..P09/{Original CT,Manual Mask}/xxx_001.png
    <out>/Dataset/Labeled Data PNG/External Dataset/P13,P16,P30,P38/{Original CT,Manual Mask}/...

Usage: python tests/make_synthetic_ctscope.py --out ./synthetic_ct --size 96 --slices 4
       bash run/01_prepare.sh ./synthetic_ct
"""
import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as iio

INTERNAL = [f"P{i:02d}" for i in range(1, 10)]
EXTERNAL = ["P13", "P16", "P30", "P38"]


def make_slice(rng, size):
    yy, xx = np.mgrid[:size, :size]
    cy, cx = rng.uniform(0.35, 0.65, 2) * size
    r_out = rng.uniform(0.28, 0.4) * size
    r_in = r_out * rng.uniform(0.6, 0.8)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = ((d < r_out) & (d > r_in)).astype(np.uint8)          # a ring of "bone"
    img = 60 + 40 * rng.standard_normal((size, size)) + 150 * mask
    img += 30 * (d < r_in)                                        # slightly brighter interior
    return np.clip(img, 0, 255).astype(np.uint8), mask * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synthetic_ct")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--slices", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(args.out) / "Dataset" / "Labeled Data PNG"
    for split, subjects in (("Internal Dataset", INTERNAL), ("External Dataset", EXTERNAL)):
        for s in subjects:
            ct = root / split / s / "Original CT"
            mk = root / split / s / "Manual Mask"
            ct.mkdir(parents=True, exist_ok=True)
            mk.mkdir(parents=True, exist_ok=True)
            for i in range(1, args.slices + 1):
                img, mask = make_slice(rng, args.size)
                iio.imwrite(ct / f"{s}_slice_{i:03d}.png", img)
                iio.imwrite(mk / f"{s}_mask_{i:03d}.png", mask)
    print(f"synthetic data written to {args.out}: internal {len(INTERNAL)}x{args.slices}, "
          f"external {len(EXTERNAL)}x{args.slices} slices, {args.size}x{args.size}")


if __name__ == "__main__":
    main()
