#!/usr/bin/env python3
"""Audit the labels in nnUNet_raw: per-slice foreground fraction; flag all-0 / all-1 / implausible masks.
Usage: python scripts/check_labels.py     # checks $nnUNet_raw/Dataset501_CTSCOPE/labelsTr and labelsTs
"""
import os, sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(os.environ.get("nnUNet_raw", "")) / "Dataset501_CTSCOPE"
bad, total = [], 0
for sub in ("labelsTr", "labelsTs"):
    d = root / sub
    if not d.is_dir():
        sys.exit(f"{d} not found")
    per_subj = {}
    for p in sorted(d.glob("*.png")):
        a = np.asarray(Image.open(p)); u = np.unique(a)
        fg = float((a > 0).mean()); total += 1
        per_subj.setdefault(p.name.split("_")[0], []).append(fg)
        if not set(u.tolist()) <= {0, 1} or fg == 0.0 or fg > 0.6:
            bad.append((sub, p.name, fg, u[:4].tolist()))
    print(f"== {sub}")
    for s, v in per_subj.items():
        print(f"  {s}: foreground fraction mean {np.mean(v)*100:5.1f}%  min {min(v)*100:5.1f}%  max {max(v)*100:5.1f}%  n={len(v)}")
if bad:
    for b in bad[:30]:
        print("  x", b)
    sys.exit(f"{len(bad)}/{total} labels are abnormal. Re-run scripts/prepare_data.py and inspect the listed files before training.")
print(f"OK: all {total} labels are binary 0/1 with a plausible foreground fraction")
