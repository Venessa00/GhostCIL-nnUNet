#!/usr/bin/env python3
"""Verify that splits_final.json separates subjects (no subject appears in both train and val of a fold)."""
import json
import os
import sys
from pathlib import Path

ds = sys.argv[1] if len(sys.argv) > 1 else "Dataset501_CTSCOPE"
pre = os.environ.get("nnUNet_preprocessed", "")
if not pre:
    sys.exit("nnUNet_preprocessed is not set")

p = Path(pre) / ds / "splits_final.json"
if not p.exists():
    sys.exit(f"{p} not found")

splits = json.loads(p.read_text())
print(f"  folds: {len(splits)}")
bad = False
for i, f in enumerate(splits):
    val_subj = sorted({c.split("_")[0] for c in f["val"]})
    tr_subj = sorted({c.split("_")[0] for c in f["train"]})
    overlap = set(val_subj) & set(tr_subj)
    print(f"  fold {i}: train {tr_subj} ({len(f['train'])} slices) / val {val_subj} ({len(f['val'])} slices)")
    if overlap:
        print(f"    x subjects shared between train and val: {sorted(overlap)}")
        bad = True
if bad:
    sys.exit("subject-level separation check FAILED; do not train")
print("  OK: subject-level separation verified")
