#!/usr/bin/env python3
"""Link the validation slices of every fold (from imagesTr, according to splits_final.json)
into valsets/fold_k/, so that nnUNetv2_predict can re-predict the internal validation
sets with --disable_tta.

nnU-Net's own fold_k/validation/ predictions are produced with mirroring TTA. The external
test set is predicted with a single fold model and TTA off; the internal validation sets
are re-predicted here under the same protocol.

Usage: python scripts/make_fold_val_dirs.py [--out valsets]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset501_CTSCOPE")
    ap.add_argument("--out", default="valsets")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking (Windows)")
    args = ap.parse_args()

    raw = Path(os.environ.get("nnUNet_raw", "")) / args.dataset
    pre = Path(os.environ.get("nnUNet_preprocessed", "")) / args.dataset
    splits_file = pre / "splits_final.json"
    if not splits_file.exists():
        sys.exit(f"{splits_file} not found")
    ending = json.loads((raw / "dataset.json").read_text(encoding="utf-8")).get("file_ending", ".png")
    splits = json.loads(splits_file.read_text(encoding="utf-8"))

    for k, fold in enumerate(splits):
        dst = Path(args.out) / f"fold_{k}"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for case in fold["val"]:
            src = raw / "imagesTr" / f"{case}_0000{ending}"
            if not src.exists():
                sys.exit(f"missing {src}")
            tgt = dst / src.name
            if tgt.exists() or tgt.is_symlink():
                tgt.unlink()
            if args.copy or os.name == "nt":
                shutil.copyfile(src, tgt)
            else:
                tgt.symlink_to(src.resolve())
            n += 1
        subj = sorted({c.split("_")[0] for c in fold["val"]})
        print(f"  fold_{k}: {n} validation slices, subjects {subj} -> {dst}")


if __name__ == "__main__":
    main()
