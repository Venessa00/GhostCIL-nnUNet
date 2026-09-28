#!/usr/bin/env python3
"""Convert the CT-SCOPE PNG data to the nnU-Net v2 raw format and create subject-level splits.

Input (BYO-UPM/Craneal_CT repository):
    Dataset/Labeled Data PNG/Internal Dataset/P01..P09/{Original CT,Manual Mask}
    Dataset/Labeled Data PNG/External Dataset/P13,P16,P30,P38/{Original CT,Manual Mask}

Output:
    $nnUNet_raw/Dataset501_CTSCOPE/{imagesTr,labelsTr,imagesTs,labelsTs,dataset.json}
    ./splits_final.json      (3 folds split by subject; copied into the preprocessed folder later)

Usage:
    python scripts/prepare_data.py --src /path/to/Craneal_CT --dataset-id 501
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import imageio.v3 as iio

INTERNAL = [f"P{i:02d}" for i in range(1, 10)]          # P01..P09
EXTERNAL = ["P13", "P16", "P30", "P38"]


def read_gray(path: Path) -> np.ndarray:
    a = iio.imread(path)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)      # RGB/RGBA CT images have identical channels; alpha ignored
    return a


def read_mask(path: Path) -> np.ndarray:
    """Read a CT-SCOPE mask as a 0/1 array.

    The mask files come in mixed encodings (8-bit grey, 16-bit grey, RGB, RGBA) and the
    background of the RGBA masks is not 0 but small values (about 3-8). Therefore:
    (1) ignore the alpha channel and take the maximum over RGB, (2) binarise at half of
    the value range. Binarising with "> 0" turns such slices into all-foreground labels.
    """
    a = np.asarray(iio.imread(path))
    if a.ndim == 3:
        a = a[..., :3].max(axis=-1)
    a = a.astype(np.float64)
    thr = 32767.5 if a.max() > 255 else 127.5   # 16-bit masks: half of the 16-bit range
    return (a > thr).astype(np.uint8)


def to_uint8_image(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def collect_pairs(subject_dir: Path):
    ct_dir = subject_dir / "Original CT"
    mask_dir = subject_dir / "Manual Mask"
    if not ct_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"{subject_dir} lacks 'Original CT' or 'Manual Mask'")

    def index_of(p: Path):
        m = re.search(r"_(\d+)\.png$", p.name, flags=re.IGNORECASE)
        return m.group(1) if m else None

    masks = {}
    for p in sorted(mask_dir.glob("*.png")):
        k = index_of(p)
        if k:
            masks[k] = p

    pairs = []
    for p in sorted(ct_dir.glob("*.png")):
        k = index_of(p)
        if k and k in masks:
            pairs.append((k, p, masks[k]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the Craneal_CT repository")
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--dataset-name", default="CTSCOPE")
    ap.add_argument("--out", default=None, help="defaults to the nnUNet_raw environment variable")
    args = ap.parse_args()

    raw_root = Path(args.out or os.environ.get("nnUNet_raw", ""))
    if not str(raw_root):
        sys.exit("export nnUNet_raw=... or pass --out")

    labeled = Path(args.src) / "Dataset" / "Labeled Data PNG"
    if not labeled.is_dir():
        sys.exit(f"{labeled} not found; --src must point to the root of the Craneal_CT repository")

    ds_dir = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds_dir / sub).mkdir(parents=True, exist_ok=True)

    per_subject_cases = defaultdict(list)
    n_tr = n_ts = 0
    fg_stats, bad = {}, []          # label audit: foreground fraction of every mask

    for split_name, subjects, img_dir, lab_dir in (
        ("Internal Dataset", INTERNAL, "imagesTr", "labelsTr"),
        ("External Dataset", EXTERNAL, "imagesTs", "labelsTs"),
    ):
        for subj in subjects:
            sdir = labeled / split_name / subj
            if not sdir.is_dir():
                sys.exit(f"missing subject directory: {sdir}")
            pairs = collect_pairs(sdir)
            if not pairs:
                sys.exit(f"no matching slice/mask pairs under {sdir}")
            for idx, ct_path, mask_path in pairs:
                case = f"{subj}_{idx}"
                img = to_uint8_image(read_gray(ct_path))
                lab = read_mask(mask_path)
                fg = float(lab.mean())
                fg_stats.setdefault(subj, []).append(fg)
                if fg == 0.0 or fg > 0.6:
                    bad.append((case, fg))
                if img.shape != lab.shape:
                    sys.exit(f"{case}: image and mask sizes differ: {img.shape} vs {lab.shape}")
                iio.imwrite(ds_dir / img_dir / f"{case}_0000.png", img)
                iio.imwrite(ds_dir / lab_dir / f"{case}.png", lab)
                if img_dir == "imagesTr":
                    per_subject_cases[subj].append(case)
                    n_tr += 1
                else:
                    n_ts += 1
            print(f"  {split_name}/{subj}: {len(pairs)} slices")

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": n_tr,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "description": "CT-SCOPE osseous structures surrounding the paranasal sinuses "
                       "(Sun et al., Data in Brief 2025). Internal subset = train/val, "
                       "external subset = independent cross-center test.",
    }
    (ds_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- subject-level 3-fold split ----
    folds = [INTERNAL[0:3], INTERNAL[3:6], INTERNAL[6:9]]
    splits = []
    for val_subjects in folds:
        val_cases, train_cases = [], []
        for subj, cases in per_subject_cases.items():
            (val_cases if subj in val_subjects else train_cases).extend(cases)
        splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})
        print(f"  fold: validation subjects {val_subjects} -> train {len(train_cases)} / val {len(val_cases)}")

    Path("splits_final.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8")

    print("\nLabel audit (foreground fraction in %; expected roughly 3-25; 0 or >60 indicates a reading error):")
    for subj, v in fg_stats.items():
        print(f"  {subj}: mean {np.mean(v)*100:5.1f}  min {min(v)*100:5.1f}  max {max(v)*100:5.1f}  n={len(v)}")
    if bad:
        for case, fg in bad[:20]:
            print(f"  x {case}: foreground fraction {fg*100:.1f}%")
        sys.exit(f"{len(bad)} abnormal masks; stopping. Inspect the files listed above before training.")
    print("  OK: all masks have a plausible foreground fraction")
    print(f"\nDone: {n_tr} training/validation slices, {n_ts} external test slices")
    print(f"dataset folder: {ds_dir}")
    print("splits_final.json was written to the current directory; copy it into "
          f"$nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/ after preprocessing")
    if n_tr != 576 or n_ts != 120:
        print(f"WARNING: expected 576/120 slices, found {n_tr}/{n_ts}; check the dataset")


if __name__ == "__main__":
    main()
