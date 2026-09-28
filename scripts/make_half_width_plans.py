#!/usr/bin/env python3
"""Create the plans file for the half-width nnU-Net control: every stage keeps half of
its channels, everything else is unchanged.

This is the simplest possible lightweight baseline (roughly 1/4 of the parameters and
FLOPs of nnU-Net). The data_identifier is kept, so the existing preprocessed data are
reused and no additional preprocessing is needed.

Usage (after preprocessing):
    python scripts/make_half_width_plans.py                 # writes nnUNetPlans_half.json
    nnUNetv2_train 501 2d 0 -tr nnUNetTrainerBaseline150 -p nnUNetPlans_half
    nnUNetv2_predict ... -p nnUNetPlans_half
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path


def halve_plans(plans: dict, configuration: str, new_name: str, min_features: int = 8) -> dict:
    new = copy.deepcopy(plans)
    new["plans_name"] = new_name
    cfg = new["configurations"][configuration]

    if "architecture" in cfg and "arch_kwargs" in cfg["architecture"]:      # nnU-Net >= 2.2
        ak = cfg["architecture"]["arch_kwargs"]
        old = list(ak["features_per_stage"])
        ak["features_per_stage"] = [max(min_features, f // 2) for f in old]
        changed = ("features_per_stage", old, ak["features_per_stage"])
    elif "UNet_base_num_features" in cfg:                                    # nnU-Net 2.0/2.1
        old = int(cfg["UNet_base_num_features"])
        cfg["UNet_base_num_features"] = max(min_features, old // 2)
        if "unet_max_num_features" in cfg:
            cfg["unet_max_num_features"] = max(min_features, int(cfg["unet_max_num_features"]) // 2)
        changed = ("UNet_base_num_features", old, cfg["UNet_base_num_features"])
    else:
        raise KeyError("plans contain neither architecture.arch_kwargs nor UNet_base_num_features; "
                       "unsupported nnU-Net version")
    return new, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset501_CTSCOPE")
    ap.add_argument("--configuration", default="2d")
    ap.add_argument("--src-plans", default="nnUNetPlans")
    ap.add_argument("--new-plans", default="nnUNetPlans_half")
    args = ap.parse_args()

    pre = os.environ.get("nnUNet_preprocessed", "")
    if not pre:
        sys.exit("nnUNet_preprocessed is not set")
    src = Path(pre) / args.dataset / f"{args.src_plans}.json"
    if not src.exists():
        sys.exit(f"{src} not found; run preprocessing first")
    dst = src.with_name(f"{args.new_plans}.json")

    plans = json.loads(src.read_text(encoding="utf-8"))
    new, (key, old, val) = halve_plans(plans, args.configuration, args.new_plans)
    dst.write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(f"wrote {dst}")
    print(f"  {key}: {old} -> {val}")
    print(f"  data_identifier unchanged ({new['configurations'][args.configuration].get('data_identifier')}); preprocessed data are reused")
    print(f"train with: nnUNetv2_train 501 {args.configuration} <fold> -tr nnUNetTrainerBaseline150 -p {args.new_plans}")


if __name__ == "__main__":
    main()
