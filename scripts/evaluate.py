#!/usr/bin/env python3
"""Per-slice Dice / IoU / HD95, aggregation by subject and by fold, and paired tests.

Source of paper Tables 1, 3 and 4; the statistical protocol corresponds to Section 2.5.

Protocol
  * The external test set is predicted by each fold model separately (no ensembling,
    no TTA); the paper reports mean +/- SD over the three fold models.
  * Slices of one subject are not independent samples, therefore:
      - internal cross-validation: every subject is a validation subject exactly once;
        its validation slices are averaged, giving 9 subject-level paired samples for a
        subject-level paired Wilcoxon signed-rank test (n = 9);
      - external test: only 4 subjects, so a subject-level test has no power. A
        slice-level Wilcoxon test (n = 120, each slice = mean over the 3 fold models) is
        reported for reference together with the direction of the difference in each
        of the 4 subjects.

Usage
    # one method, one fold
    python scripts/evaluate.py --pred preds/Ghost_CIL-nnUNet_ext_f0 --gt $nnUNet_raw/Dataset501_CTSCOPE/labelsTs \\
                               --name Ghost_CIL-nnUNet --fold 0 --out results/ext_Ghost_CIL-nnUNet__fold0.json

    # aggregation + paired tests (external test set)
    python scripts/evaluate.py --compare "results/ext_*.json" --baseline nnU-Net_v2 --level external
    # internal cross-validation (subject level)
    python scripts/evaluate.py --compare "results/int_*.json" --baseline nnU-Net_v2 --level internal
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.stats import wilcoxon

METRICS = ("dice", "iou", "hd95")


def read_binary(path: Path) -> np.ndarray:
    import imageio.v3 as iio            # only needed for metric computation, not for aggregation
    a = iio.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 0).astype(np.uint8)


def surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(bool)
    return mask.astype(bool) ^ binary_erosion(mask.astype(bool))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> float:
    sp, sg = surface(pred), surface(gt)
    if sp.sum() == 0 and sg.sum() == 0:
        return 0.0
    if sp.sum() == 0 or sg.sum() == 0:
        return float("nan")            # undefined when one side is empty; counted separately
    dt_g = distance_transform_edt(~sg)
    dt_p = distance_transform_edt(~sp)
    d_pg = dt_g[sp]                    # predicted surface -> reference surface
    d_gp = dt_p[sg]
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95)) * spacing


def per_case_metrics(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> dict:
    inter = float((pred & gt).sum())
    ps, gs = float(pred.sum()), float(gt.sum())
    dice = (2 * inter) / (ps + gs) if (ps + gs) > 0 else 1.0
    union = ps + gs - inter
    iou = inter / union if union > 0 else 1.0
    return dict(dice=dice, iou=iou, hd95=hd95(pred, gt, spacing))


def evaluate_dir(pred_dir: Path, gt_dir: Path, spacing: float = 1.0) -> list:
    gts = {p.stem: p for p in sorted(gt_dir.glob("*.png"))}
    rows, missing = [], []
    for case, gt_path in gts.items():
        pp = pred_dir / f"{case}.png"
        if not pp.exists():
            missing.append(case)
            continue
        pred, gt = read_binary(pp), read_binary(gt_path)
        if gt.mean() > 0.6 or gt.sum() == 0:
            print(f"  WARNING {case}: reference foreground fraction {gt.mean()*100:.1f}%; the label may be corrupt "
                  f"(see the mask-format note in scripts/prepare_data.py)")
        if pred.shape != gt.shape:
            raise ValueError(f"{case}: prediction and reference sizes differ {pred.shape} vs {gt.shape}")
        m = per_case_metrics(pred, gt, spacing)
        m["case"] = case
        m["subject"] = case.split("_")[0]
        rows.append(m)
    if missing and len(missing) == len(gts):
        raise FileNotFoundError(f"no predictions found in {pred_dir}")
    if missing:
        # an internal validation folder only holds the validation subjects of its fold
        print(f"  ({len(missing)} cases have no prediction in {pred_dir.name}; skipped)")
    return rows


def mean_sd(values):
    v = np.array(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def fmt(m, s, k):
    if k == "hd95":
        return f"{m:.2f}±{s:.2f}"
    return f"{m:.4f}±{s:.4f}"


def parse_name_fold(path: str, d: dict):
    name = d.get("name")
    fold = d.get("fold")
    if fold is None:
        mm = re.search(r"__fold(\d+)", Path(path).stem)
        fold = int(mm.group(1)) if mm else 0
    return name, int(fold)


def load_results(patterns):
    files = sorted(set(sum([glob.glob(p) for p in patterns], [])))
    by_method = defaultdict(dict)          # name -> fold -> {case: row}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        name, fold = parse_name_fold(f, d)
        by_method[name][fold] = {r["case"]: r for r in d["cases"]}
    return by_method


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.all(a == b):
        return float("nan"), len(a)
    return float(wilcoxon(a, b).pvalue), len(a)


def compare_external(by_method, baseline):
    """External set: per-slice mean over folds; fold mean +/- SD, slice-level Wilcoxon, per-subject direction."""
    print("== External test set (4 independent subjects, 120 slices; each fold model predicts separately, no ensembling) ==")
    per_slice = {}                     # name -> case -> {metric: fold-mean}
    fold_means = {}                    # name -> metric -> [fold-level slice means]
    for name, folds in by_method.items():
        cases = sorted(set.intersection(*[set(c) for c in folds.values()]))
        ps = {}
        for c in cases:
            ps[c] = {k: float(np.nanmean([folds[f][c][k] for f in folds])) for k in METRICS}
            ps[c]["subject"] = c.split("_")[0]
        per_slice[name] = ps
        fold_means[name] = {k: [mean_sd(folds[f][c][k] for c in cases)[0] for f in folds]
                            for k in METRICS}
        n_nan = sum(1 for c in cases if not np.isfinite(ps[c]["hd95"]))
        if n_nan:
            print(f"  note: {name} has {n_nan} slices with undefined HD95 (empty prediction or reference); excluded from HD95")

    print(f"\n{'method':22s} {'folds':>5s} {'Dice (fold mean±SD)':>20s} {'IoU':>16s} {'HD95':>14s}")
    for name in by_method:
        line = f"{name:22s} {len(by_method[name]):5d}"
        for k in METRICS:
            m, s = mean_sd(fold_means[name][k])
            line += f"  {fmt(m, s, k):>16s}"
        print(line)

    print("\nDice per subject (mean over the 3 fold models):")
    subjects = sorted({r["subject"] for r in next(iter(per_slice.values())).values()})
    print(f"{'method':22s}" + "".join(f"{s:>10s}" for s in subjects))
    for name, ps in per_slice.items():
        vals = [np.mean([r["dice"] for r in ps.values() if r["subject"] == s]) for s in subjects]
        print(f"{name:22s}" + "".join(f"{v:10.4f}" for v in vals))

    if baseline not in per_slice:
        print(f"\n(no results for baseline {baseline}; tests skipped)")
        return
    base = per_slice[baseline]
    print(f"\nPaired tests against {baseline}: slice-level Wilcoxon (n≈120; slices are not independent, "
          "for reference only) + direction of the difference in each of the 4 subjects")
    for name, ps in per_slice.items():
        if name == baseline:
            continue
        shared = sorted(set(base) & set(ps))
        out = [name.ljust(22)]
        for k in METRICS:
            a = [base[c][k] for c in shared]
            b = [ps[c][k] for c in shared]
            p, n = paired_wilcoxon(a, b)
            diffs = []
            for s in subjects:
                cs = [c for c in shared if base[c]["subject"] == s]
                d = np.nanmean([ps[c][k] for c in cs]) - np.nanmean([base[c][k] for c in cs])
                diffs.append(d)
            better = sum(1 for d in diffs if (d > 0) if k != "hd95") + \
                sum(1 for d in diffs if (d < 0) if k == "hd95")
            ptxt = "P=  nan " if not np.isfinite(p) else f"P={p:.4f}{'*' if p < 0.05 else ' '}"
            out.append(f"{k}: {ptxt} better than baseline in {better}/{len(subjects)} subjects")
        print("  " + "  ".join(out))
    print("  * slice-level P<0.05. A method is only called better when P<0.05 and the subject-level directions agree.")


def compare_internal(by_method, baseline):
    """Internal CV: every subject is validated once -> 9 subject-level paired samples."""
    print("== Internal subject-level 3-fold cross-validation (9 subjects) ==")
    subj_level = {}                    # name -> subject -> {metric}
    slice_level = {}                   # name -> metric -> [fold-level slice means]
    for name, folds in by_method.items():
        subj = defaultdict(lambda: defaultdict(list))
        for f, cases in folds.items():
            for c, r in cases.items():
                for k in METRICS:
                    subj[r["subject"]][k].append(r[k])
        subj_level[name] = {s: {k: float(np.nanmean(v[k])) for k in METRICS}
                            for s, v in subj.items()}
        slice_level[name] = {k: [mean_sd(r[k] for r in cases.values())[0]
                                 for f, cases in sorted(folds.items())] for k in METRICS}
        if len(subj_level[name]) != 9:
            print(f"  note: {name} covers {len(subj_level[name])} subjects instead of 9 "
                  "(expected for single-fold ablations; otherwise check that all 3 folds were evaluated)")

    print(f"\n{'method':22s} {'Dice (subject mean±SD)':>24s} {'IoU':>16s} {'HD95':>14s}   | Dice (fold mean±SD)")
    for name in by_method:
        line = f"{name:22s}"
        for k in METRICS:
            m, s = mean_sd(v[k] for v in subj_level[name].values())
            line += f"  {fmt(m, s, k):>16s}"
        m, s = mean_sd(slice_level[name]["dice"])
        line += f"   | {fmt(m, s, 'dice')}"
        print(line)

    if baseline not in subj_level:
        print(f"\n(no results for baseline {baseline}; tests skipped)")
        return
    base = subj_level[baseline]
    print(f"\nSubject-level paired Wilcoxon signed-rank test against {baseline} (n=9)")
    for name, sl in subj_level.items():
        if name == baseline:
            continue
        shared = sorted(set(base) & set(sl))
        out = [name.ljust(22)]
        for k in METRICS:
            p, n = paired_wilcoxon([base[s][k] for s in shared], [sl[s][k] for s in shared])
            ptxt = "P=  nan " if not np.isfinite(p) else f"P={p:.4f}{'*' if p < 0.05 else ' '}"
            out.append(f"{k}: {ptxt}(n={n})")
        print("  " + "  ".join(out))
    print("  * P<0.05.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--gt")
    ap.add_argument("--name", default="method")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=1.0,
                    help="pixel spacing in mm; default 1.0 reports HD95 in pixels")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="*", help="result json files (glob patterns) to aggregate and test")
    ap.add_argument("--baseline", default="nnU-Net_v2", help="reference method for the paired tests")
    ap.add_argument("--level", choices=["external", "internal"], default="external")
    args = ap.parse_args()

    if args.compare:
        by_method = load_results(args.compare)
        if not by_method:
            ap.error("no result json matched")
        if args.level == "external":
            compare_external(by_method, args.baseline)
        else:
            compare_internal(by_method, args.baseline)
        return

    if not (args.pred and args.gt):
        ap.error("give --pred/--gt or --compare")

    rows = evaluate_dir(Path(args.pred), Path(args.gt), args.spacing)
    print(f"{args.name} fold{args.fold}: {len(rows)} slices")
    for k in METRICS:
        m, s = mean_sd(r[k] for r in rows)
        print(f"  {k:5s} = {fmt(m, s, k)}")
    subjects = sorted({r["subject"] for r in rows})
    print("  Dice per subject:", ", ".join(
        f"{s}={np.mean([r['dice'] for r in rows if r['subject'] == s]):.4f}" for s in subjects))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(name=args.name, fold=args.fold, spacing=args.spacing,
                                       cases=rows), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
