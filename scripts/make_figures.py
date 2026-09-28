#!/usr/bin/env python3
"""Re-create the result figures of the paper from the files in results/.

Figure 4 (accuracy vs. complexity) needs only results/ and is produced by default:
    python scripts/make_figures.py --results results --out figures

Figure 3 (qualitative external-test examples) needs the CT-SCOPE images, the reference
masks and prediction folders produced by run/03_predict_eval.sh; it is therefore only
drawn when --preds is given (the dataset itself is not redistributed in this repository):
    python scripts/make_figures.py --preds preds --raw $nnUNet_raw/Dataset501_CTSCOPE
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# result label -> (display name, colour); order = drawing order
METHODS = [
    ("U-Net", "U-Net", "#9AA5B1"),
    ("Attention_U-Net", "Attention U-Net", "#B8C4CE"),
    ("nnU-Net_v2", "nnU-Net v2", "#5B9BD5"),
    ("nnU-Net_v2_half", "nnU-Net v2 (half width)", "#8FBFE8"),
    ("Ghost_CIL-nnUNet", "Ghost_CIL-nnUNet (proposed)", "#ED9B54"),
    ("Ghost_CIL+MDA", "+ multi-domain attention", "#F0C674"),
    ("GMA-nnUNet", "+ attention + SAHL", "#D26D87"),
]
# result label -> label used in complexity.json
COMPLEXITY_LABEL = {"U-Net": "U-Net", "Attention_U-Net": "Attention U-Net", "nnU-Net_v2": "nnU-Net v2",
                    "nnU-Net_v2_half": "nnU-Net v2 half", "Ghost_CIL-nnUNet": "Ghost_CIL-nnUNet",
                    "Ghost_CIL+MDA": "Ghost_CIL+MDA", "GMA-nnUNet": "GMA-nnUNet"}


def external_stats(results: Path, name: str):
    vals = []
    for f in range(3):
        d = json.loads((results / f"ext_{name}__fold{f}.json").read_text(encoding="utf-8"))["cases"]
        vals.append((np.mean([x["dice"] for x in d]), np.nanmean([x["hd95"] for x in d])))
    dice = [v[0] for v in vals]
    hd = [v[1] for v in vals]
    return np.mean(dice), np.std(dice, ddof=1), np.mean(hd), np.std(hd, ddof=1)


def figure4(results: Path, out: Path):
    cx = json.loads((results / "complexity.json").read_text(encoding="utf-8"))
    gpu = cx.get("meta", {}).get("gpu", "GPU")
    rows = {}
    for r in cx["rows"]:
        rows[r["method"].split("(")[0].strip()] = r     # tolerate labels such as "GMA-nnUNet(...)"
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    # label offsets in points and horizontal alignment, per panel
    offs_a = {"Ghost_CIL-nnUNet": (7, 8), "Ghost_CIL+MDA": (7, -4), "GMA-nnUNet": (7, -16),
              "nnU-Net_v2_half": (0, 9), "nnU-Net_v2": (10, 0)}
    ha_a = {"nnU-Net_v2_half": "center"}
    offs_b = {"nnU-Net_v2": (8, 4), "nnU-Net_v2_half": (8, -4), "Ghost_CIL-nnUNet": (8, -12),
              "Ghost_CIL+MDA": (8, 4), "GMA-nnUNet": (8, -10), "U-Net": (-8, -4), "Attention_U-Net": (8, 4)}
    ha_b = {"U-Net": "right"}
    for key, lab, col in METHODS:
        r = rows[COMPLEXITY_LABEL[key]]
        d, dsd, h, hsd = external_stats(results, key)
        a.scatter(r["params_M"], d, s=r["flops_G"] * 3.2, color=col, alpha=0.75, edgecolor="#333", lw=0.8, zorder=3)
        a.errorbar(r["params_M"], d, yerr=dsd, fmt="none", ecolor="#555", elinewidth=0.8, capsize=2, zorder=2)
        a.annotate(lab, (r["params_M"], d), xytext=offs_a.get(key, (7, 7)), textcoords="offset points",
                   fontsize=8.5, ha=ha_a.get(key, "left"))
        b.scatter(r["latency_ms"], h, s=70, color=col, edgecolor="#333", lw=0.8, zorder=3)
        b.errorbar(r["latency_ms"], h, yerr=hsd, fmt="none", ecolor="#555", elinewidth=0.8, capsize=2, zorder=2)
        b.annotate(lab, (r["latency_ms"], h), xytext=offs_b.get(key, (7, 6)), textcoords="offset points",
                   fontsize=8.5, ha=ha_b.get(key, "left"))
    a.set_xlabel("Parameters / 10⁶")
    a.set_ylabel("External-test Dice (mean over 3 fold models, bars = SD)")
    a.set_xlim(0, 44); a.set_ylim(0.80, 0.955)
    a.set_title("a  Dice vs. parameters (bubble area ∝ FLOPs)", loc="left", fontweight="bold")
    b.set_xlabel(f"Inference time per slice / ms ({gpu}, batch 1, FP32)")
    b.set_ylabel("External-test HD95 / pixels (mean over 3 fold models)")
    b.set_xlim(4, 24); b.set_ylim(0, 32)
    b.set_title("b  HD95 vs. inference time", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=3)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig4_accuracy_complexity.png", dpi=300, facecolor="white")
    fig.savefig(out / "fig4_accuracy_complexity.pdf", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / 'fig4_accuracy_complexity.png'}")


def figure3(preds: Path, raw: Path, out: Path, fold: int, slices):
    from PIL import Image
    from scipy.ndimage import binary_dilation, uniform_filter

    def rd(p):
        return np.asarray(Image.open(p))

    def dice(a, b):
        s = a.sum() + b.sum()
        return 2 * (a & b).sum() / s if s else 1.0

    methods = [("nnU-Net_v2", "nnU-Net v2"), ("nnU-Net_v2_half", "nnU-Net v2 (half width)"),
               ("Ghost_CIL-nnUNet", "Ghost_CIL-nnUNet (proposed)"), ("GMA-nnUNet", "+ attention + SAHL")]
    gt_dir, img_dir = raw / "labelsTs", raw / "imagesTs"
    if not slices:
        # pick the slice where the proposed method and nnU-Net v2 differ most, and the one where they differ least
        score = []
        for g in sorted(gt_dir.glob("*.png")):
            s = g.stem
            gt = rd(g) > 0
            a = rd(preds / f"{methods[0][0]}_ext_f{fold}" / f"{s}.png") > 0
            b = rd(preds / f"{methods[2][0]}_ext_f{fold}" / f"{s}.png") > 0
            score.append((abs(dice(a, gt) - dice(b, gt)), s))
        score.sort(reverse=True)
        slices = [score[0][1], score[-1][1]]
    print("figure 3 slices:", slices)
    fig, axes = plt.subplots(2 * len(slices), 6, figsize=(13, 4.6 * len(slices)))
    for r2, s in enumerate(slices):
        img = rd(img_dir / f"{s}_0000.png")
        gt = rd(gt_dir / f"{s}.png") > 0
        preds_m = [rd(preds / f"{m}_ext_f{fold}" / f"{s}.png") > 0 for m, _ in methods]
        H, W = gt.shape
        z = min(96, H, W)
        dis = np.zeros(gt.shape, float)
        for p in preds_m:
            dis += (p ^ gt)
        dens = uniform_filter(dis, z)
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        y0, x0 = int(np.clip(cy - z // 2, 0, H - z)), int(np.clip(cx - z // 2, 0, W - z))
        panels = [("CT", None), ("Reference", gt)] + [(n, p) for (_, n), p in zip(methods, preds_m)]
        for c, (title, mask) in enumerate(panels):
            for row, zoom in ((2 * r2, False), (2 * r2 + 1, True)):
                ax = axes[row, c]
                im = img[y0:y0 + z, x0:x0 + z] if zoom else img
                ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                if mask is not None:
                    m = mask[y0:y0 + z, x0:x0 + z] if zoom else mask
                    edge = m ^ binary_dilation(m)
                    overlay = np.zeros((*m.shape, 4))
                    overlay[m] = (1.0, 0.85, 0.1, 0.45)
                    overlay[edge] = (1.0, 0.2, 0.1, 0.95)
                    ax.imshow(overlay, interpolation="nearest")
                if not zoom:
                    ax.add_patch(Rectangle((x0, y0), z, z, fill=False, ec="#00e5ff", lw=1.3))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if row % 2 == 0:
                    ax.set_title(title, fontsize=9.5)
                if c == 0:
                    ax.set_ylabel(f"{s}\n{'zoom' if zoom else 'full slice'}", fontsize=9)
                if mask is not None and c >= 2 and not zoom:
                    ax.text(6, 24, f"Dice {dice(mask, gt):.3f}", color="w", fontsize=8,
                            bbox=dict(fc="k", alpha=0.5, lw=0))
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig3_qualitative.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out / 'fig3_qualitative.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--preds", help="folder with <method>_ext_f<fold>/ prediction folders (enables figure 3)")
    ap.add_argument("--raw", help="$nnUNet_raw/Dataset501_CTSCOPE (needed for figure 3)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--slices", nargs="*", help="slice ids for figure 3, e.g. P13_083 P13_095 (default: automatic)")
    args = ap.parse_args()
    figure4(Path(args.results), Path(args.out))
    if args.preds:
        if not args.raw:
            ap.error("--raw is required together with --preds")
        figure3(Path(args.preds), Path(args.raw), Path(args.out), args.fold, args.slices)


if __name__ == "__main__":
    main()
