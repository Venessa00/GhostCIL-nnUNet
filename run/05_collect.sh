#!/usr/bin/env bash
# Archive the results of steps 03/04 together with the training logs, a few sample predictions
# and the environment description (no weights).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M)
OUT="ghostcil_results_${STAMP}.tar.gz"

mkdir -p collect/logs collect/samples

# 1) metrics and complexity
cp -r results collect/ 2>/dev/null || true

# 2) training logs (text and progress plots only)
if [ -n "${nnUNet_results:-}" ]; then
  find "$nnUNet_results" -name "training_log_*.txt" -exec bash -c \
    'cp "$1" "collect/logs/$(echo "$1" | sed "s#.*/Dataset#Dataset#;s#/#__#g")"' _ {} \;
  find "$nnUNet_results" -name "progress.png" -exec bash -c \
    'cp "$1" "collect/logs/$(echo "$1" | sed "s#.*/Dataset#Dataset#;s#/#__#g")"' _ {} \;
fi

# 3) the same 6 sample predictions of every method (for the qualitative figure)
for d in preds/*_ext_f*/; do
  d="${d%/}"
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "collect/samples/$name"
  ls "$d"/*.png 2>/dev/null | head -6 | xargs -r -I{} cp {} "collect/samples/$name/"
done

# 4) the matching images and reference masks
if [ -n "${nnUNet_raw:-}" ]; then
  mkdir -p collect/samples/_gt collect/samples/_image
  ls "$nnUNet_raw"/Dataset501_CTSCOPE/labelsTs/*.png | head -6 | xargs -r -I{} cp {} collect/samples/_gt/
  ls "$nnUNet_raw"/Dataset501_CTSCOPE/imagesTs/*.png | head -6 | xargs -r -I{} cp {} collect/samples/_image/
fi

# 5) environment
python scripts/env_info.py > collect/env.txt

# 6) radial gains learned by the frequency branch (ablation models); failure does not stop the archive
python scripts/dump_radial_gain.py --out collect/results/radial_gain.json || true

tar czf "$OUT" collect
rm -rf collect
echo "archived: $OUT"
