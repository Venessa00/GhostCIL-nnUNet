#!/usr/bin/env bash
# Prediction + evaluation + statistics.
#   External test set (P13/P16/P30/P38, 120 in-vivo slices from other centres): each fold model
#   predicts separately (no ensembling, no TTA).
#   Internal cross-validation: each fold model re-predicts its validation slices with TTA off,
#   so that both evaluations follow the same protocol (nnU-Net's own fold_k/validation/ uses mirroring TTA).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${nnUNet_raw:?nnUNet_raw is not set}"
: "${nnUNet_results:?nnUNet_results is not set}"

DATASET_ID=501
CONFIG=2d
DS="Dataset$(printf %03d $DATASET_ID)_CTSCOPE"
IMAGES_TS="$nnUNet_raw/$DS/imagesTs"
LABELS_TS="$nnUNet_raw/$DS/labelsTs"
LABELS_TR="$nnUNet_raw/$DS/labelsTr"
SPACING="${GCIL_SPACING:-1.0}"     # pixel spacing in mm if known (export GCIL_SPACING=0.xx); default reports HD95 in pixels
DEVICE="${GCIL_DEVICE:-cuda}"
export nnUNet_compile="${nnUNet_compile:-false}"

mkdir -p preds results
echo "== validation slice folders per fold (valsets/fold_k) =="
python scripts/make_fold_val_dirs.py --out valsets

# trainer:plans:label (labels are used as file names; no spaces)
SPECS=(
  "nnUNetTrainerBaseline150:nnUNetPlans:nnU-Net_v2"
  "nnUNetTrainerBaseline150:nnUNetPlans_half:nnU-Net_v2_half"
  "nnUNetTrainerGhostCIL:nnUNetPlans:Ghost_CIL-nnUNet"
  "nnUNetTrainerGhostCIL_MDA:nnUNetPlans:Ghost_CIL+MDA"
  "nnUNetTrainerGMA:nnUNetPlans:GMA-nnUNet"
  "nnUNetTrainerGMA_NoFreq:nnUNetPlans:GMA_NoFreq"
  "nnUNetTrainerGMA_DiceOnly:nnUNetPlans:GMA_DiceOnly"
  "nnUNetTrainerGMA_DiceFocal:nnUNetPlans:GMA_DiceFocal"
  "nnUNetTrainerGMA_DiceBoundary:nnUNetPlans:GMA_DiceBoundary"
  "nnUNetTrainerUNet:nnUNetPlans:U-Net"
  "nnUNetTrainerAttUNet:nnUNetPlans:Attention_U-Net"
  "nnUNetTrainerExtModel:nnUNetPlans:${GCIL_EXT_NAME:-ExtModel}"
)

for spec in "${SPECS[@]}"; do
  IFS=: read -r tr plans name <<<"$spec"
  model_dir="$nnUNet_results/$DS/${tr}__${plans}__${CONFIG}"
  [ -d "$model_dir" ] || { echo "skip $name (not trained)"; continue; }

  for f in 0 1 2; do
    ckpt="$model_dir/fold_${f}/checkpoint_final.pth"
    [ -f "$ckpt" ] || continue

    echo "== external test set: $name fold $f (single fold model, TTA off) =="
    out="preds/${name}_ext_f${f}"
    mkdir -p "$out"
    nnUNetv2_predict -i "$IMAGES_TS" -o "$out" -d $DATASET_ID -c $CONFIG \
        -tr "$tr" -p "$plans" -f "$f" -device "$DEVICE" --disable_tta
    python scripts/evaluate.py --pred "$out" --gt "$LABELS_TS" --spacing "$SPACING" \
        --name "$name" --fold "$f" --out "results/ext_${name}__fold${f}.json"

    echo "== internal validation set: $name fold $f (validation subjects of this fold, TTA off) =="
    out_int="preds/${name}_int_f${f}"
    mkdir -p "$out_int"
    nnUNetv2_predict -i "valsets/fold_${f}" -o "$out_int" -d $DATASET_ID -c $CONFIG \
        -tr "$tr" -p "$plans" -f "$f" -device "$DEVICE" --disable_tta
    python scripts/evaluate.py --pred "$out_int" --gt "$LABELS_TR" --spacing "$SPACING" \
        --name "$name" --fold "$f" --out "results/int_${name}__fold${f}.json"
  done
done

echo
echo "== external test set: aggregation + tests against nnU-Net v2 =="
python scripts/evaluate.py --compare "results/ext_*.json" --baseline "nnU-Net_v2" --level external \
  | tee results/external_summary.txt

echo
echo "== internal cross-validation: aggregation + subject-level tests against nnU-Net v2 =="
python scripts/evaluate.py --compare "results/int_*.json" --baseline "nnU-Net_v2" --level internal \
  | tee results/internal_summary.txt

echo
echo "Next: bash run/04_complexity.sh"
