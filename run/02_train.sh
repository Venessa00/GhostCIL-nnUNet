#!/usr/bin/env bash
# Train all configurations. Re-running the same command skips finished configurations and
# continues interrupted ones from nnU-Net's latest checkpoint (--c).
#
# Usage:
#   bash run/02_train.sh              # main experiment: 5 configurations x 3 folds (Tables 1-3)
#   bash run/02_train.sh loss         # loss ablation: 3 configurations x fold 0 (Table 4)
#   bash run/02_train.sh attn         # attention without frequency branch x fold 0
#   bash run/02_train.sh base         # U-Net / Attention U-Net (/ external network) x 3 folds
#   bash run/02_train.sh all          # everything (25 trainings)
#
#   Smoke test before the real run (about 1 min per configuration):
#       GCIL_EPOCHS=1 GCIL_ITERS=5 GCIL_VAL_ITERS=2 GCIL_FOLDS=0 bash run/02_train.sh all
#   then delete $nnUNet_results/Dataset501_CTSCOPE (otherwise the smoke-test runs count as finished).
#
# Environment variables:
#   GCIL_EPOCHS=150      epochs (paper: 150)
#   GCIL_ITERS / GCIL_VAL_ITERS   iterations per epoch, smoke tests only
#   GCIL_FOLDS="0 1 2"   folds to train
#   GCIL_DEVICE=cuda     training device
#   nnUNet_compile       off by default (torch.compile is untested with the FFT-based attention module)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${nnUNet_results:?nnUNet_results is not set}"
: "${nnUNet_preprocessed:?nnUNet_preprocessed is not set}"
export GCIL_EPOCHS="${GCIL_EPOCHS:-150}"
export nnUNet_compile="${nnUNet_compile:-false}"
# reduces memory fragmentation on GPUs with 12 GB or less (torch >= 2.1)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FAILED=()
DEVICE="${GCIL_DEVICE:-cuda}"
FOLDS="${GCIL_FOLDS:-0 1 2}"

DATASET_ID=501
CONFIG=2d
MODE="${1:-main}"
DS="Dataset$(printf %03d $DATASET_ID)_CTSCOPE"

# main experiment: trainer:plans
MAIN=(
  "nnUNetTrainerBaseline150:nnUNetPlans"       # nnU-Net v2 baseline
  "nnUNetTrainerBaseline150:nnUNetPlans_half"  # half-width nnU-Net control
  "nnUNetTrainerGhostCIL:nnUNetPlans"          # Ghost_CIL-nnUNet (proposed)
  "nnUNetTrainerGhostCIL_MDA:nnUNetPlans"      # ablation: + multi-domain attention
  "nnUNetTrainerGMA:nnUNetPlans"               # ablation: + attention + SAHL
)
LOSS=(
  "nnUNetTrainerGMA_DiceOnly:nnUNetPlans"
  "nnUNetTrainerGMA_DiceFocal:nnUNetPlans"
  "nnUNetTrainerGMA_DiceBoundary:nnUNetPlans"
)
ATTN=(
  "nnUNetTrainerGMA_NoFreq:nnUNetPlans"        # attention without the frequency branch
)
BASE=(
  "nnUNetTrainerUNet:nnUNetPlans"              # U-Net
  "nnUNetTrainerAttUNet:nnUNetPlans"           # Attention U-Net
)
# third-party network (UNeXt etc.): added automatically when GCIL_EXT_* is set, see ghostcil_nnunet/baselines.py
[ -n "${GCIL_EXT_MODEL:-}" ] && BASE+=("nnUNetTrainerExtModel:nnUNetPlans")

ensure_half_plans () {
  if [ ! -f "$nnUNet_preprocessed/$DS/nnUNetPlans_half.json" ]; then
    echo "== creating the half-width plans =="
    python scripts/make_half_width_plans.py
  fi
}

train () {
  local tr="${1%%:*}" plans="${1##*:}" fold="$2"
  [ "$plans" = "nnUNetPlans_half" ] && ensure_half_plans
  local done_flag="$nnUNet_results/$DS/${tr}__${plans}__${CONFIG}/fold_${fold}/checkpoint_final.pth"
  if [ -f "$done_flag" ]; then
    echo "  skip (finished): $tr [$plans] fold $fold"
    return
  fi
  echo "  >>> training $tr [$plans] fold $fold (GCIL_EPOCHS=$GCIL_EPOCHS)"
  # --c: continue from checkpoint_latest if present, otherwise start from scratch
  # a failure (typically out of memory) does not stop the batch; failures are listed at the end
  if ! nnUNetv2_train $DATASET_ID $CONFIG "$fold" -tr "$tr" -p "$plans" -device "$DEVICE" --c; then
    echo "  FAILED: $tr [$plans] fold $fold (out of memory? reduce the batch size with scripts/set_batch_size.py)"
    FAILED+=("$tr [$plans] fold $fold")
  fi
}

if [ "$MODE" = "main" ] || [ "$MODE" = "all" ]; then
  echo "== main experiment: 5 configurations x 3 folds =="
  for spec in "${MAIN[@]}"; do
    for fold in $FOLDS; do train "$spec" "$fold"; done
  done
fi

if [ "$MODE" = "loss" ] || [ "$MODE" = "all" ]; then
  echo "== loss ablation: 3 configurations x fold 0 =="
  for spec in "${LOSS[@]}"; do train "$spec" 0; done
fi

if [ "$MODE" = "attn" ] || [ "$MODE" = "all" ]; then
  echo "== attention ablation: no frequency branch x fold 0 =="
  for spec in "${ATTN[@]}"; do train "$spec" 0; done
fi

if [ "$MODE" = "base" ] || [ "$MODE" = "all" ]; then
  echo "== baselines: ${#BASE[@]} configurations x 3 folds =="
  for spec in "${BASE[@]}"; do
    for fold in $FOLDS; do train "$spec" "$fold"; done
  done
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "${#FAILED[@]} configuration(s) failed:"
  printf '   %s\n' "${FAILED[@]}"
  echo "   fix the cause and re-run the same command; finished configurations are skipped."
  exit 1
fi
echo "Training finished. Next: bash run/03_predict_eval.sh"
