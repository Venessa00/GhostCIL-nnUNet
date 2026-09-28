#!/usr/bin/env bash
# Download the data -> convert to nnU-Net format -> label audit -> preprocessing -> subject-level splits
# Usage: bash run/01_prepare.sh [path/to/Craneal_CT]
#        without a path the repository is cloned to ./Craneal_CT
set -euo pipefail
cd "$(dirname "$0")/.."

: "${nnUNet_raw:?nnUNet_raw is not set (source ~/.bashrc or run run/00_env.sh)}"
: "${nnUNet_preprocessed:?nnUNet_preprocessed is not set}"

DATASET_ID=501
DATASET_NAME=CTSCOPE
SRC="${1:-./Craneal_CT}"

if [ ! -d "$SRC" ]; then
  echo "== cloning the CT-SCOPE repository (1-2 GB) =="
  git clone --depth 1 https://github.com/BYO-UPM/Craneal_CT "$SRC"
fi

echo "== converting to the nnU-Net v2 raw format =="
python scripts/prepare_data.py --src "$SRC" --dataset-id $DATASET_ID --dataset-name $DATASET_NAME

echo "== label audit (mask encodings are mixed; must pass before training) =="
python scripts/check_labels.py

echo "== nnU-Net preprocessing and automatic configuration =="
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 2d --verify_dataset_integrity

echo "== installing the subject-level 3-fold split (replaces nnU-Net's random 5-fold split) =="
TARGET="$nnUNet_preprocessed/Dataset$(printf %03d $DATASET_ID)_${DATASET_NAME}/splits_final.json"
cp splits_final.json "$TARGET"
python scripts/check_splits.py

echo
echo "Data ready. Next: python scripts/set_batch_size.py --value 2  (as in the paper), then bash run/02_train.sh"
