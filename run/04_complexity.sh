#!/usr/bin/env bash
# Parameters / FLOPs / inference latency (paper Table 2).
# Needs only the preprocessed dataset (plans), not trained weights.
# Measure the latency on an idle GPU: a concurrent training run inflates the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${nnUNet_preprocessed:?nnUNet_preprocessed is not set}"
DEVICE="${GCIL_DEVICE:-cuda}"

mkdir -p results
python scripts/complexity.py --dataset-id 501 --dataset-name CTSCOPE \
  --configuration 2d --device "$DEVICE" --out results/complexity.json 2>&1 | tee results/complexity.txt

echo
echo "Next: bash run/05_collect.sh (optional: archive results, logs and sample predictions)"
