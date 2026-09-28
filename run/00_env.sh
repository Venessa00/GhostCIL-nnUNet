#!/usr/bin/env bash
# Environment setup. Run once.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 PyTorch (skipped if already installed) =="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "== 2/5 nnU-Net v2 =="
pip install "nnunetv2>=2.5,<3.0"

echo "== 3/5 this package =="
pip install -e .

echo "== 4/5 nnU-Net folders (exported in ~/.bashrc) =="
WORK="${GCIL_WORK:-$HOME/ghostcil_work}"
mkdir -p "$WORK"/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}
for v in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  grep -q "export $v=" ~/.bashrc || echo "export $v=$WORK/$v" >> ~/.bashrc
done
export nnUNet_raw="$WORK/nnUNet_raw"
export nnUNet_preprocessed="$WORK/nnUNet_preprocessed"
export nnUNet_results="$WORK/nnUNet_results"
echo "   nnUNet_raw=$nnUNet_raw"

echo "== 5/5 register the custom trainers =="
python scripts/install_trainers.py

echo
echo "CPU smoke test:"
python tests/test_modules_cpu.py

echo
echo "Ready. Next: bash run/01_prepare.sh /path/to/Craneal_CT"
