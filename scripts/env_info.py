#!/usr/bin/env python3
"""Print the software / hardware environment and the effective training configuration."""
import json
import os
import platform
from datetime import datetime

print("date:", datetime.now().isoformat(timespec="seconds"))
try:
    import ghostcil_nnunet
    print("ghostcil_nnunet:", ghostcil_nnunet.__version__)
except Exception as e:                                             # noqa: BLE001
    print("ghostcil_nnunet: not importable:", e)
print("os:", platform.platform())
print("python:", platform.python_version())

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram_GB:", round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1))
except Exception as e:                                             # noqa: BLE001
    print("torch: not available:", e)

try:
    import importlib.metadata as _im
    print("nnunetv2:", _im.version("nnunetv2"))
except Exception as e:                                             # noqa: BLE001
    print("nnunetv2: not available:", e)

print("GCIL_EPOCHS:", os.environ.get("GCIL_EPOCHS", "150 (default)"))
print("nnUNet_compile:", os.environ.get("nnUNet_compile", "(unset)"))
print("nnUNet_n_proc_DA:", os.environ.get("nnUNet_n_proc_DA", "(unset)"))

plans = os.path.join(os.environ.get("nnUNet_preprocessed", ""),
                     "Dataset501_CTSCOPE", "nnUNetPlans.json")
try:
    cfg = json.load(open(plans))["configurations"]["2d"]
    print("patch_size:", cfg["patch_size"])
    print("batch_size:", cfg["batch_size"])
    print("normalization:", cfg.get("normalization_schemes"))
except Exception as e:                                             # noqa: BLE001
    print("plans not readable:", e)
