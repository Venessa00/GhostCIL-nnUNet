#!/usr/bin/env python3
"""Parameters / FLOPs / single-slice inference latency of every network (paper Table 2).

FLOPs are counted with forward hooks on Conv2d / ConvTranspose2d / Linear as multiply-
accumulate operations (1 MAC = 1 multiplication + 1 addition). The FFT inside the
frequency branch of the attention module is not a MAC-type operator and is therefore
not included in the FLOPs; it is included in the measured latency.

Protocol: batch 1, one slice of the patch size chosen by nnU-Net, float32, deep
supervision off, no test-time augmentation, no torch.compile, 20 warm-up runs and
100 timed runs.

No trainer is instantiated (that would create folders and logs in nnUNet_results);
the static ``build_network_architecture`` of each trainer class is called directly with
arguments adapted to the installed nnU-Net version (2.0/2.1, 2.2-2.7, 2.8+). A failure
for one model does not stop the others; errors are printed and stored in the JSON.

Usage (after preprocessing):
    python scripts/complexity.py --dataset-id 501 --configuration 2d [--device cuda|cpu]
"""
import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# (label, trainer, plans). The half-width nnU-Net uses the same trainer with different plans.
TRAINERS = [
    ("nnU-Net v2",        "nnUNetTrainerBaseline150", "nnUNetPlans"),
    ("nnU-Net v2 half",   "nnUNetTrainerBaseline150", "nnUNetPlans_half"),
    ("Ghost_CIL-nnUNet",  "nnUNetTrainerGhostCIL",    "nnUNetPlans"),
    ("Ghost_CIL+MDA",     "nnUNetTrainerGhostCIL_MDA","nnUNetPlans"),
    ("GMA-nnUNet",        "nnUNetTrainerGMA",         "nnUNetPlans"),
    ("GMA-nnUNet no-freq","nnUNetTrainerGMA_NoFreq",  "nnUNetPlans"),
    ("U-Net",             "nnUNetTrainerUNet",        "nnUNetPlans"),
    ("Attention U-Net",   "nnUNetTrainerAttUNet",     "nnUNetPlans"),
]
if os.environ.get("GCIL_EXT_MODEL"):
    TRAINERS.append((os.environ.get("GCIL_EXT_NAME", "ExtModel"), "nnUNetTrainerExtModel", "nnUNetPlans"))


def count_flops(model: nn.Module, input_shape, device) -> int:
    """Multiply-accumulate operations of one forward pass (1 MAC = 1 multiplication + 1 addition)."""
    total = [0]
    handles = []

    def conv_hook(m, inp, out):
        out_elems = out.numel() // out.shape[0]          # without batch
        kernel_ops = int(np.prod(m.kernel_size)) * (m.in_channels // m.groups)
        total[0] += out_elems * kernel_ops

    def deconv_hook(m, inp, out):
        in_elems = inp[0].numel() // inp[0].shape[0]
        kernel_ops = int(np.prod(m.kernel_size)) * (m.out_channels // m.groups)
        total[0] += in_elems * kernel_ops

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.ConvTranspose2d):
            handles.append(m.register_forward_hook(deconv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *input_shape, device=device))
    for h in handles:
        h.remove()
    return total[0]


def measure_latency(model, input_shape, device, warmup=20, repeat=100) -> float:
    model.eval()
    x = torch.randn(1, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / repeat * 1000.0      # ms


def find_trainer_class(name):
    try:                                                   # nnU-Net >= 2.5
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        return recursive_find_trainer_class_by_name(name)
    except ImportError:                                    # older versions
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2.training.nnUNetTrainer as tr_pkg
        return recursive_find_python_class(str(Path(tr_pkg.__file__).parent), name,
                                           "nnunetv2.training.nnUNetTrainer")


def build_network(cls, plans: dict, dataset_json: dict, configuration: str):
    """Call build_network_architecture with the arguments of the installed nnU-Net version."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    pm = PlansManager(plans)
    cm = pm.get_configuration(configuration)
    n_in = determine_num_input_channels(pm, cm, dataset_json)
    n_out = pm.get_label_manager(dataset_json).num_segmentation_heads
    params = inspect.signature(cls.build_network_architecture).parameters
    if "plans_manager" in params and "dataset_json" in params:          # nnU-Net 2.0 / 2.1
        net = cls.build_network_architecture(pm, dataset_json, cm, n_in, False)
    elif "plans_manager" in params:                                       # nnU-Net >= 2.8
        net = cls.build_network_architecture(pm, cm, n_in, n_out, False)
    else:                                                                 # nnU-Net 2.2 - 2.7
        net = cls.build_network_architecture(
            cm.network_arch_class_name, cm.network_arch_init_kwargs,
            cm.network_arch_init_kwargs_req_import, n_in, n_out, False)
    # deep supervision off: only the full-resolution output is counted, as for the other networks
    if hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision"):
        net.decoder.deep_supervision = False
    return net, (n_in, *tuple(cm.patch_size))


def gpu_busy_warning():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        procs = [l for l in out.stdout.splitlines() if l.strip()]
        if len(procs) > 1:
            print(f"WARNING: {len(procs) - 1} other process(es) are using the GPU; "
                  f"latencies will be inflated. Re-run this step on an idle GPU.")
    except Exception:               # noqa: BLE001  nvidia-smi not available
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--dataset-name", default="CTSCOPE")
    ap.add_argument("--configuration", default="2d")
    ap.add_argument("--plans-name", default="nnUNetPlans")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/complexity.json")
    args = ap.parse_args()

    pre = Path(os.environ.get("nnUNet_preprocessed", ""))
    ds = pre / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    plans = json.loads((ds / f"{args.plans_name}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((ds / "dataset.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda":
        gpu_busy_warning()

    meta = dict(torch=torch.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                dtype="float32", tta="off", batch=1, compile="off",
                flops_note="MAC count over Conv2d/ConvTranspose2d/Linear only; FFT not included")
    try:
        import importlib.metadata as im
        meta["nnunetv2"] = im.version("nnunetv2")
    except Exception:                # noqa: BLE001
        pass

    rows, errors = [], []
    plans_cache = {args.plans_name: plans}
    for label, trainer_name, plans_name in TRAINERS:
        if plans_name not in plans_cache:
            pf = ds / f"{plans_name}.json"
            if not pf.exists():
                print(f"{label:22s} skipped: {pf} not found (run scripts/make_half_width_plans.py first)")
                continue
            plans_cache[plans_name] = json.loads(pf.read_text(encoding="utf-8"))
        try:
            cls = find_trainer_class(trainer_name)
            if cls is None:
                raise RuntimeError(f"trainer {trainer_name} not found; run scripts/install_trainers.py")
            net, in_shape = build_network(cls, plans_cache[plans_name], dataset_json, args.configuration)
            net = net.to(device)
            params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            flops = count_flops(net, in_shape, device)
            lat = measure_latency(net, in_shape, device)
        except Exception as e:       # noqa: BLE001  one failing model must not stop the others
            print(f"{label:22s} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append(dict(method=label, trainer=trainer_name, plans=plans_name,
                               error=f"{type(e).__name__}: {e}"))
            continue
        rows.append(dict(method=label, trainer=trainer_name, plans=plans_name,
                         params_M=params / 1e6, flops_G=flops / 1e9, latency_ms=lat,
                         input_shape=list(in_shape)))
        print(f"{label:22s} params {params/1e6:8.3f}M   FLOPs {flops/1e9:9.3f}G   "
              f"latency {lat:7.2f}ms")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = next((r for r in rows if r["trainer"] == "nnUNetTrainerBaseline150"
                 and r["plans"] == "nnUNetPlans"), None)
    if base is None:
        print("\n(nnU-Net v2 reference row missing; relative reductions not computed)")
    else:
        print("\nReduction relative to nnU-Net v2:")
        for r in rows:
            if r is base:
                continue
            dp = (1 - r["params_M"] / base["params_M"]) * 100
            df = (1 - r["flops_G"] / base["flops_G"]) * 100
            dl = (1 - r["latency_ms"] / base["latency_ms"]) * 100
            r.update(d_params_pct=dp, d_flops_pct=df, d_latency_pct=dl)
            print(f"  {r['method']:22s} params {dp:+6.1f}%   FLOPs {df:+6.1f}%   latency {dl:+6.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, rows=rows, errors=errors),
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    if errors:
        print(f"WARNING: {len(errors)} model(s) failed, see the tracebacks above and the 'errors' field")
        sys.exit(1)


if __name__ == "__main__":
    main()
