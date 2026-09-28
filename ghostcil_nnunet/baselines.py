"""Baseline networks trained inside the nnU-Net pipeline.

All baselines are trained with exactly the same data pipeline, augmentation, optimiser
and schedule as nnU-Net; only the network returned by ``build_network_architecture`` is
exchanged. This is what allows the comparison to attribute differences to the network
alone.

Contents
  PlainUNet       original U-Net (Ronneberger et al., 2015: base width 64, 5 levels,
                  double convolutions, transposed-convolution up-sampling)
  AttentionUNet   Attention U-Net (Oktay et al., 2018: attention gates on the skips)
  load_external   loads a third-party network (e.g. UNeXt, CMUNeXt) from environment
                  variables. Requirements: the constructor accepts the number of input
                  channels and classes, and ``forward`` returns (B, num_classes, H, W)
                  logits (no sigmoid / softmax inside).
                    export GCIL_EXT_PATH=/path/to/UNeXt-pytorch
                    export GCIL_EXT_MODEL="archs:UNext"
                    export GCIL_EXT_KWARGS='{"num_classes": 2, "input_channels": 1, "img_size": 512}'
                  The two keyword names can be changed with GCIL_EXT_IN_KEY /
                  GCIL_EXT_OUT_KEY (defaults: input_channels / num_classes).

These networks have a single output, so the corresponding trainers disable nnU-Net's
deep supervision (see trainers.py).
"""
import importlib
import json
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PlainUNet", "AttentionUNet", "load_external", "SingleOutputWrapper",
           "infer_io_channels"]


class _DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class _DummyDecoder:
    """nnU-Net accesses ``network.decoder.deep_supervision``; provide a stand-in."""
    deep_supervision = False


class SingleOutputWrapper(nn.Module):
    """Wraps any single-output network so that nnU-Net finds ``.decoder.deep_supervision``."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.decoder = _DummyDecoder()

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)):        # some networks return several heads in training mode: use the last (full resolution)
            out = out[-1]
        return out


class PlainUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 64, depth: int = 5):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        c = in_channels
        for ch in chs:
            self.downs.append(_DoubleConv(c, ch))
            c = ch
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.up_convs.append(_DoubleConv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self.decoder = _DummyDecoder()

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, conv in zip(self.ups, self.up_convs):
            x = up(x)
            s = skips.pop()
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([s, x], dim=1))
        return self.head(x)


class _AttentionGate(nn.Module):
    """Additive attention gate of Oktay et al.: the decoder signal g re-weights the skip feature x."""

    def __init__(self, ch_x, ch_g, ch_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(ch_x, ch_int, 1, bias=False), nn.BatchNorm2d(ch_int))
        self.wg = nn.Sequential(nn.Conv2d(ch_g, ch_int, 1, bias=False), nn.BatchNorm2d(ch_int))
        self.psi = nn.Sequential(nn.Conv2d(ch_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(F.relu(self.wx(x) + self.wg(g), inplace=True))
        return x * a


class AttentionUNet(PlainUNet):
    def __init__(self, in_channels: int, num_classes: int, base: int = 64, depth: int = 5):
        super().__init__(in_channels, num_classes, base, depth)
        chs = [base * (2 ** i) for i in range(depth)]
        self.gates = nn.ModuleList(
            [_AttentionGate(chs[i - 1], chs[i - 1], max(chs[i - 1] // 2, 8)) for i in range(depth - 1, 0, -1)])

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, conv, gate in zip(self.ups, self.up_convs, self.gates):
            x = up(x)
            s = skips.pop()
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            s = gate(s, x)
            x = conv(torch.cat([s, x], dim=1))
        return self.head(x)


def load_external(in_channels: int, num_classes: int) -> nn.Module:
    spec = os.environ.get("GCIL_EXT_MODEL", "")
    if ":" not in spec:
        raise RuntimeError("set GCIL_EXT_MODEL='module:ClassName', e.g. archs:UNext")
    path = os.environ.get("GCIL_EXT_PATH", "")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    mod_name, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    kwargs = json.loads(os.environ.get("GCIL_EXT_KWARGS", "{}"))
    kwargs.setdefault(os.environ.get("GCIL_EXT_IN_KEY", "input_channels"), in_channels)
    kwargs.setdefault(os.environ.get("GCIL_EXT_OUT_KEY", "num_classes"), num_classes)
    net = cls(**kwargs)
    print(f"[GhostCIL] loaded external model {spec} kwargs={kwargs}")
    return SingleOutputWrapper(net)


def infer_io_channels(nnunet_net: nn.Module):
    """Read the number of input channels and classes from a network built by nnU-Net."""
    in_ch = out_ch = None
    for m in nnunet_net.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    seg = getattr(getattr(nnunet_net, "decoder", None), "seg_layers", None)
    if seg is not None and len(seg) > 0:
        out_ch = seg[-1].out_channels
    else:
        for m in nnunet_net.modules():
            if isinstance(m, nn.Conv2d):
                out_ch = m.out_channels
    if in_ch is None or out_ch is None:
        raise RuntimeError("could not infer input/output channels from the nnU-Net network; "
                           "print(net) and adapt infer_io_channels")
    return in_ch, out_ch
