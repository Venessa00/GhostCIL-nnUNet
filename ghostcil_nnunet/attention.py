"""Multi-domain attention (frequency / spatial / channel / coordinate branches).

This module is evaluated as an *ablation* in the paper; it is not part of the proposed
Ghost_CIL-nnUNet. Each branch produces one attention map, the maps are multiplied
together (with broadcasting) and the result re-weights the input feature map. Shape and
channel count are preserved, so the module can be inserted after any encoder stage of
the network that nnU-Net builds.

Frequency branch: a learnable radial frequency response. The normalised radial
frequency r in [0, 1] is divided into K bins, each with one learnable real gain. The
spectrum of the channel-averaged feature map is multiplied by the gain obtained by
linear interpolation over r and transformed back to the spatial domain.
  * real gains -> zero-phase filtering, the output stays aligned with the input;
  * normalised frequency -> independent of the feature-map size, so one module works
    at every resolution level;
  * only K parameters (default 16);
  * initialised to all-pass (all ones) so that the branch is the identity at start.
The FFT is applied to the channel mean, which keeps the complex tensor at O(H*W)
instead of O(C*H*W).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiDomainAttention", "insert_attention_into_encoder", "RadialFrequencyFilter"]


class RadialFrequencyFilter(nn.Module):
    """Learnable radial frequency response. Input (B,1,H,W) real, output same shape."""

    def __init__(self, n_bins: int = 16):
        super().__init__()
        assert n_bins >= 2
        self.n_bins = n_bins
        self.gain = nn.Parameter(torch.ones(n_bins))     # all-pass initialisation

    def radial_gain(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Per-coefficient gain for an rfft2 spectrum of shape (H, W//2+1)."""
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype)      # [-0.5, 0.5)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype)     # [0, 0.5]
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (0.5 * math.sqrt(2.0))
        r = r.clamp(0.0, 1.0) * (self.n_bins - 1)
        lo = r.floor().long()
        hi = (lo + 1).clamp(max=self.n_bins - 1)
        frac = r - lo.to(r.dtype)
        gain = self.gain.to(dtype)
        return gain[lo] * (1.0 - frac) + gain[hi] * frac

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        b, c, h, w = g.shape
        assert c == 1, "the radial filter operates on a single-channel map"
        g32 = g.float()
        spec = torch.fft.rfft2(g32, norm="ortho")                  # (B,1,H,W//2+1) complex
        spec = spec * self.radial_gain(h, w, spec.device, torch.float32)
        out = torch.fft.irfft2(spec, s=(h, w), norm="ortho")      # (B,1,H,W) real
        return out.to(g.dtype)


class _FrequencyBranch(nn.Module):
    def __init__(self, n_bins: int = 16, hidden: int = 8):
        super().__init__()
        self.filt = RadialFrequencyFilter(n_bins)
        self.post = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=1, keepdim=True)                 # (B,1,H,W)
        filtered = self.filt(g)
        return torch.sigmoid(self.post(filtered))       # (B,1,H,W)


class _SpatialBranch(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B,1,H,W)


class _ChannelBranch(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        v = x.mean(dim=(2, 3))                                 # (B,C)
        return torch.sigmoid(self.mlp(v)).view(b, c, 1, 1)


class _CoordinateBranch(nn.Module):
    """Coordinate attention (Hou et al., CVPR 2021)."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.InstanceNorm2d(hidden, affine=True)
        self.act = nn.LeakyReLU(1e-2, inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)                 # (B,C,H,1) pooled along width
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B,C,W,1) pooled along height
        y = torch.cat([x_h, x_w], dim=2)                  # (B,C,H+W,1)
        y = self.act(self.norm(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)                     # (B,hidden,1,W)
        a_h = torch.sigmoid(self.conv_h(y_h))             # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w))             # (B,C,1,W)
        return a_h * a_w                                  # (B,C,H,W) by broadcasting


class MultiDomainAttention(nn.Module):
    """Frequency + spatial + channel + coordinate attention. Output shape equals input shape."""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, coord_reduction: int = 32,
                 freq_bins: int = 16,
                 use_freq: bool = True, use_spatial: bool = True,
                 use_channel: bool = True, use_coord: bool = True):
        super().__init__()
        self.freq = _FrequencyBranch(freq_bins) if use_freq else None
        self.spatial = _SpatialBranch(spatial_kernel) if use_spatial else None
        self.channel = _ChannelBranch(channels, reduction) if use_channel else None
        self.coord = _CoordinateBranch(channels, coord_reduction) if use_coord else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = None
        for branch in (self.freq, self.spatial, self.channel, self.coord):
            if branch is None:
                continue
            w = branch(x)
            m = w if m is None else m * w
        return x if m is None else x * m

    def learned_radial_gain(self) -> Optional[torch.Tensor]:
        """The K radial gains learned by the frequency branch (None if the branch is off)."""
        if self.freq is None:
            return None
        return self.freq.filt.gain.detach().cpu()


class _StageWithAttention(nn.Module):
    """Wraps one encoder stage together with the attention module that follows it."""

    def __init__(self, stage: nn.Module, attention: nn.Module):
        super().__init__()
        self.stage = stage
        self.attention = attention

    def forward(self, x):
        return self.attention(self.stage(x))


def insert_attention_into_encoder(net: nn.Module, min_stage: int = 0,
                                  **attn_kwargs) -> int:
    """Insert a MultiDomainAttention module after every encoder stage of ``net``.

    Only ``encoder.stages`` is modified; the decoder and the skip connections are left
    as they are, so the forward topology of the nnU-Net network is unchanged. Returns
    the number of modules inserted (0 means that ``encoder.stages`` was not found).
    """
    encoder = getattr(net, "encoder", None)
    if encoder is None:
        return 0
    stages = getattr(encoder, "stages", None)
    if stages is None:
        return 0

    out_channels = getattr(encoder, "output_channels", None)
    inserted = 0
    for i in range(len(stages)):
        if i < min_stage:
            continue
        stage = stages[i]
        if isinstance(stage, _StageWithAttention):
            continue
        if out_channels is not None:
            ch = int(out_channels[i])
        else:
            ch = _infer_out_channels(stage)
            if ch is None:
                continue
        stages[i] = _StageWithAttention(stage, MultiDomainAttention(ch, **attn_kwargs))
        inserted += 1
    return inserted


def _infer_out_channels(module: nn.Module) -> Optional[int]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m.out_channels
    return last
