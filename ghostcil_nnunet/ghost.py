"""Ghost convolution and in-place replacement of standard convolutions (Ghost_CIL unit).

The standard convolutions of the network that nnU-Net configures automatically are
replaced *in place* by Ghost convolutions. The number of stages, the channels per
stage and the number of down-sampling steps are left untouched, so every other
decision made by nnU-Net's automatic configuration still holds. Together with the
instance-normalisation layer and the LeakyReLU that follow it, a replaced
convolution forms the Ghost_CIL unit described in the paper.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn

__all__ = ["GhostConv2d", "convert_to_ghost", "count_ghost_modules"]


class GhostConv2d(nn.Module):
    """Ghost convolution (Han et al., CVPR 2020).

    A primary convolution produces ``out_channels / ratio`` intrinsic feature maps and
    a cheap depth-wise convolution generates the remaining maps from them; the two sets
    are concatenated along the channel axis. Input and output shapes are identical to
    those of the ``nn.Conv2d`` being replaced.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, bias: bool = False,
                 ratio: int = 2, dw_kernel: int = 3):
        super().__init__()
        assert ratio >= 2, "ratio must be >= 2, otherwise nothing is saved"
        self.out_channels = out_channels
        self.ratio = ratio

        init_channels = int(math.ceil(out_channels / ratio))
        new_channels = init_channels * (ratio - 1)

        # Intrinsic features: same kernel / stride / padding / dilation as the original
        # convolution so that the spatial size of the output matches.
        self.primary = nn.Conv2d(in_channels, init_channels, kernel_size,
                                 stride=stride, padding=padding,
                                 dilation=dilation, bias=bias)
        # Cheap operation: depth-wise convolution, stride 1, "same" padding.
        self.cheap = nn.Conv2d(init_channels, new_channels, dw_kernel,
                               stride=1, padding=dw_kernel // 2,
                               groups=init_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.primary(x)
        y2 = self.cheap(y1)
        return torch.cat([y1, y2], dim=1)[:, : self.out_channels]


def _should_replace(module: nn.Conv2d, min_out_channels: int) -> bool:
    if not isinstance(module, nn.Conv2d):
        return False
    if isinstance(module, GhostConv2d):
        return False
    if module.groups != 1:          # already a grouped / depth-wise convolution
        return False
    if module.kernel_size == (1, 1):  # 1x1 convolutions are cheap already
        return False
    if module.out_channels < min_out_channels:  # very narrow layers: not worth splitting
        return False
    return True


def convert_to_ghost(net: nn.Module, ratio: int = 2, min_out_channels: int = 16,
                     skip_keywords: Tuple[str, ...] = ("seg_layers", "seg_output",
                                                       "transpconv")) -> List[str]:
    """Replace every eligible ``nn.Conv2d`` inside ``net`` by a ``GhostConv2d`` in place.

    Segmentation heads (``seg_layers``) and transposed convolutions are skipped: the
    former are 1x1 output layers, the latter perform the up-sampling.

    Returns the list of fully qualified names of the replaced modules so that the
    number of replacements can be checked in the training log.
    """
    to_replace = []
    for parent_name, parent in net.named_modules():
        if any(k in parent_name for k in skip_keywords):
            continue
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(k in full_name for k in skip_keywords):
                continue
            if _should_replace(child, min_out_channels):
                to_replace.append((parent, child_name, full_name, child))

    # The same nn.Conv2d object can be reachable through several attribute paths: nnU-Net's
    # ConvDropoutNormReLU stores it both as `.conv` and as `.all_modules[0]`. Exactly one
    # GhostConv2d must be created per convolution object and every reference must point to
    # that instance; otherwise the network would carry a second, unused copy of the
    # parameters and the parameter count would be inflated.
    ghosts = {}          # id(conv) -> GhostConv2d
    replaced = []
    for parent, child_name, full_name, conv in to_replace:
        ghost = ghosts.get(id(conv))
        if ghost is None:
            ghost = GhostConv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                bias=conv.bias is not None, ratio=ratio,
            )
            ghost = ghost.to(conv.weight.device, conv.weight.dtype)
            ghosts[id(conv)] = ghost
            replaced.append(full_name)
        setattr(parent, child_name, ghost)
    return replaced


def count_ghost_modules(net: nn.Module) -> int:
    return sum(1 for m in net.modules() if isinstance(m, GhostConv2d))
