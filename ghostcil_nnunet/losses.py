"""SAHL hybrid loss = Dice + Focal + gradient-difference (boundary) term.

This loss is used only in the loss ablation of the paper; the proposed
Ghost_CIL-nnUNet keeps nnU-Net's default Dice + CE loss. Each term can be switched
off independently, so all loss-ablation trainers are obtained through the switches
without touching the code.

The interface follows nnU-Net v2: ``forward(logits, target)`` where ``target`` has
shape (B, 1, H, W) and stores class indices.

Notes
* The focal term carries no class-balancing factor by default (``focal_alpha=None``).
  Pass e.g. ``focal_alpha=0.75`` to weight foreground by alpha and background by 1-alpha.
* The third term is the L1 difference between the Sobel gradient magnitudes of the
  foreground probability map and of the one-hot label, i.e. a gradient-difference loss
  in the sense of Mathieu et al. (ICLR 2016), not a distance-map based boundary loss.
  The attribute name ``use_boundary`` is kept for that term.
* The whole loss is computed in float32 with autocast disabled. nnU-Net trains with
  fp16 mixed precision on GPU; the epsilon inside sqrt(gx^2 + gy^2 + eps) underflows
  to zero in fp16, and the backward pass of sqrt(0) on flat regions produces NaN.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SAHLLoss"]


_SOBEL_EPS = 1e-6      # safe in fp32; 1e-8 underflows to 0 in fp16 and the backward pass yields NaN


def _sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of a (B,1,H,W) map, always computed in float32."""
    x = x.float()
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    with torch.autocast(device_type=x.device.type, enabled=False):
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + _SOBEL_EPS)


class SAHLLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4,
                 focal_alpha: Optional[float] = None, focal_gamma: float = 2.0,
                 smooth: float = 1e-5, batch_dice: bool = False,
                 use_dice: bool = True, use_focal: bool = True,
                 use_boundary: bool = True, ignore_label: Optional[int] = None):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.use_dice, self.use_focal, self.use_boundary = use_dice, use_focal, use_boundary
        self.ignore_label = ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # net_output: (B,C,H,W) logits; target: (B,1,H,W) long or (B,H,W)
        # The loss is evaluated in float32 with autocast disabled (see module docstring).
        with torch.autocast(device_type=net_output.device.type, enabled=False):
            return self._forward_fp32(net_output.float(), target)

    def _forward_fp32(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == net_output.ndim - 1:
            target = target.unsqueeze(1)
        target = target.long()

        num_classes = net_output.shape[1]
        probs = torch.softmax(net_output, dim=1)

        mask = None
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            target = torch.where(mask, target, torch.zeros_like(target))

        with torch.no_grad():
            onehot = torch.zeros_like(probs)
            onehot.scatter_(1, target, 1.0)
            if mask is not None:
                onehot = onehot * mask

        loss = net_output.new_zeros(())

        if self.use_dice:
            loss = loss + self.alpha * self._dice(probs, onehot, mask)
        if self.use_focal:
            loss = loss + self.beta * self._focal(probs, target, mask)
        if self.use_boundary:
            loss = loss + self.gamma * self._boundary(probs, onehot, num_classes)
        return loss

    def _dice(self, probs, onehot, mask):
        # Foreground classes only, background excluded (same as nnU-Net's do_bg=False)
        p = probs[:, 1:]
        g = onehot[:, 1:]
        if mask is not None:
            p = p * mask
        axes = list(range(2, p.ndim))
        if self.batch_dice:
            axes = [0] + axes
        inter = (p * g).sum(dim=axes)
        denom = p.sum(dim=axes) + g.sum(dim=axes)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def _focal(self, probs, target, mask):
        pt = probs.gather(1, target).clamp_min(1e-7)      # (B,1,H,W) probability of the true class
        w = (1.0 - pt).pow(self.focal_gamma)
        if self.focal_alpha is not None:                  # class balancing: alpha for foreground, 1-alpha for background
            alpha_t = torch.where(target > 0, torch.full_like(pt, self.focal_alpha),
                                  torch.full_like(pt, 1.0 - self.focal_alpha))
            w = w * alpha_t
        fl = -(w * torch.log(pt))
        if mask is not None:
            fl = fl * mask
            denom = mask.sum().clamp_min(1)
            return fl.sum() / denom
        return fl.mean()

    def _boundary(self, probs, onehot, num_classes):
        # L1 difference between the Sobel gradient magnitudes of the foreground
        # probability map and of the label map
        fg_p = probs[:, 1:].sum(dim=1, keepdim=True)
        fg_g = onehot[:, 1:].sum(dim=1, keepdim=True)
        gp = _sobel_gradient_magnitude(fg_p)
        gg = _sobel_gradient_magnitude(fg_g)
        return (gp - gg).abs().mean()
