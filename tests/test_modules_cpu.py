#!/usr/bin/env python3
"""CPU smoke tests for the modules (no data, no GPU, a few seconds).

    python tests/test_modules_cpu.py
"""
import sys

import torch
import torch.nn as nn

from ghostcil_nnunet.ghost import GhostConv2d, convert_to_ghost, count_ghost_modules
from ghostcil_nnunet.attention import (MultiDomainAttention, insert_attention_into_encoder,
                                       RadialFrequencyFilter)
from ghostcil_nnunet.losses import SAHLLoss
from ghostcil_nnunet.baselines import PlainUNet, AttentionUNet, infer_io_channels

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        FAILED.append(name)


def test_ghost_shape():
    for k, s, p in [(3, 1, 1), (3, 2, 1), (5, 1, 2)]:
        ref = nn.Conv2d(16, 32, k, stride=s, padding=p, bias=False)
        gc = GhostConv2d(16, 32, k, stride=s, padding=p, bias=False)
        x = torch.randn(2, 16, 32, 28)
        assert ref(x).shape == gc(x).shape, f"k={k} s={s}: shape mismatch"


def test_ghost_fewer_params():
    ref = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    gc = GhostConv2d(64, 128, 3, padding=1, bias=False)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_gc = sum(p.numel() for p in gc.parameters())
    assert n_gc < n_ref, f"Ghost convolution does not reduce the parameters: {n_gc} vs {n_ref}"
    print(f"       ({n_ref} -> {n_gc}, {n_gc / n_ref:.1%})")


def test_attention_shape_and_grad():
    for c, h, w in [(16, 32, 28), (64, 16, 16), (32, 15, 13)]:  # includes odd sizes
        att = MultiDomainAttention(c)
        x = torch.randn(2, c, h, w, requires_grad=True)
        y = att(x)
        assert y.shape == x.shape, f"C={c}: shape changed to {y.shape}"
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "non-finite gradient"


def test_radial_filter_identity_and_selectivity():
    """All-pass initialisation must be the identity; raising the high-frequency gains must sharpen edges."""
    f = RadialFrequencyFilter(n_bins=16)
    x = torch.randn(2, 1, 24, 20)
    y = f(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4), f"all-pass initialisation is not the identity, max error {(y-x).abs().max():.2e}"
    step = torch.zeros(1, 1, 32, 32); step[:, :, :, 16:] = 1.0
    with torch.no_grad():
        f.gain[8:] = 3.0
    hp = f(step)
    g0 = (step[..., 1:] - step[..., :-1]).abs().sum()
    g1 = (hp[..., 1:] - hp[..., :-1]).abs().sum()
    assert g1 > g0, "edge response did not increase after raising the high-frequency gains"
    assert sum(p.numel() for p in f.parameters()) == 16, "the radial filter must have exactly K=16 parameters"
    assert f(torch.randn(1, 1, 15, 9)).shape == (1, 1, 15, 9)     # size independent


def test_attention_branches_toggle():
    x = torch.randn(2, 16, 16, 16)
    for kw in [dict(use_freq=False), dict(use_spatial=False),
               dict(use_channel=False), dict(use_coord=False)]:
        y = MultiDomainAttention(16, **kw)(x)
        assert y.shape == x.shape


def test_loss_values():
    b, c, h, w = 2, 2, 16, 16
    logits = torch.randn(b, c, h, w, requires_grad=True)
    target = (torch.rand(b, 1, h, w) > 0.7).long()
    for kw in [dict(), dict(use_focal=False, use_boundary=False),
               dict(use_boundary=False), dict(use_focal=False), dict(focal_alpha=0.75)]:
        loss = SAHLLoss(**kw)(logits, target)
        assert loss.ndim == 0, "loss is not a scalar"
        assert torch.isfinite(loss), f"non-finite loss: {loss}"
        loss.backward(retain_graph=True)


def test_loss_perfect_prediction():
    """The Dice term must approach 0 for a perfect prediction (checks the sign of the loss)."""
    b, h, w = 1, 32, 32
    target = torch.zeros(b, 1, h, w, dtype=torch.long)
    target[:, :, 8:24, 8:24] = 1
    logits = torch.full((b, 2, h, w), -10.0)
    logits[:, 0][target[:, 0] == 0] = 10.0
    logits[:, 1][target[:, 0] == 1] = 10.0
    l_dice = SAHLLoss(use_focal=False, use_boundary=False, alpha=1.0)(logits, target)
    l_bad = SAHLLoss(use_focal=False, use_boundary=False,
                     alpha=1.0)(torch.zeros_like(logits), target)
    assert l_dice < 0.05, f"Dice loss of a perfect prediction should be close to 0, got {l_dice.item():.4f}"
    assert l_bad > l_dice, "loss has the wrong sign"
    print(f"       (perfect {l_dice.item():.4f} < uninformative {l_bad.item():.4f})")


def test_convert_on_toy_unet():
    """Replacement and insertion logic on a toy U-Net, independent of nnU-Net."""
    class Stage(nn.Sequential):
        def __init__(self, cin, cout):
            super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                             nn.InstanceNorm2d(cout), nn.LeakyReLU(1e-2))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([Stage(1, 32), Stage(32, 64), Stage(64, 128)])
            self.output_channels = [32, 64, 128]

        def forward(self, x):
            for s in self.stages:
                x = s(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.seg_layers = nn.ModuleList([nn.Conv2d(128, 2, 1)])

        def forward(self, x):
            return self.seg_layers[0](self.encoder(x))

    net = Net()
    x = torch.randn(1, 1, 32, 32)
    before = net(x).shape

    replaced = convert_to_ghost(net)
    assert len(replaced) == 3, f"expected 3 replaced 3x3 convolutions, got {replaced}"
    assert count_ghost_modules(net) == len(replaced)
    assert all("seg_layers" not in r for r in replaced), "the segmentation head must not be replaced"

    inserted = insert_attention_into_encoder(net)
    assert inserted == 3, f"expected 3 attention modules, got {inserted}"

    after = net(x)
    assert after.shape == before, f"output shape changed: {after.shape} vs {before}"
    after.sum().backward()


def test_baselines_shape():
    x = torch.randn(1, 1, 64, 48)
    for cls in (PlainUNet, AttentionUNet):
        net = cls(1, 2, base=8, depth=4)
        y = net(x)
        assert y.shape == (1, 2, 64, 48), f"{cls.__name__} output shape {y.shape}"
        y.sum().backward()
        assert hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision")
    n64 = sum(p.numel() for p in PlainUNet(1, 2, base=64, depth=5).parameters())
    assert 30e6 < n64 < 32e6, f"the original U-Net should have about 31M parameters, got {n64/1e6:.1f}M"
    # infer channels from an nnU-Net-style network
    class Dec(nn.Module):
        def __init__(self):
            super().__init__(); self.seg_layers = nn.ModuleList([nn.Conv2d(8, 2, 1)])
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = nn.Conv2d(1, 8, 3); self.decoder = Dec()
    assert infer_io_channels(Net()) == (1, 2)


def test_half_width_plans():
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "scripts"))
    from make_half_width_plans import halve_plans
    plans_new = {"plans_name": "nnUNetPlans", "configurations": {"2d": {
        "data_identifier": "nnUNetPlans_2d",
        "architecture": {"network_class_name": "x", "arch_kwargs": {
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512]}}}}}
    out, (k, old, new) = halve_plans(plans_new, "2d", "nnUNetPlans_half")
    assert out["plans_name"] == "nnUNetPlans_half"
    assert out["configurations"]["2d"]["architecture"]["arch_kwargs"]["features_per_stage"] == \
        [16, 32, 64, 128, 256, 256, 256]
    assert plans_new["configurations"]["2d"]["architecture"]["arch_kwargs"]["features_per_stage"][0] == 32, "original plans were modified"
    plans_old = {"plans_name": "nnUNetPlans", "configurations": {"2d": {
        "UNet_base_num_features": 32, "unet_max_num_features": 512}}}
    out, _ = halve_plans(plans_old, "2d", "nnUNetPlans_half")
    assert out["configurations"]["2d"]["UNet_base_num_features"] == 16
    assert out["configurations"]["2d"]["unet_max_num_features"] == 256


def test_ghost_shared_conv():
    """nnU-Net's ConvDropoutNormReLU references the same conv as .conv and as .all_modules[0];
    both references must point to one GhostConv2d and the parameter count must not double."""
    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 3, padding=1)
            self.all_modules = nn.Sequential(self.conv, nn.LeakyReLU())

        def forward(self, x):
            return self.all_modules(x)

    net = nn.Sequential(Block(16, 32), Block(32, 32))
    replaced = convert_to_ghost(net)
    assert len(replaced) == 2, f"expected 2 replaced convolution objects, got {len(replaced)}"
    for blk in net:
        assert isinstance(blk.conv, GhostConv2d)
        assert blk.all_modules[0] is blk.conv, "both references must be the same Ghost instance"
    n_expect = sum(sum(p.numel() for p in GhostConv2d(ci, 32, 3, padding=1, bias=True).parameters())
                   for ci in (16, 32))
    n_net = sum(p.numel() for p in net.parameters())
    assert n_net == n_expect, f"parameter count {n_net} != {n_expect}: duplicated dead parameters"
    y = net(torch.randn(1, 16, 8, 8))
    assert y.shape == (1, 32, 8, 8)


def test_trainer_signatures():
    """nnU-Net inspects the signatures of __init__ and build_network_architecture; ours must match the parent."""
    try:
        import nnunetv2  # noqa: F401
    except ImportError:
        print("       (nnunetv2 not installed, skipped)")
        return
    import inspect
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from ghostcil_nnunet import trainers as T
    p_init = list(inspect.signature(nnUNetTrainer.__init__).parameters)
    p_build = list(inspect.signature(nnUNetTrainer.build_network_architecture).parameters)
    for name in T.__all__:
        cls = getattr(T, name)
        assert list(inspect.signature(cls.__init__).parameters) == p_init, f"{name}.__init__ signature differs"
        assert list(inspect.signature(cls.build_network_architecture).parameters) == p_build, \
            f"{name}.build_network_architecture signature differs"
    print(f"       ({len(T.__all__)} trainers match the nnU-Net parent signatures)")


def test_loss_half_precision_safe():
    """The Sobel term must have a finite backward pass on flat fp16 inputs (mixed-precision safety)."""
    from ghostcil_nnunet.losses import _sobel_gradient_magnitude
    x = torch.full((1, 1, 16, 16), 0.3, dtype=torch.float16, requires_grad=True)
    g = _sobel_gradient_magnitude(x)
    g.mean().backward()
    assert torch.isfinite(x.grad).all(), "NaN/inf in the backward pass on a flat fp16 region"
    loss_fn = SAHLLoss(batch_dice=True)
    logits = torch.zeros(2, 2, 24, 24, dtype=torch.float16)          # completely flat output
    logits[:, 1, 8:16, 8:16] = 4.0
    logits.requires_grad_(True)
    target = torch.zeros(2, 1, 24, 24, dtype=torch.long); target[:, :, 6:18, 6:18] = 1
    l = loss_fn(logits, target)
    l.backward()
    assert torch.isfinite(l) and torch.isfinite(logits.grad).all(), "SAHL produced NaN with fp16 inputs"
    assert l.dtype == torch.float32, "the loss must be computed in float32"


def main():
    torch.manual_seed(0)
    print("Ghost convolution")
    check("output shape equals the standard convolution", test_ghost_shape)
    check("fewer parameters", test_ghost_fewer_params)
    print("Multi-domain attention")
    check("shape preserved and gradients flow", test_attention_shape_and_grad)
    check("radial filter: identity init / frequency selectivity / size independence", test_radial_filter_identity_and_selectivity)
    check("branches can be switched off individually", test_attention_branches_toggle)
    print("SAHL loss")
    check("all term combinations are finite and differentiable", test_loss_values)
    check("perfect prediction gives near-zero loss", test_loss_perfect_prediction)
    check("Sobel term and full loss are finite in fp16 (mixed-precision regression)", test_loss_half_precision_safe)
    print("Network modification")
    check("replacement + insertion on a toy U-Net", test_convert_on_toy_unet)
    check("one Ghost instance for a convolution referenced twice", test_ghost_shared_conv)
    check("trainer signatures match nnU-Net (needs nnunetv2)", test_trainer_signatures)
    print("Baselines")
    check("U-Net / Attention U-Net shapes and parameter count", test_baselines_shape)
    print("Half-width control")
    check("plans channel halving (both nnU-Net plans formats)", test_half_width_plans)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
