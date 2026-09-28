"""Custom nnU-Net v2 trainers used in the paper.

Design: nnU-Net's own network-building logic is not re-implemented. The parent class
builds the automatically configured network first; the trainers then modify it in
place (replace convolutions, insert attention) or exchange it for a baseline network.
Everything else (pre-processing, augmentation, optimiser, schedule) is nnU-Net's.

Signature mirroring: nnU-Net inspects the signature of ``__init__`` (to read its own
arguments back from ``locals()``) and of ``build_network_architecture`` (its
signature changed in nnU-Net 2.8). The overrides below therefore forward
``*args, **kwargs`` but expose the parent's signature through ``__signature__``
(see ``_mirror``), which keeps them compatible with nnU-Net 2.2 to 2.8.

Trainers (paper mapping)
  nnUNetTrainerBaseline150      nnU-Net v2, 150 epochs (baseline). With plans
                                ``nnUNetPlans_half`` it is the half-width control.
  nnUNetTrainerGhostCIL         Ghost_CIL-nnUNet  <- the proposed method
  nnUNetTrainerGhostCIL_MDA     ablation: + multi-domain attention (Dice + CE loss)
  nnUNetTrainerGMA              ablation: + multi-domain attention + SAHL loss
  nnUNetTrainerGMA_DiceOnly / _DiceFocal / _DiceBoundary
                                loss ablation on the Ghost + attention network
  nnUNetTrainerGMA_NoFreq       attention without the frequency branch, SAHL loss
  nnUNetTrainerUNet / nnUNetTrainerAttUNet
                                U-Net / Attention U-Net trained in the same pipeline
  nnUNetTrainerExtModel         third-party network selected through GCIL_EXT_* variables

Environment variables
  GCIL_EPOCHS      number of epochs (default 150)
  GCIL_ITERS       iterations per epoch (default: nnU-Net's 250; only for smoke tests)
  GCIL_VAL_ITERS   validation iterations per epoch (default 50; only for smoke tests)
"""
import inspect
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from ghostcil_nnunet.ghost import convert_to_ghost
from ghostcil_nnunet.attention import insert_attention_into_encoder
from ghostcil_nnunet.losses import SAHLLoss
from ghostcil_nnunet.baselines import PlainUNet, AttentionUNet, load_external, infer_io_channels

__all__ = [
    "nnUNetTrainerBaseline150",
    "nnUNetTrainerGhostCIL",
    "nnUNetTrainerGhostCIL_MDA",
    "nnUNetTrainerGMA",
    "nnUNetTrainerGMA_DiceOnly",
    "nnUNetTrainerGMA_DiceFocal",
    "nnUNetTrainerGMA_DiceBoundary",
    "nnUNetTrainerGMA_NoFreq",
    "nnUNetTrainerUNet",
    "nnUNetTrainerAttUNet",
    "nnUNetTrainerExtModel",
]


# ---------------------------------------------------------------------------
# Signature mirroring: inspect.signature(our function) == inspect.signature(parent function)
# ---------------------------------------------------------------------------
def _mirror(parent_func):
    sig = inspect.signature(parent_func)

    def deco(func):
        func.__signature__ = sig
        return func
    return deco


_PARENT_BUILD = nnUNetTrainer.build_network_architecture   # staticmethod -> plain function


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


class _EpochMixin:
    """Sets the number of epochs (default 150, overridable through the environment).
    This has to happen in ``__init__`` because ``initialize()`` builds the PolyLR
    scheduler from ``num_epochs``."""

    @_mirror(nnUNetTrainer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _env_int("GCIL_EPOCHS", 150)
        self.num_iterations_per_epoch = _env_int("GCIL_ITERS", self.num_iterations_per_epoch)
        self.num_val_iterations_per_epoch = _env_int("GCIL_VAL_ITERS", self.num_val_iterations_per_epoch)


class nnUNetTrainerBaseline150(_EpochMixin, nnUNetTrainer):
    """Unmodified nnU-Net v2 trained for 150 epochs, so that all configurations share the schedule."""
    pass


class nnUNetTrainerGhostCIL(_EpochMixin, nnUNetTrainer):
    """Ghost_CIL-nnUNet: in-place Ghost convolution replacement only (proposed method)."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        net = _PARENT_BUILD(*args, **kwargs)
        replaced = convert_to_ghost(net)
        print(f"[GhostCIL] replaced {len(replaced)} convolution layers by Ghost convolutions")
        return net


class nnUNetTrainerGhostCIL_MDA(_EpochMixin, nnUNetTrainer):
    """Ablation: Ghost convolutions + multi-domain attention, nnU-Net default Dice + CE loss."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        net = _PARENT_BUILD(*args, **kwargs)
        replaced = convert_to_ghost(net)
        inserted = insert_attention_into_encoder(net)
        print(f"[GhostCIL] replaced {len(replaced)} convolution layers, inserted {inserted} attention modules")
        assert inserted > 0, "encoder.stages not found; print(net) and adapt insert_attention_into_encoder"
        return net


class _SAHLMixin:
    """Replaces nnU-Net's default loss by SAHL; deep supervision is wrapped as in the parent."""

    SAHL_KWARGS: dict = {}

    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based labels are not supported"
        loss = SAHLLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ignore_label=self.label_manager.ignore_label,
            **self.SAHL_KWARGS,
        )
        if getattr(self, "enable_deep_supervision", True):
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = torch.tensor(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))],
                dtype=torch.float32)
            weights[-1] = 0            # lowest resolution gets no loss, as in nnU-Net
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerGMA(_SAHLMixin, _EpochMixin, nnUNetTrainer):
    """Ablation: Ghost convolutions + multi-domain attention + SAHL loss."""

    SAHL_KWARGS = dict(use_dice=True, use_focal=True, use_boundary=True)

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        net = _PARENT_BUILD(*args, **kwargs)
        replaced = convert_to_ghost(net)
        inserted = insert_attention_into_encoder(net)
        print(f"[GhostCIL] replaced {len(replaced)} convolution layers, inserted {inserted} attention modules")
        assert inserted > 0, "encoder.stages not found; print(net) and adapt insert_attention_into_encoder"
        return net


# ---- Loss ablation (paper Table 4): network fixed to Ghost + attention, loss terms switched ----

class nnUNetTrainerGMA_DiceOnly(nnUNetTrainerGMA):
    SAHL_KWARGS = dict(use_dice=True, use_focal=False, use_boundary=False, alpha=1.0)


class nnUNetTrainerGMA_DiceFocal(nnUNetTrainerGMA):
    SAHL_KWARGS = dict(use_dice=True, use_focal=True, use_boundary=False,
                       alpha=0.5, beta=0.5)


class nnUNetTrainerGMA_DiceBoundary(nnUNetTrainerGMA):
    SAHL_KWARGS = dict(use_dice=True, use_focal=False, use_boundary=True,
                       alpha=0.5, gamma=0.5)


# ---- Attention ablation: without the frequency branch ----

class nnUNetTrainerGMA_NoFreq(nnUNetTrainerGMA):
    """Ghost + attention with spatial / channel / coordinate branches only (no frequency branch), SAHL loss."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        net = _PARENT_BUILD(*args, **kwargs)
        replaced = convert_to_ghost(net)
        inserted = insert_attention_into_encoder(net, use_freq=False)
        print(f"[GhostCIL] replaced {len(replaced)} convolution layers, inserted {inserted} three-branch attention modules (no frequency branch)")
        assert inserted > 0, "encoder.stages not found; print(net) and adapt insert_attention_into_encoder"
        return net


# ---- Baselines: same pipeline, different network, deep supervision off (single output) ----

class _NoDeepSupervisionMixin:
    """Disables nnU-Net deep supervision; compatible with the v2.2 - v2.8 implementations."""

    @_mirror(nnUNetTrainer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_deep_supervision = False

    def _get_deep_supervision_scales(self):
        return None

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _build_loss(self):
        # Identical to nnU-Net's default loss (Dice + CE), just without the DeepSupervisionWrapper
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        assert not self.label_manager.has_regions, "region-based labels are not supported"
        return DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice, "smooth": 1e-5,
             "do_bg": False, "ddp": self.is_ddp},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)


def _build_replacing(factory, *args, **kwargs):
    """Build the reference nnU-Net network only to read its I/O channels, then replace it."""
    ref = _PARENT_BUILD(*args, **kwargs)
    in_ch, out_ch = infer_io_channels(ref)
    del ref
    net = factory(in_ch, out_ch)
    n = sum(p.numel() for p in net.parameters())
    print(f"[GhostCIL] baseline network {type(net).__name__}: in={in_ch} out={out_ch} params={n/1e6:.2f}M")
    return net


class nnUNetTrainerUNet(_NoDeepSupervisionMixin, _EpochMixin, nnUNetTrainer):
    """Original U-Net (base width 64)."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        return _build_replacing(lambda i, o: PlainUNet(i, o, base=64, depth=5), *args, **kwargs)


class nnUNetTrainerAttUNet(_NoDeepSupervisionMixin, _EpochMixin, nnUNetTrainer):
    """Attention U-Net (base width 64)."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        return _build_replacing(lambda i, o: AttentionUNet(i, o, base=64, depth=5), *args, **kwargs)


class nnUNetTrainerExtModel(_NoDeepSupervisionMixin, _EpochMixin, nnUNetTrainer):
    """Third-party network (UNeXt, CMUNeXt, ...) selected through GCIL_EXT_* variables, see baselines.py."""

    @staticmethod
    @_mirror(_PARENT_BUILD)
    def build_network_architecture(*args, **kwargs):
        return _build_replacing(load_external, *args, **kwargs)
