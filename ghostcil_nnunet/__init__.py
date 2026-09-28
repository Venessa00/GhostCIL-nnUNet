"""GhostCIL-nnUNet.

Ghost-convolution lightweight nnU-Net for CT segmentation of the osseous structures
surrounding the paranasal sinuses, together with the components that are evaluated
as ablations in the paper (multi-domain attention, SAHL loss) and the same-pipeline
baselines (U-Net, Attention U-Net, half-width nnU-Net, external networks).
"""
from ghostcil_nnunet.ghost import GhostConv2d, convert_to_ghost, count_ghost_modules
from ghostcil_nnunet.attention import MultiDomainAttention, insert_attention_into_encoder
from ghostcil_nnunet.losses import SAHLLoss
from ghostcil_nnunet.baselines import PlainUNet, AttentionUNet

__all__ = [
    "GhostConv2d", "convert_to_ghost", "count_ghost_modules",
    "MultiDomainAttention", "insert_attention_into_encoder",
    "SAHLLoss", "PlainUNet", "AttentionUNet",
]
__version__ = "1.0.0"
