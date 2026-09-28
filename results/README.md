# Results reported in the paper

All numbers in the paper were computed from the files in this folder. Nothing was
entered by hand.

| File | Content |
|---|---|
| `ext_<method>__fold<k>.json` | per-slice Dice / IoU / HD95 of fold model *k* on the external test set (P13, P16, P30, P38; 120 in-vivo slices, 5 hospitals, 4 scanners) |
| `int_<method>__fold<k>.json` | per-slice metrics of fold model *k* on its own validation subjects (internal subject-level 3-fold CV, 9 ex-vivo subjects, 576 slices) |
| `external_summary.txt` | `scripts/evaluate.py --compare "results/ext_*.json" --level external` |
| `internal_summary.txt` | `scripts/evaluate.py --compare "results/int_*.json" --level internal` |
| `complexity.json` | parameters, MAC-FLOPs and single-slice latency of every network (`scripts/complexity.py`) |
| `radial_gain.json` | radial gains learned by the frequency branch of the attention module in the ablation models (`scripts/dump_radial_gain.py`) |
| `env.txt` | software / hardware environment of the runs (`scripts/env_info.py`) |
| `training_logs/` | the nnU-Net training log of every one of the 25 trainings |

HD95 is given in pixels (the PNG release of CT-SCOPE carries no pixel spacing).
Predictions use one fold model at a time, no ensembling, no test-time augmentation.
All configurations were trained for 150 epochs with batch size 2 and a 512 x 448
patch on one RTX 3080 Ti (11.6 GB), nnU-Net v2.8.1, PyTorch 2.12.1.

## Method labels

| Result label | Trainer | Plans | Role in the paper |
|---|---|---|---|
| `nnU-Net_v2` | `nnUNetTrainerBaseline150` | `nnUNetPlans` | baseline |
| `nnU-Net_v2_half` | `nnUNetTrainerBaseline150` | `nnUNetPlans_half` | half-width control |
| `Ghost_CIL-nnUNet` | `nnUNetTrainerGhostCIL` | `nnUNetPlans` | **proposed method** |
| `Ghost_CIL+MDA` | `nnUNetTrainerGhostCIL_MDA` | `nnUNetPlans` | ablation: + multi-domain attention (Dice + CE) |
| `GMA-nnUNet` | `nnUNetTrainerGMA` | `nnUNetPlans` | ablation: + multi-domain attention + SAHL |
| `GMA_DiceOnly` / `GMA_DiceFocal` / `GMA_DiceBoundary` | `nnUNetTrainerGMA_*` | `nnUNetPlans` | loss ablation, fold 0 only (Table 4) |
| `GMA_NoFreq` | `nnUNetTrainerGMA_NoFreq` | `nnUNetPlans` | attention without frequency branch, fold 0 only |
| `U-Net` / `Attention_U-Net` | `nnUNetTrainerUNet` / `nnUNetTrainerAttUNet` | `nnUNetPlans` | same-pipeline baselines |

The label `GMA` (Ghost + multi-domain attention) is the name the ablation
configurations carried during the experiments; it is kept so that the result files,
training logs and trainer names stay consistent. In `complexity.json` the label
`GMA-nnUNet(...)` written by the original run was shortened to `GMA-nnUNet`; the
numbers are untouched.
