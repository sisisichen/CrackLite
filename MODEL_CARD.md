# CrackLite Model Card

## Model Details

CrackLite is a lightweight binary segmentation model for concrete crack images.
The released code implements the manuscript architecture with Direction-Guided
Topology Aggregation (DGTA), Normal-Calibrated Local Geometry Refinement (NLGR),
and training-only centerline and boundary auxiliary heads.

The repository does not redistribute trained weights. Users can train a model
with `python train.py` or place an externally provided checkpoint under
`outputs/checkpoint/checkpoint_best.pth.tar`.

## Intended Use

CrackLite is intended for research and prototyping in pixel-level concrete crack
segmentation. Typical use cases include bridge, pavement, tunnel, retaining
wall, and other concrete surface inspection images where thin crack continuity
and local boundary quality matter.

## Out-of-Scope Use

The model should not be used as the sole basis for safety-critical maintenance,
load-rating, or public-risk decisions. Field deployment should include qualified
human review, site-specific calibration, and independent validation.

## Training and Evaluation Context

The manuscript evaluates CrackLite on:

| Dataset | F1 | mIoU |
| --- | ---: | ---: |
| Bridge Crack | 0.8655 | 0.7862 |
| Crack500 | 0.8824 | 0.8037 |
| Concrete-Crack-Segmentation | 0.9247 | 0.8684 |

The reported deployment profile is 3.264M parameters, 54.859 GFLOPs, 95.21 FPS,
and 0.78G inference memory under the tested GPU-side setting.

## Limitations

- DGTA uses a discrete direction set and may be less adaptive at highly
  irregular junctions or abrupt orientation changes.
- Extremely faint crack tips, severe blur, stains, joints, and elongated rough
  texture can still cause false negatives or false positives.
- The current release focuses on segmentation metrics; topology-sensitive
  evaluation should be added for deployment studies.
- Embedded-device performance has not been validated by this repository.

## Responsible Release Notes

When publishing results, report the Git commit hash, dataset split, input size,
threshold, checkpoint path or hash, PyTorch/CUDA versions, GPU model, random
seed, and exact training/evaluation command.
