# CrackLite

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model Card](https://img.shields.io/badge/model-card-informational)](MODEL_CARD.md)
[![Citation](https://img.shields.io/badge/citation-CFF-lightgrey)](CITATION.cff)

Official PyTorch implementation for **CrackLite: Lightweight Concrete Crack
Segmentation with Direction-Guided Topology Aggregation and Local Geometry
Refinement**.

CrackLite is designed for field concrete crack inspection, where cracks are
thin, elongated, low-contrast, and easily confused with stains, joints, pores,
shadows, and heterogeneous concrete texture. Instead of treating crack
segmentation as generic foreground extraction, CrackLite explicitly separates
two morphology-sensitive objectives:

- **Topology continuity:** keep long and weak crack paths connected.
- **Local geometry fidelity:** keep crack boundaries slender and suppress
  crack-like background texture.

## Highlights

- **Direction-Guided Topology Aggregation (DGTA):** propagates features along
  candidate crack directions `0`, `pi/4`, `pi/2`, and `3pi/4`.
- **Normal-Calibrated Local Geometry Refinement (NLGR):** uses normal-direction
  cues to refine weak boundaries and suppress texture-induced false responses.
- **Training-only auxiliary supervision:** centerline and boundary heads
  regularize structure during training without increasing inference cost.
- **Compact deployment profile:** 3.264M parameters, 54.859 GFLOPs, and 95.21
  FPS under the reported GPU-side setting.
- **Reproducible release package:** configs, model card, citation metadata,
  checkpoint manifest, dataset protocol, demo data, and command wrappers.

## Performance-Complexity Overview

Fig. 1 shows CrackLite in the accuracy-complexity operating region of the Bridge
Crack test set. The bubble size represents GFLOPs.
<div align="center"><img src="assets/figures/fig1_performance_complexity.png" alt="Performance-complexity comparison on the Bridge Crack test set" width="86%"></div>

## Visual Overview

The full CrackLite network follows a lightweight encoder-decoder design. The
decoder fuses multi-scale features and the auxiliary centerline/boundary branch
is removed during inference.
<div align="center"><img src="assets/figures/fig2_architecture.png" alt="Overall CrackLite architecture" width="100%"></div>

The core block contains DGTA and NLGR. DGTA estimates a local orientation prior
and performs tangent/normal directional strip aggregation. NLGR then uses the
normal cue as a geometry gate for local 3x3/5x5 refinement.
<div align="center"><img src="assets/figures/fig3_cracklite_block.png" alt="CrackLite block with DGTA and NLGR" width="100%"></div>

## Results

CrackLite is evaluated on one self-collected field bridge dataset and two public
benchmarks. The main reporting metrics are F1 and class-averaged mIoU with a
fixed threshold of `0.5`.
<div align="center"><img src="assets/figures/fig4_dataset_miou.png" alt="mIoU comparison across the three crack datasets" width="100%"></div>

The qualitative examples show the intended behavior: CrackLite aims to recover
continuous crack trajectories while keeping masks thin and restrained around
texture clutter.
<div align="center"><img src="assets/figures/fig5_qualitative_comparison.png" alt="Qualitative crack segmentation comparison" width="100%"></div>

## Internal Response Interpretation

The manuscript visualizes intermediate DGTA/NLGR responses to explain how the
network evolves from diffuse crack-like activation toward concentrated crack
paths, sharper local boundaries, and reduced false responses.
<div align="center"><img src="assets/figures/fig6_internal_response.png" alt="DGTA and NLGR internal response evolution" width="88%"></div>

## Efficiency

The model is positioned as an accuracy-efficiency trade-off for practical crack
inspection rather than as a heavyweight accuracy-only model.
<div align="center"><img src="assets/figures/fig8_efficiency_tradeoff.png" alt="Throughput and complexity trade-off" width="100%"></div>

## Release Contents

```text
CrackLite/
  assets/                         # manuscript and generated release figures
  checkpoints/
    README.md
    manifest.json                 # expected checkpoint metadata
  configs/
    paper_cracklite.json          # manuscript-default settings
    demo_cpu.json                 # smoke-test setting record
  demo_data/
    manifest.csv                  # synthetic mini data manifest
    train/, val/, test/           # tiny generated smoke-test dataset
  docs/
    DATASET_PROTOCOL.md
    REPRODUCIBILITY.md
    MODEL_CARD.md
    results/
  scripts/
    run_cracklite.ps1
    run_cracklite.sh
  tools/
    make_demo_dataset.py
    train.py
    evaluate.py
    predict.py
    complexity.py
  model.py
  losses.py
  dataload.py
  train.py
  evaluate.py
  predict.py
  calc_complexity.py
  config.py
  CITATION.cff
  MODEL_CARD.md
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  requirements.txt
```

## Installation

Create an environment:

```bash
git clone https://github.com/sisisichen/CrackLite.git
cd CrackLite

python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your CUDA/CPU platform first, then install the remaining
dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The manuscript experiments used Python 3.9, CUDA 11.2, and an NVIDIA RTX 3090.

## Data Layout

By default, the code expects this structure:

```text
data/
  train/
    image/
    label/
  val/
    image/
    label/
  test/
    image/
    label/
```

Images and masks are paired by filename stem, so `image/0001.jpg` can match
`label/0001.png`. See [docs/DATASET_PROTOCOL.md](docs/DATASET_PROTOCOL.md) for
mask polarity and split details.

To keep data outside the repository:

```powershell
$env:CRACKLITE_DATA_DIR = "D:\path\to\data"
```

## Quick Smoke Test

The included `demo_data/` directory is a tiny synthetic dataset for checking the
software path only. It is not used for manuscript metrics.

Regenerate it if needed:

```bash
python tools/make_demo_dataset.py --out demo_data --image_size 128 --train 8 --val 2 --test 2 --overwrite
```

For a full training run, use your real dataset. To point the scripts at the demo
data for local debugging, set:

```powershell
$env:CRACKLITE_DATA_DIR = "demo_data"
```

## Training

```bash
python train.py
```

or through the release wrapper:

```bash
python tools/train.py
```

Important manuscript settings are recorded in
[configs/paper_cracklite.json](configs/paper_cracklite.json) and mirrored in
`config.py`.

| Setting | Value |
| --- | --- |
| Input size | `1024 x 1024` |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Batch size | `6` |
| Epochs | `100` |
| DGTA directions | `0`, `pi/4`, `pi/2`, `3pi/4` |
| DGTA tangent / normal length | `15 / 5` |
| Loss weights | `1.0 BCE + 1.0 Dice + 0.5 clDice + 0.3 centerline + 0.3 boundary` |

The best checkpoint is selected by validation class-averaged mIoU and saved to:

```text
outputs/checkpoint/checkpoint_best.pth.tar
```

## Evaluation

```bash
python evaluate.py
```

The evaluation script uses the manuscript reporting rule: probability maps are
thresholded at `0.5`, and the main metrics are F1 and class-averaged mIoU.

## Inference

```bash
python predict.py
```

Predicted binary masks are saved to:

```text
outputs/predictions/
```

## Complexity

```bash
python calc_complexity.py
```

This reports parameter count, FLOPs when `thop` is available, and FPS with batch
size 1. The custom DGTA directional strip operator is counted by
`calc_complexity.py`.

## Checkpoints

Trained weights are not redistributed in this repository. See
[checkpoints/README.md](checkpoints/README.md) and
[checkpoints/manifest.json](checkpoints/manifest.json) for expected paths and
metadata.

When publishing a checkpoint, provide the SHA256 checksum, dataset split,
training command, and Git commit hash.

## Reproducibility

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). At minimum, record the
Git commit, dataset split, input size, threshold, checkpoint hash, PyTorch/CUDA
versions, GPU model, and exact command.

## Third-party Code

No third-party source code is vendored in this release. Dependencies retain
their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

CrackLite source code and release metadata are provided under the MIT License.

## Citation

GitHub renders citation metadata from [CITATION.cff](CITATION.cff).

```bibtex
@misc{bao2026cracklite,
  title  = {CrackLite: Lightweight Concrete Crack Segmentation with Direction-Guided Topology Aggregation and Local Geometry Refinement},
  author = {Bao, Longsheng and Chen, Si and Bao, Yuyang and Li, Baoxian and Zhao, Jiakang and Yu, Ling},
  year   = {2026},
  note   = {Manuscript}
}
```
