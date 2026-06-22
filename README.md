# CrackLite

Official PyTorch implementation for **CrackLite: Lightweight Concrete Crack
Segmentation with Direction-Guided Topology Aggregation and Local Geometry
Refinement**.

CrackLite is a lightweight binary segmentation network for concrete crack
inspection images. It decouples crack morphology modeling into two parts:

- **Direction-Guided Topology Aggregation (DGTA)** propagates features along
  four candidate crack directions: `0`, `pi/4`, `pi/2`, and `3pi/4`.
- **Normal-Calibrated Local Geometry Refinement (NLGR)** uses normal-direction
  cues to sharpen weak crack boundaries and suppress crack-like texture noise.
- A **training-only structure-aware auxiliary branch** predicts centerline and
  boundary maps during optimization. The auxiliary heads are not used during
  inference, so deployment uses only the main segmentation branch.

## Repository Structure

```text
CrackLite/
├── model.py              # CrackLite, DGTA, NLGR, decoder, ablation switches
├── losses.py             # BCE + Dice + soft-clDice + auxiliary losses
├── dataload.py           # image-mask dataset loader
├── train.py              # training with AMP and best-mIoU checkpointing
├── evaluate.py           # F1 / mIoU evaluation with threshold 0.5
├── predict.py            # inference and prediction mask export
├── calc_complexity.py    # Params / FLOPs / FPS measurement
├── config.py             # paths and manuscript hyperparameters
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The manuscript experiments used Python 3.9, CUDA 11.2, and an NVIDIA RTX 3090.

## Data Layout

By default, the code expects this structure:

```text
CrackLite/data/
├── train/
│   ├── image/
│   └── label/
├── val/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

Images and masks are paired by filename stem, so `image/0001.jpg` can match
`label/0001.png`. To keep data outside the repository, set:

```bash
set CRACKLITE_DATA_DIR=D:\path\to\data
```

## Training

```bash
python train.py
```

Important manuscript settings are already in `config.py`:

- input size: `1024 x 1024`
- optimizer: Adam
- learning rate: `1e-3`
- batch size: `6`
- epochs: `100`
- DGTA tangent strip length: `15`
- DGTA normal strip length: `5`
- loss weights: `1.0 BCE + 1.0 Dice + 0.5 clDice + 0.3 centerline + 0.3 boundary`

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

This reports parameter count, FLOPs when `thop` is available, and FPS with
batch size 1.

## Manuscript Results

The manuscript reports the following CrackLite results:

| Dataset | F1 | mIoU |
| --- | ---: | ---: |
| Bridge Crack | 0.8655 | 0.7862 |
| Crack500 | 0.8824 | 0.8037 |
| Concrete-Crack-Segmentation | 0.9247 | 0.8684 |

The reported deployment profile is **3.264M parameters**, **54.859 GFLOPs**,
**95.21 FPS**, and **0.78G inference memory** under the tested GPU setting.

## Ablation Variants

`CrackLite` can instantiate the four manuscript variants:

```python
from model import CrackLite

backbone = CrackLite(use_dgta=False, use_nlgr=False)
dgta_only = CrackLite(use_dgta=True, use_nlgr=False)
nlgr_only = CrackLite(use_dgta=False, use_nlgr=True)
full = CrackLite(use_dgta=True, use_nlgr=True)
```
