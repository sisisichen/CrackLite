# Dataset Protocol

CrackLite uses paired image-mask data for binary concrete crack segmentation.
Images and masks are paired by filename stem.

## Layout

```text
data/
  train/
    image/*.png
    label/*.png
  val/
    image/*.png
    label/*.png
  test/
    image/*.png
    label/*.png
```

The loader supports `jpg`, `jpeg`, `png`, `bmp`, `tif`, and `tiff` files.
For example, `image/0001.jpg` can pair with `label/0001.png`.

## Mask Convention

Masks are read as grayscale and binarized with a threshold of 127:

```text
pixel > 127  -> crack foreground
pixel <= 127 -> background
```

If your annotations use dark crack pixels on a light background, invert the masks
before training or adapt `dataload.py` for that dataset.

## Splits Used in the Manuscript

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Bridge Crack | 1360 | 300 | 340 |
| Crack500 | 2700 | 300 | 295 |
| Concrete-Crack-Segmentation | 1360 | 240 | 240 |

## Leakage Rule

Ground-truth masks, skeletons, and boundary maps may be used only for training
losses and labeled evaluation. Normal prediction uses input images only.

## Environment Variable

To keep datasets outside the repository:

```powershell
$env:CRACKLITE_DATA_DIR = "D:\path\to\data"
```

or on Linux/macOS:

```bash
export CRACKLITE_DATA_DIR=/path/to/data
```
