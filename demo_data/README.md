# CrackLite Demo Data

This directory contains a tiny synthetic crack segmentation dataset for software
smoke testing. It is not used for manuscript metrics.

The data was generated with:

```bash
python tools/make_demo_dataset.py --out demo_data --image_size 128 --train 8 --val 2 --test 2 --overwrite
```

Layout:

```text
demo_data/
  train/image, train/label
  val/image, val/label
  test/image, test/label
  manifest.csv
```

Each mask uses white crack pixels on a black background, matching the default
CrackLite loader.
