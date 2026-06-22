# Reproducibility Checklist

Record these items when reporting CrackLite results:

- Git commit hash.
- `configs/paper_cracklite.json`.
- Dataset name, source, split, and mask polarity.
- Input size and threshold.
- Checkpoint path and SHA256 checksum.
- PyTorch, torchvision, CUDA, Python, and GPU versions.
- Random seed if used by the experiment wrapper.
- Exact training, evaluation, prediction, and complexity commands.

Recommended commands:

```bash
python train.py
python evaluate.py
python predict.py
python calc_complexity.py
```

Keep private datasets, large checkpoints, and generated outputs out of git.
