# Checkpoints

This directory documents expected CrackLite checkpoints. Trained weights are not
redistributed in this repository.

Expected local paths:

```text
outputs/checkpoint/checkpoint_best.pth.tar   # selected by validation mIoU
outputs/checkpoint/checkpoint_last.pth.tar   # latest training state
```

To train from scratch:

```bash
python train.py
```

To evaluate an externally provided checkpoint, place it at
`outputs/checkpoint/checkpoint_best.pth.tar` or update `CHECKPOINT_PATH` in
`config.py`.

When publishing a checkpoint, also publish its SHA256 checksum, training commit,
dataset split, and exact command.
