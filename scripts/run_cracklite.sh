#!/usr/bin/env bash
set -euo pipefail

export CRACKLITE_DATA_DIR="${1:-data}"

python train.py
python evaluate.py
python predict.py
