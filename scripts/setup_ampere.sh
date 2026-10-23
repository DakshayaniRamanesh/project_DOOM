#!/usr/bin/env bash
# Run on ampere AFTER copying this project and ensuring appropriate login/compute node.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# For a cluster, obtain a CUDA-compatible torch first if the provided Python install is CPU-only.
# Follow the cluster's approved PyTorch/CUDA procedure, not a guessed CUDA version.
python -m pip install -e '.[dev]'
python - <<'PY'
import torch
print('PyTorch:',torch.__version__,'CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
 print('GPU:',torch.cuda.get_device_name(0))
PY
python -m pytest -q
python -m doomrsnn.cli --config configs/base.yaml smoke
