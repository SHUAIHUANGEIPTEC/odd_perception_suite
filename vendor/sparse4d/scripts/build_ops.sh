#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
# Default matches the A800 used by this experiment. Override for other GPUs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
python bev_vae/models/ops/deformable_aggregation/setup.py build_ext --inplace
python -m tools.check_sparse4d_cuda_op
