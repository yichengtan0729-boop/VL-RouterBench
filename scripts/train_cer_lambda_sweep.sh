#!/bin/bash
# Train CER Lite over the default lambda_cost sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" routers/cer/train_and_eval.py \
  --dataset_dir . \
  --output_dir outputs/cer_router \
  --text_encoder BAAI/bge-m3 \
  --vision_encoder facebook/dinov2-base \
  --lambda_list 0 0.01 0.03 0.1 0.3 1.0 \
  --hidden_dim 512 \
  --dropout 0.15 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 8192 \
  --epochs 30 \
  --alpha_brier 0.3 \
  --beta_rank 0.0 \
  --rank_margin 0.05 \
  --rank_pairs_per_sample 0 \
  --beta_route_ce 0.0 \
  --enable_dev \
  "$@"
