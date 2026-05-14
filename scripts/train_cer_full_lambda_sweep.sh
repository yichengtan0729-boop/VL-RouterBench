#!/bin/bash
# Train CER-Full: counterfactual evidence features + cost-aware ranking + route CE.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FEATURE_CSV="outputs/features/counterfactual_evidence_features.csv"
OUTPUT_DIR="outputs/cer_router_full"

ARGS=("$@")
idx=0
while [ "$idx" -lt "${#ARGS[@]}" ]; do
  case "${ARGS[$idx]}" in
    --extra-feature-csv|--extra_feature_csv)
      idx=$((idx + 1))
      FEATURE_CSV="${ARGS[$idx]:-}"
      ;;
    --output-dir|--output_dir)
      idx=$((idx + 1))
      OUTPUT_DIR="${ARGS[$idx]:-}"
      ;;
  esac
  idx=$((idx + 1))
done

if [ ! -f "$FEATURE_CSV" ]; then
  echo "Missing CER-Full feature CSV: $FEATURE_CSV"
  echo "Run first:"
  echo "  bash scripts/build_cer_evidence_features.sh --output-csv $FEATURE_CSV"
  exit 2
fi

python routers/cer/train_and_eval.py \
  --dataset_dir . \
  --output_dir "$OUTPUT_DIR" \
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
  --extra_feature_csv "$FEATURE_CSV" \
  --beta_rank 0.2 \
  --rank_margin 0.05 \
  --rank_pairs_per_sample 4 \
  --beta_route_ce 0.1 \
  --enable_dev \
  "$@"
