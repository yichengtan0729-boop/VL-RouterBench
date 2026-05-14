#!/bin/bash
# Run CER-Full ablations into outputs/ablation_<mode>.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FEATURE_CSV="outputs/features/counterfactual_evidence_features.csv"
OUTPUT_BASE="outputs"
COMMON_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --feature-csv|--extra-feature-csv|--extra_feature_csv)
      FEATURE_CSV="$2"
      shift 2
      ;;
    --output-base|--output_base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    *)
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ! -f "$FEATURE_CSV" ]; then
  echo "Missing CER-Full feature CSV: $FEATURE_CSV"
  echo "Run first:"
  echo "  bash scripts/build_cer_evidence_features.sh --output-csv $FEATURE_CSV"
  exit 2
fi

MODES=(
  no_extra_features
  no_query_type
  no_model_profile
  no_calibration
  no_ranking
  no_route_ce
  sample_difficulty_only
)

for mode in "${MODES[@]}"; do
  out_dir="$OUTPUT_BASE/ablation_${mode}"
  echo "Running CER ablation: $mode -> $out_dir"
  python routers/cer/train_and_eval.py \
    --dataset_dir . \
    --output_dir "$out_dir" \
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
    --ablation_mode "$mode" \
    --enable_dev \
    "${COMMON_ARGS[@]}"
done
