#!/bin/bash
# Train CER-Router over lambda_cost values.
#
# Example:
#   bash scripts/train_cer_lambda_sweep.sh --dataset-dir . --output-dir outputs/cer

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cer_router}"
TEXT_ENCODER="${TEXT_ENCODER:-BAAI/bge-m3}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
LAMBDA_LIST="${LAMBDA_LIST:-0 0.01 0.03 0.1 0.3 1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.15}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
EPOCHS="${EPOCHS:-30}"
ALPHA_BRIER="${ALPHA_BRIER:-0.3}"
ENABLE_DEV=true
EXTRA_FEATURE_CSV="${EXTRA_FEATURE_CSV:-}"

show_usage() {
  cat << EOF
Usage: bash scripts/train_cer_lambda_sweep.sh [OPTIONS]

Options:
  --dataset-dir DIR          Dataset root (default: .)
  --output-dir DIR           Output directory (default: outputs/cer_router)
  --lambda-list "VALUES"     Space/comma separated lambdas (default: 0 0.01 0.03 0.1 0.3 1.0)
  --text-encoder NAME        Text encoder (default: BAAI/bge-m3)
  --vision-encoder NAME      Vision encoder (default: facebook/dinov2-base)
  --extra-feature-csv PATH   Optional counterfactual evidence feature CSV
  --hidden-dim N             Hidden dim (default: 512)
  --dropout X                Dropout (default: 0.15)
  --lr X                     Learning rate (default: 1e-3)
  --weight-decay X           Weight decay (default: 1e-4)
  --batch-size N             Pair batch size (default: 8192)
  --epochs N                 Epochs (default: 30)
  --alpha-brier X            Brier loss weight (default: 0.3)
  --no-dev                   Disable dev monitoring
  --help, -h                 Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --lambda-list|--lambdas) LAMBDA_LIST="$2"; shift 2;;
    --text-encoder) TEXT_ENCODER="$2"; shift 2;;
    --vision-encoder) VISION_ENCODER="$2"; shift 2;;
    --extra-feature-csv) EXTRA_FEATURE_CSV="$2"; shift 2;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2;;
    --dropout) DROPOUT="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --alpha-brier) ALPHA_BRIER="$2"; shift 2;;
    --no-dev) ENABLE_DEV=false; shift;;
    --help|-h) show_usage; exit 0;;
    *) echo "Unknown option: $1"; echo "Use --help"; exit 1;;
  esac
done

mkdir -p "$OUTPUT_DIR"

read -r -a LAMBDAS <<< "${LAMBDA_LIST//,/ }"

EXTRA_ARGS=()
if [ "$ENABLE_DEV" = true ]; then
  EXTRA_ARGS+=(--enable_dev)
fi
if [ -n "$EXTRA_FEATURE_CSV" ]; then
  EXTRA_ARGS+=(--extra_feature_csv "$EXTRA_FEATURE_CSV")
fi

python routers/cer/train_and_eval.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --text_encoder "$TEXT_ENCODER" \
  --vision_encoder "$VISION_ENCODER" \
  --lambda_list "${LAMBDAS[@]}" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --alpha_brier "$ALPHA_BRIER" \
  "${EXTRA_ARGS[@]}"
