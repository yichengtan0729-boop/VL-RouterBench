#!/bin/bash
# Reserved CER-Full generalization suite for heldout task/model experiments.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
FEATURE_CSV="outputs/features/counterfactual_evidence_features.csv"
OUTPUT_BASE="outputs/cer_generalization"
HELDOUT_TASKS="${HELDOUT_TASKS:-}"
HELDOUT_MODELS="${HELDOUT_MODELS:-}"
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
    --heldout-tasks|--heldout_tasks)
      HELDOUT_TASKS="$2"
      shift 2
      ;;
    --heldout-models|--heldout_models)
      HELDOUT_MODELS="$2"
      shift 2
      ;;
    *)
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "Available query_type values in this dataset:"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

try:
    import pandas as pd
    from routers.cer.features import parse_query_type
except Exception as exc:
    print(f"  unavailable: {exc}")
    raise SystemExit(0)

path = Path("data/registry/meta.parquet")
if not path.exists():
    print("  data/registry/meta.parquet not found")
else:
    meta = pd.read_parquet(path)
    counts = {}
    for _, row in meta.iterrows():
        query_type = parse_query_type(row)
        counts[query_type] = counts.get(query_type, 0) + 1
    for query_type, count in sorted(counts.items()):
        print(f"  {query_type}: {count}")
PY

echo "Available model list:"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pickle

path = Path("data/registry/model_index.pkl")
if not path.exists():
    print("  data/registry/model_index.pkl not found")
else:
    with path.open("rb") as handle:
        models = pickle.load(handle)
    for idx, model in enumerate(models):
        print(f"  {idx}: {model}")
PY

if [ ! -f "$FEATURE_CSV" ]; then
  echo "Missing CER-Full feature CSV: $FEATURE_CSV"
  echo "Run first:"
  echo "  bash scripts/build_cer_evidence_features.sh --output-csv $FEATURE_CSV"
  exit 2
fi

run_full() {
  local split_mode="$1"
  local output_dir="$2"
  shift 2
  "$PYTHON_BIN" routers/cer/train_and_eval.py \
    --dataset_dir . \
    --output_dir "$output_dir" \
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
    --split_mode "$split_mode" \
    --enable_dev \
    "${COMMON_ARGS[@]}" \
    "$@"
}

ran_any=0
if [ -n "$HELDOUT_TASKS" ]; then
  echo "Running heldout_task suite for: $HELDOUT_TASKS"
  run_full "heldout_task" "$OUTPUT_BASE/heldout_task" --heldout_tasks "$HELDOUT_TASKS"
  ran_any=1
fi

if [ -n "$HELDOUT_MODELS" ]; then
  echo "Running heldout_model suite for: $HELDOUT_MODELS"
  run_full "heldout_model" "$OUTPUT_BASE/heldout_model" --heldout_models "$HELDOUT_MODELS"
  ran_any=1
fi

if [ "$ran_any" -eq 0 ]; then
  echo "No heldout task/model was specified, so no training run was launched."
  echo "Examples:"
  echo "  bash scripts/run_cer_generalization_suite.sh --heldout-tasks ocr"
  echo "  bash scripts/run_cer_generalization_suite.sh --heldout-models 0,gpt-4o-mini"
fi
