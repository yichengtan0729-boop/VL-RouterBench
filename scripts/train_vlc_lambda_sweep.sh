#!/bin/bash
# Train VLC (end-to-end) with soft labels across different lambdas, and select best by test.rank_score.
#
# Example:
#   bash scripts/train_vlc_lambda_sweep.sh --dataset-dir . --output-base outputs/vlc --vlc-model-type lxmert
#
# Notes:
# - VLC supports --model_type: visualbert | lxmert | uniter | vilbert
# - train_lambda can be numeric or "inf" (inf => hard labels)

set -euo pipefail

# Make script runnable from any working directory (assume repo root is parent of scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Benchmark behavior (latency profiling)
# - default: DO measure latency (skip=0)
# - override: export VLM_ROUTER_SKIP_LATENCY=1 to skip profiling
export VLM_ROUTER_SKIP_LATENCY="${VLM_ROUTER_SKIP_LATENCY:-0}"
export VLM_ROUTER_LATENCY_WARMUP_RUNS="${VLM_ROUTER_LATENCY_WARMUP_RUNS:-5}"
export VLM_ROUTER_LATENCY_TEST_RUNS="${VLM_ROUTER_LATENCY_TEST_RUNS:-50}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_DIR="${DATASET_DIR:-.}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/vlc}"

# ==============================
# Lambda sweep config (EDIT HERE)
# ==============================
# - Default lambdas are defined by LAMBDA_LIST below.
# - Override via CLI:   --lambdas "0,10,100,1000,10000,inf"
# - Or via env var:     LAMBDA_LIST="0,10,100,1000,10000,inf" bash ...
LAMBDA_LIST="${LAMBDA_LIST:-0,10,100,1000,10000,inf}"

MAX_EPOCHS="${MAX_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PATIENCE="${PATIENCE:-2}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

VLC_MODEL_TYPE="${VLC_MODEL_TYPE:-lxmert}"
STOP_ON_FAILURE=false
ENABLE_MONITORING=true

show_usage() {
  cat << EOF
Usage: bash scripts/train_vlc_lambda_sweep.sh [OPTIONS]

Options:
  --dataset-dir DIR           Dataset root (default: .)
  --output-base DIR           Output base directory (default: outputs/vlc)
  --lambdas CSV               Comma-separated lambdas (default: 0,10,100,1000,10000,inf)

  --vlc-model-type NAME       visualbert|lxmert|uniter|vilbert (default: lxmert)
  --max-epochs N              Max epochs (default: 5)
  --batch-size N              Batch size (default: 16)
  --learning-rate VAL         Learning rate (default: 2e-5)
  --patience N                Early stop patience (default: 2)
  --device cuda|cpu           Device (default: cuda)
  --max-train-samples N       Limit train samples (optional)

  --no-monitoring             Disable training monitoring (default: enabled)
  --stop-on-failure           Stop immediately on first failed run (default: continue)
  --help, -h                  Show help
EOF
}

need_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "❌ Missing required file: $path"
    exit 1
  fi
}

run_one() {
  local lambda="$1"
  local tag="lambda$(echo "$lambda" | tr '.' '_' | tr -d '+')"
  local out_dir="$OUTPUT_BASE/$tag"

  echo ""
  echo "================================================================================"
  echo "🚀 VLC lambda sweep: model_type=$VLC_MODEL_TYPE  train_lambda=$lambda"
  echo "out_dir: $out_dir"
  echo "================================================================================"

  mkdir -p "$out_dir"

  local extra_args=()
  if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
    extra_args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
  fi

  if [ "$ENABLE_MONITORING" = true ]; then
    # VLC's train_and_eval.py has enable_monitoring default=True, and exposes --patience.
    extra_args+=(--enable_monitoring --patience "$PATIENCE")
  fi

  set +e
  python routers/vlc/train_and_eval.py \
    --model_type "$VLC_MODEL_TYPE" \
    --dataset_dir "$DATASET_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --use_soft_labels \
    --train_lambda "$lambda" \
    --device "$DEVICE" \
    "${extra_args[@]}" \
    --output_dir "$out_dir"
  local code=$?
  set -e

  if [ $code -ne 0 ]; then
    echo "❌ FAILED: train_lambda=$lambda (exit=$code)"
    if [ "$STOP_ON_FAILURE" = true ]; then
      exit $code
    fi
  else
    echo "✅ DONE: train_lambda=$lambda"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --output-base) OUTPUT_BASE="$2"; shift 2;;
    --lambdas) LAMBDA_LIST="$2"; shift 2;;
    --vlc-model-type) VLC_MODEL_TYPE="$2"; shift 2;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --learning-rate) LEARNING_RATE="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2;;
    --no-monitoring) ENABLE_MONITORING=false; shift;;
    --stop-on-failure) STOP_ON_FAILURE=true; shift;;
    --help|-h) show_usage; exit 0;;
    *) echo "Unknown option: $1"; echo "Use --help"; exit 1;;
  esac
done

export OUTPUT_BASE

echo "================================================================================"
echo "VLC lambda sweep"
echo "================================================================================"
echo "dataset_dir: $DATASET_DIR"
echo "output_base: $OUTPUT_BASE"
echo "lambdas: $LAMBDA_LIST"
echo "model_type: $VLC_MODEL_TYPE"
echo "soft_labels: true"
echo "max_epochs: $MAX_EPOCHS"
echo "batch_size: $BATCH_SIZE"
echo "learning_rate: $LEARNING_RATE"
echo "patience: $PATIENCE (monitoring=$ENABLE_MONITORING)"
echo "device: $DEVICE"
echo "max_train_samples: ${MAX_TRAIN_SAMPLES:-<full>}"
echo "================================================================================"

need_file "$DATASET_DIR/data/matrices/Y.npz"
if [ ! -f "$DATASET_DIR/data/matrices/C.npy" ] && [ ! -f "$DATASET_DIR/data/matrices/C.npz" ]; then
  echo "❌ Missing cost matrix: $DATASET_DIR/data/matrices/C.npy (or C.npz)"
  exit 1
fi
need_file "$DATASET_DIR/data/registry/meta.parquet"

mkdir -p "$OUTPUT_BASE"

IFS=',' read -r -a LAMBDAS <<< "$LAMBDA_LIST"
for lam in "${LAMBDAS[@]}"; do
  run_one "$lam"
done

echo ""
echo "================================================================================"
echo "Selecting best config by test.rank_score"
echo "================================================================================"

python - << 'PY'
import json
import os
from pathlib import Path
import csv

output_base = Path(os.environ.get("OUTPUT_BASE", "outputs/vlc"))
rows = []

for sub in sorted([p for p in output_base.iterdir() if p.is_dir()]):
    # Prefer top-level report to avoid nested duplicates.
    reports = sorted(sub.glob("*_report.json"))
    if not reports:
        reports = sorted(sub.rglob("*_report.json"))
    if not reports:
        continue
    report_path = max(reports, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        # VLC report schema: top-level {"dev": {...}, "test": {...}}
        # Other routers: {"results": {"test": {...}}}
        test = data.get("test") or (data.get("results") or {}).get("test") or {}
        rs = test.get("rank_score", None)
        acc = test.get("accuracy", None)
        cost = test.get("avg_cost", None)
        model = (
            data.get("model")
            or data.get("model_type")
            or report_path.stem.replace("_report", "")
        )
        rs_val = float(rs) if rs is not None else float("-inf")
        rows.append((rs_val, sub.name, model, acc, cost, str(report_path)))
    except Exception:
        continue

rows.sort(reverse=True, key=lambda x: x[0])
if not rows:
    print("❌ No reports found to rank. Check output directories.")
    raise SystemExit(2)

best = rows[0]
print(f"✅ Best: {best[1]}  rank_score={best[0]:.6f}")
print(f"   model: {best[2]}")
print(f"   report: {best[5]}")

out_csv = output_base / "sweep_summary.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rank_score", "lambda_tag", "model", "accuracy", "avg_cost", "report_path"])
    for r in rows:
        w.writerow([f"{r[0]:.6f}", r[1], r[2], r[3], r[4], r[5]])
print(f"✓ Wrote: {out_csv}")
PY

echo ""
echo "Done. Outputs under: $OUTPUT_BASE"


