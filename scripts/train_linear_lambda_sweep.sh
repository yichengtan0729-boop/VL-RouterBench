#!/bin/bash
# Train Linear (feature-level router) with different train_lambda values, and select best by test.rank_score.
#
# Example:
#   bash scripts/train_linear_lambda_sweep.sh --dataset-dir . --output-base outputs/linear
#
# Notes:
# - For linear/mlp, we pass --use_soft_labels plus --train_lambda; when train_lambda=inf the python code will switch to hard-label mode.

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

DATASET_DIR="${DATASET_DIR:-.}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/linear}"

TEXT_ENCODER="${TEXT_ENCODER:-BAAI/bge-m3}"
VISION_ENCODER="${VISION_ENCODER:-facebook/dinov2-base}"
FUSION_METHOD="${FUSION_METHOD:-normalize_concat}"

# ==============================
# Lambda sweep config (EDIT HERE)
# ==============================
# - Override via CLI:   --lambdas "0,10,100,1000,10000,inf"
# - Or via env var:     LAMBDA_LIST="0,10,100,1000,10000,inf" bash ...
LAMBDA_LIST="${LAMBDA_LIST:-0,10,100,1000,10000,inf}"

PATIENCE="${PATIENCE:-3}"
ENABLE_MONITORING=true
STOP_ON_FAILURE=false

show_usage() {
  cat << EOF
Usage: bash scripts/train_linear_lambda_sweep.sh [OPTIONS]

Options:
  --dataset-dir DIR           Dataset root (default: .)
  --output-base DIR           Output base directory (default: outputs/linear)
  --lambdas CSV               Comma-separated lambdas (default: 0,10,100,1000,10000,inf)

  --text-encoder NAME         Text encoder (default: BAAI/bge-m3)
  --vision-encoder NAME       Vision encoder (default: facebook/dinov2-base)
  --fusion-method NAME        Fusion method (default: normalize_concat)

  --patience N                Early stop patience (default: 3)
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

encoder_to_filename() {
  local name="$1"
  if [[ "$name" == */* ]]; then
    echo "${name##*/}"
  else
    echo "$name"
  fi
}

check_embeddings() {
  local text_fn vision_fn
  text_fn="$(encoder_to_filename "$TEXT_ENCODER")"
  vision_fn="$(encoder_to_filename "$VISION_ENCODER")"
  need_file "$DATASET_DIR/EMBEDDINGS/text/${text_fn}.parquet"
  need_file "$DATASET_DIR/EMBEDDINGS/vision/${vision_fn}.parquet"
}

run_one() {
  local lambda="$1"
  local tag="lambda$(echo "$lambda" | tr '.' '_' | tr -d '+')"
  local out_dir="$OUTPUT_BASE/$tag"

  echo ""
  echo "================================================================================"
  echo "🚀 Linear lambda sweep: train_lambda=$lambda"
  echo "out_dir: $out_dir"
  echo "================================================================================"

  mkdir -p "$out_dir"

  local extra_args=()
  if [ "$ENABLE_MONITORING" = true ]; then
    extra_args+=(--enable_monitoring --patience "$PATIENCE")
  fi

  set +e
  python routers/linear/train_and_eval.py \
    --dataset_dir "$DATASET_DIR" \
    --fusion_method "$FUSION_METHOD" \
    --use_soft_labels \
    --train_lambda "$lambda" \
    --text_encoder "$TEXT_ENCODER" \
    --vision_encoder "$VISION_ENCODER" \
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
    --text-encoder) TEXT_ENCODER="$2"; shift 2;;
    --vision-encoder) VISION_ENCODER="$2"; shift 2;;
    --fusion-method) FUSION_METHOD="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --no-monitoring) ENABLE_MONITORING=false; shift;;
    --stop-on-failure) STOP_ON_FAILURE=true; shift;;
    --help|-h) show_usage; exit 0;;
    *) echo "Unknown option: $1"; echo "Use --help"; exit 1;;
  esac
done

export OUTPUT_BASE

echo "================================================================================"
echo "Linear lambda sweep"
echo "================================================================================"
echo "dataset_dir: $DATASET_DIR"
echo "output_base: $OUTPUT_BASE"
echo "lambdas: $LAMBDA_LIST"
echo "text_encoder: $TEXT_ENCODER"
echo "vision_encoder: $VISION_ENCODER"
echo "fusion_method: $FUSION_METHOD"
echo "soft_labels: true (auto-off when lambda=inf)"
echo "patience: $PATIENCE (monitoring=$ENABLE_MONITORING)"
echo "================================================================================"

need_file "$DATASET_DIR/data/matrices/Y.npz"
if [ ! -f "$DATASET_DIR/data/matrices/C.npy" ] && [ ! -f "$DATASET_DIR/data/matrices/C.npz" ]; then
  echo "❌ Missing cost matrix: $DATASET_DIR/data/matrices/C.npy (or C.npz)"
  exit 1
fi
need_file "$DATASET_DIR/data/registry/meta.parquet"
check_embeddings

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
import csv
import json
import os
from pathlib import Path

output_base = Path(os.environ.get("OUTPUT_BASE", "outputs/linear"))
rows = []

for sub in sorted([p for p in output_base.iterdir() if p.is_dir()]):
    reports = sorted(sub.glob("*_report.json"))
    if not reports:
        reports = sorted(sub.rglob("*_report.json"))
    if not reports:
        continue
    report_path = max(reports, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        test = (data.get("results") or {}).get("test") or data.get("test") or {}
        rs = test.get("rank_score", None)
        acc = test.get("accuracy", None)
        cost = test.get("avg_cost", None)
        model = data.get("model", report_path.stem.replace("_report", ""))
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


