#!/bin/bash
# Build query-conditioned counterfactual evidence features for CER-Full.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python tools/build_counterfactual_evidence_features.py \
  --dataset-dir . \
  --output-csv outputs/features/counterfactual_evidence_features.csv \
  "$@"
