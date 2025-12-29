<div align="center">

<p align="center">
  <img src="assets/icon.png" width="220" alt="VL-RouterBench logo" />
</p>

## VL-RouterBench

### VL-RouterBench: A Benchmark for Vision–Language Model Routing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red.svg)](#citation)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/NPULH/OpenRouterBench)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

This repository provides a clean, reproducible implementation of **VL-RouterBench**, a benchmark and toolkit for **routing across a pool of Vision–Language Models (VLMs)** under both **performance** and **performance–cost** objectives.

<p align="center">
  <img src="assets/pipeline.png" width="900" alt="VL-RouterBench pipeline" />
</p>

---

## 🌟 Highlights

- **End-to-end pipeline (Steps 1–6)**: build a routing benchmark from VLMEvalKit outputs, compute token statistics, construct quality/cost matrices, extract embeddings, and evaluate baselines.
- **Cost-aware evaluation**: supports RouterArena-style **Rank Score** that combines accuracy and (token-based) cost.
- **Two router families (paper-aligned)**:
  - **Feature-level routers** (train on pre-extracted embeddings; no end-to-end VLM finetuning): `knn`, `prknn`, `ovr`, `kmeans`, `linear`, `mlp`
  - **End-to-end routers** (train directly from text+image): `cosinecls`, `routerdc`, `zooter`, `vlc`
- **Config-driven model/dataset pools**: see `config/models.yaml`, `config/datasets.yaml`, `config/pricing.yaml`.

---

## 🚀 Installation

### Option A: Conda (recommended)

```bash
bash setup_env.sh
conda activate vl-routerbench
```

### Option B: pip only

```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation

VL-RouterBench converts **VLMEvalKit** outputs into a unified routing benchmark.

By default, the pipeline expects the following directories (relative to repo root):

```bash
vlm_router_data/
  VLMEvalKit_evaluation/   # required (for is_correct / evaluation)
  VLMEvalKit_inference/    # required for accurate output-token counting (Step 2)
  TSV_images/              # optional (for TSV-packed image datasets)
```

Notes:
- **`VLMEvalKit_evaluation/`** is used by Step 1 & 4 (contains correctness signals such as `is_correct`).
- **`VLMEvalKit_inference/`** is used by Step 2 (extract real model outputs to count output tokens).
- **`TSV_images/`** is optional. If missing, TSV-based samples will fall back to empty images (still runnable, but image-dependent routers/features may degrade).

---

## 🎯 Quick Start (Steps 1–6)

### Run everything (recommended)

```bash
bash scripts/run_all.sh
```

### Run step-by-step

```bash
# Step 1: Build benchmark (BENCHMARKS/ ORACLE/ SPLITS/)
bash scripts/run_step1_build_benchmark.sh

# Step 2: Compute token statistics (reports/token_statistics/)
bash scripts/run_step2_calculate_tokens.sh

# Step 3: Build matrices (data/matrices/Y.npz, C.npy, cost_bounds.json; data/registry/meta.parquet)
bash scripts/run_step3_build_matrices.sh

# Step 4: Validate integrity (reports/data_integrity/)
bash scripts/run_step4_validate_data.sh

# Step 5: Extract features (EMBEDDINGS/)
bash scripts/run_step5_extract_features.sh

# Step 6: Evaluate baselines (reports/baselines_evaluation/)
bash scripts/run_step6_evaluate_baselines.sh
```

### Run a range of steps

```bash
# Start from Step 2
bash scripts/run_all.sh --start-from 2

# End at Step 3
bash scripts/run_all.sh --end-at 3
```

### Common path overrides

```bash
bash scripts/run_all.sh \
  --output-dir . \
  --vlmevalkit-eval-dir vlm_router_data/VLMEvalKit_evaluation \
  --vlmevalkit-infer-dir vlm_router_data/VLMEvalKit_inference
```

---

## 📁 Outputs (what you get)

After Steps 1–6, you will typically see:

```text
BENCHMARKS/                     # Step 1: per-sample JSONL with prompt + assets
ORACLE/score/                   # Step 1: parquet correctness table (sample_id, model_id, quality)
SPLITS/                         # Step 1: train/dev/test jsonl
reports/token_statistics/       # Step 2: token counts + token-based costs
data/matrices/                  # Step 3: Y.npz (quality), C.npy (cost), cost_bounds.json
data/registry/                  # Step 3: meta.parquet, model_index.pkl, ...
EMBEDDINGS/                     # Step 5: text/ and vision/ embeddings (parquet)
reports/baselines_evaluation/   # Step 6: baseline summary + per-sample/per-dataset reports
```

---

## 🧠 Routers

### Baselines (no learning)

Evaluated in Step 6:
- `StrongestGlobal`
- `StrongestPerDataset`
- `CheapestGlobal`
- `Oracle` (upper bound; uses ground-truth Y/C at test time)
- `RandomRouter`

### Feature-level routers (train on embeddings)

These routers use Step 5 embeddings (e.g., `bge-m3` + `dinov2-base`) and optional fusion in `routers/utils/fusion.py`.

Example: train & evaluate Linear router

```bash
python routers/linear/train_and_eval.py --dataset_dir . --output_dir outputs/linear_router
```

Hyperparameter sweeps are provided under `scripts/`:
- `scripts/train_linear_lambda_sweep.sh`
- `scripts/train_mlp_lambda_sweep.sh`
- `scripts/train_knn_prknn_ovr_kmeans.sh`

### End-to-end routers (train from text+image)

These routers train directly from `meta + BENCHMARKS` (text prompts and image assets), via the unified loader in `routers/utils/benchmarks_data.py`.

Example: train & evaluate VLC router

```bash
python routers/vlc/train_and_eval.py --dataset_dir . --model_type visualbert --output_dir outputs/vlc
```

Additional end-to-end sweeps:
- `scripts/train_cosinecls_lambda_sweep.sh`
- `scripts/train_routerdc_lambda_sweep.sh`
- `scripts/train_zooter_lambda_sweep.sh`
- `scripts/train_vlc_lambda_sweep.sh`

---

## 📏 Metrics (Rank Score)

We provide a RouterArena-compatible **Rank Score** implementation in `routers/utils/rank_score.py`, which combines:
- **Accuracy** (higher is better)
- **Log-normalized cost** using `data/matrices/cost_bounds.json` (cheaper is better)

This is the default metric used in:
- baseline evaluation (`routers/utils/eval_baselines.py`)
- optional dev monitoring / early stopping for several routers

---

## ⚙️ Configuration

- `config/datasets.yaml`: dataset pool and split ratios (train/dev/test)
- `config/models.yaml`: model pool (canonical IDs + aliases)
- `config/pricing.yaml`: token-based pricing (USD per 1M tokens) and budget points

---

## 🗂️ Project Structure

```text
vl_routerbench_v1/
  scripts/        # step runners + sweeps
  tools/          # benchmark construction + token stats + matrix building + validation
  routers/        # baselines + feature-level routers + end-to-end routers
  config/         # dataset/model/pricing configs
  tests/          # lightweight checks
  assets/         # icon + pipeline figure
```

---

## 🔧 Troubleshooting

- **Step 2 fails with tokenizer/model access issues**: ensure `transformers` can access tokenizers; some models may require a Hugging Face token.
- **GPU OOM in Step 5**: reduce batch size, e.g. `bash scripts/run_step5_extract_features.sh --batch-size 16` or use CPU `--device cpu`.
- **TSV images missing**: `TSV_images/` is optional; TSV-based samples will not have real images and image-heavy routers may degrade.

---

## 📝 Citation

If you find this benchmark useful, please cite:

```bibtex
@article{vlrouterbench,
  title   = {VL-RouterBench: A Benchmark for Vision--Language Model Routing},
  author  = {TODO},
  year    = {2025},
  note    = {Coming soon}
}
```

---

## 🙏 Acknowledgements

- VLMEvalKit for providing the underlying VLM evaluation outputs.
- RouterArena for the Rank Score formulation inspiration.



