<div align="center">

<p align="center">
  <img src="assets/icon.png" width="220" alt="VL-RouterBench logo" />
</p>

### VL-RouterBench: A Benchmark for Vision–Language Model Routing

[![arXiv](https://img.shields.io/badge/arXiv-2512.23562-red.svg)](https://arxiv.org/abs/2512.23562)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/KinghtH/VL-RouterBench)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)

</div>

This repository provides a clean, reproducible implementation of **VL-RouterBench**, a benchmark and toolkit for **routing across a pool of Vision–Language Models (VLMs)** under both **performance** and **performance–cost** objectives.

<p align="center">
  <img src="assets/pipeline.png" width="900" alt="VL-RouterBench pipeline" />
</p>

---

## 🌟 VL-RouterBench — a VLM routing benchmark

- VL-RouterBench as the **first unified benchmark tailored to multimodal VLM routing**.
- Datasets (14 total) grouped into 3 task families: **General, STEM, Charts OCR**.
- **15 open-source + 2 API models (GPT-4o and Gemini-Flash-2.5)**, spanning roughly **1B** to **78B** parameters, selected to reflect a realistic quality–cost–latency trade space.
- **30,540 samples**, **519,180** sample–model inference records, and **~34.5M** total tokens (input+output), constructed from VLM inference/scoring artifacts (VLMEvalKit logs).
- The derived **Accuracy–cost-aware soft labels** allocates probability mass only to correct models, smoothly interpolating from accuracy-only ($\lambda=0$) to “cheapest correct model” ($\lambda\rightarrow\infty$).
- Two Router architecture paradigms: 
  - **Feature-level routers**: frozen text+image encoders + lightweight classifier/fusion.
  - **End-to-end routers**: fine-tune multimodal backbones to directly predict the routed model.
- Primary metrics: **Average Accuracy**, **Average Cost**, **Rank Score**, and **Throughput (K tokens/s)**.

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

VL-RouterBench converts [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit) outputs into a unified routing benchmark.

To make data setup easier, we provide a pre-packaged archive **`vlm_router_data.tar.gz`** that contains everything needed to run the pipeline. You can download it from any of the following channels and extract it under the repo root:

- **Google Drive**: [vlm_router_data.tar.gz](https://drive.google.com/file/d/1Va18MW8nJqvatxDXQDQq0t9NAqr93hMg/view?usp=sharing)
- **Baidu Netdisk**: [vlm_router_data.tar.gz](https://pan.baidu.com/s/1D_P8YwY_E5kDA5dUB-ovng) (code: xb1s)
- **Hugging Face**: [vlm_router_data.tar.gz](https://huggingface.co/datasets/KinghtH/VL-RouterBench)

After downloading, extract it as:

```bash
tar -xzf vlm_router_data.tar.gz
```

By default, the pipeline expects the following directories (relative to repo root):

```bash
vlm_router_data/
  VLMEvalKit_evaluation/   # required (for is_correct / evaluation)
  VLMEvalKit_inference/    # required for accurate output-token counting (Step 2)
  TSV_images/              # optional (for TSV-packed image datasets)
```

Notes:
- **`VLMEvalKit_evaluation/`** is used by Step 1 & 4 (contains correctness signals).
- **`VLMEvalKit_inference/`** is used by Step 2 (extract real model outputs to count output tokens).
- **`TSV_images/`** is used by routers for training and inference to make routing decisions.

---

## 🎯 Quick Start

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

---

### 📁 Outputs (what you get)

After Steps 1–6, you will typically see:

```text
BENCHMARKS/                     # Step 1: per-sample JSONL with prompt + assets
ORACLE/score/                   # Step 1: parquet correctness table (sample_id, model_id, quality)
SPLITS/                         # Step 1: train/dev/test jsonl
reports/token_statistics/       # Step 2: token counts + token-based costs
data/matrices/                  # Step 3: Y.npz (quality), C.npy (cost), cost_bounds.json
data/registry/                  # Step 3: meta.parquet, model_index.pkl, ...
EMBEDDINGS/                     # Step 5: text/ and vision/ embeddings (parquet)
outputs/baselines_evaluation/   # Step 6: baseline summary + per-sample/per-dataset reports
```

---

## 🧠 Routers

### Baselines (no learning)

Evaluated in Step 6:
- `Oracle` (upper bound)
- `StrongestGlobal`
- `CheapestGlobal`
- `StrongestPerDataset`
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
python routers/vlc/train_and_eval.py --dataset_dir . --model_type lr --output_dir outputs/vlc
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
@misc{huang2025vlrouterbenchbenchmarkvisionlanguagemodel,
      title={VL-RouterBench: A Benchmark for Vision-Language Model Routing}, 
      author={Zhehao Huang and Baijiong Lin and Jingyuan Zhang and Jingying Wang and Yuhang Liu and Ning Lu and Tao Li and Xiaolin Huang},
      year={2025},
      eprint={2512.23562},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.23562}, 
}
```

---

## 🙏 Acknowledgements

- VLMEvalKit for providing the underlying VLM evaluation outputs.
- RouterArena for the Rank Score formulation inspiration.



