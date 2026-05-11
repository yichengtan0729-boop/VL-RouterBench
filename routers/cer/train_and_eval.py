#!/usr/bin/env python3
"""Train and evaluate CER-Router over a lambda_cost sweep."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.utils.rank_score import get_cost_bounds_from_config, rank_score


def parse_lambda_list(values: Iterable[str]) -> List[float]:
    lambdas: List[float] = []
    for value in values:
        for piece in str(value).replace(",", " ").split():
            if piece:
                lambdas.append(float(piece))
    if not lambdas:
        raise ValueError("--lambda_list must contain at least one value")
    return lambdas


def lambda_tag(value: float) -> str:
    return str(value).replace("+", "").replace("-", "neg").replace(".", "_")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def model_name_to_filename(model_name: str) -> str:
    return model_name.split("/")[-1] if "/" in model_name else model_name


def find_missing_dataset_artifacts(dataset_dir: Path, text_encoder: str, vision_encoder: str) -> List[str]:
    text_filename = model_name_to_filename(text_encoder)
    vision_filename = model_name_to_filename(vision_encoder)
    required = [
        dataset_dir / "data/matrices/Y.npz",
        dataset_dir / "data/registry/meta.parquet",
        dataset_dir / "data/registry/model_index.pkl",
        dataset_dir / "EMBEDDINGS/text" / f"{text_filename}.parquet",
        dataset_dir / "EMBEDDINGS/vision" / f"{vision_filename}.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if not (dataset_dir / "data/matrices/C.npy").exists() and not (dataset_dir / "data/matrices/C.npz").exists():
        missing.append(str(dataset_dir / "data/matrices/C.npy") + " or " + str(dataset_dir / "data/matrices/C.npz"))
    return missing


def split_ids_or_all(data: dict, split_name: str):
    split_ids = data["splits"].get(split_name)
    if split_ids:
        return set(split_ids)
    if split_name == "train":
        if "sample_id" in data["meta"].columns:
            return set(data["meta"]["sample_id"].values)
        return set(data["sample_ids"])
    return set()


def aligned_split(data: dict, split_name: str):
    ids = split_ids_or_all(data, split_name)
    if not ids:
        return None
    X_text, X_vision, Y, C, meta, meta_indices, embedding_indices = align_train_data(data, ids)
    if len(meta) == 0:
        return None
    return {
        "name": split_name,
        "X_text": X_text,
        "X_vision": X_vision,
        "Y": Y,
        "C": C,
        "meta": meta.reset_index(drop=True),
        "meta_indices": meta_indices,
        "embedding_indices": embedding_indices,
    }


def load_cost_bounds(dataset_dir: Path, C: np.ndarray) -> Tuple[float, float, str]:
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    try:
        cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
        return float(cmin), float(cmax), str(cost_bounds_file)
    except Exception:
        finite = C[np.isfinite(C)]
        if finite.size == 0:
            return 0.0, 1.0, "fallback_empty_cost"
        cmin = float(np.min(finite))
        cmax = float(np.max(finite))
        if cmax <= cmin:
            cmax = cmin + 1.0
        return cmin, cmax, "fallback_C_min_max"


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    finite = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[finite].astype(int)
    scores = scores[finite].astype(float)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def oracle_metrics(Y: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float, float]:
    Y = np.asarray(Y)
    C = np.asarray(C)
    preds = np.zeros(Y.shape[0], dtype=int)
    for i in range(Y.shape[0]):
        correct = np.where(Y[i] == 1)[0]
        if len(correct):
            costs = C[i, correct]
            preds[i] = int(correct[np.nanargmin(costs)])
        else:
            preds[i] = int(np.nanargmin(C[i]))
    correct_values = Y[np.arange(Y.shape[0]), preds].astype(float)
    cost_values = C[np.arange(Y.shape[0]), preds].astype(float)
    return preds, float(np.nanmean(correct_values)), float(np.nanmean(cost_values))


def evaluate_cer_router(
    router,
    split: dict,
    models: List[str],
    cmin: float,
    cmax: float,
    beta: float,
    pred_path: Optional[Path] = None,
) -> Dict[str, float]:
    from routers.cer.router import parse_query_type

    X_text = split["X_text"]
    X_vision = split["X_vision"]
    Y = np.asarray(split["Y"])
    C = np.asarray(split["C"])
    meta = split["meta"].reset_index(drop=True)

    risks = router.predict_risk_matrix(X_text=X_text, X_vision=X_vision, meta=meta)
    preds = router.predict(X_text=X_text, X_vision=X_vision, meta=meta, C=C)
    correct = Y[np.arange(len(preds)), preds].astype(float)
    selected_costs = C[np.arange(len(preds)), preds].astype(float)
    accuracy = float(np.nanmean(correct)) if len(correct) else 0.0
    avg_cost = float(np.nanmean(selected_costs)) if len(selected_costs) else 0.0
    rs = float(rank_score(accuracy, avg_cost, cmin, cmax, beta=beta)) if len(correct) else 0.0

    _, oracle_accuracy, oracle_avg_cost = oracle_metrics(Y, C)
    fail_labels = (1.0 - Y).astype(np.float32)
    brier = float(np.mean((risks - fail_labels) ** 2)) if fail_labels.size else 0.0
    auc = binary_auc(fail_labels, risks)

    if pred_path is not None:
        import pandas as pd

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        sample_ids = meta["sample_id"].tolist() if "sample_id" in meta.columns else list(range(len(preds)))
        datasets = meta["dataset"].tolist() if "dataset" in meta.columns else [""] * len(preds)
        query_types = [parse_query_type(row) for _, row in meta.iterrows()]
        pred_models = [models[p] if 0 <= p < len(models) else f"model_{p}" for p in preds]
        selected_risks = risks[np.arange(len(preds)), preds]
        norm_cost = router._normalize_cost_matrix(C)
        selected_norm_costs = norm_cost[np.arange(len(preds)), preds]
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "dataset": datasets,
                "query_type": query_types,
                "pred_model_idx": preds,
                "pred_model": pred_models,
                "correct": correct.astype(int),
                "cost": selected_costs,
                "failure_risk": selected_risks,
                "normalized_cost": selected_norm_costs,
                "route_score": selected_risks + router.lambda_cost * selected_norm_costs,
            }
        ).to_csv(pred_path, index=False)

    return {
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "rank_score": rs,
        "oracle_accuracy": float(oracle_accuracy),
        "oracle_avg_cost": float(oracle_avg_cost),
        "oracle_gap": float(oracle_accuracy - accuracy),
        "failure_auc": float(auc),
        "brier": brier,
        "num_samples": int(len(preds)),
        "num_correct": int(np.nansum(correct)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CER-Router")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--output_dir", default="outputs/cer_router", help="Output directory")
    parser.add_argument("--lambda_list", nargs="+", default=["0", "0.01", "0.03", "0.1", "0.3", "1.0"])
    parser.add_argument("--text_encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision_encoder", default="facebook/dinov2-base")
    parser.add_argument("--extra_feature_csv", default=None)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha_brier", type=float, default=0.3)
    parser.add_argument("--enable_dev", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--rank_score_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambda_list(args.lambda_list)

    print("=" * 80)
    print("CER-Router training and evaluation")
    print("=" * 80)
    print(f"dataset_dir: {dataset_dir}")
    print(f"output_dir: {output_dir}")
    print(f"lambda_list: {lambdas}")
    print(f"text_encoder: {args.text_encoder}")
    print(f"vision_encoder: {args.vision_encoder}")
    print(f"extra_feature_csv: {args.extra_feature_csv}")

    missing = find_missing_dataset_artifacts(dataset_dir, args.text_encoder, args.vision_encoder)
    if missing:
        print("Missing required dataset artifacts:")
        for item in missing:
            print(f"  - {item}")
        print("Build matrices and embeddings first; no data will be downloaded by this script.")
        raise SystemExit(2)

    from routers.cer.router import CERRouter
    from routers.utils.train_utils import align_train_data as _align_train_data
    from routers.utils.train_utils import load_data_for_training as _load_data_for_training
    import pandas as pd

    globals()["align_train_data"] = _align_train_data
    globals()["load_data_for_training"] = _load_data_for_training
    globals()["pd"] = pd

    data = load_data_for_training(
        dataset_dir,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
    )
    cmin, cmax, cost_bounds_source = load_cost_bounds(dataset_dir, data["C"])
    print(f"cost_bounds: cmin={cmin:.8f}, cmax={cmax:.8f}, source={cost_bounds_source}")

    train_split = aligned_split(data, "train")
    if train_split is None:
        raise ValueError("No aligned training samples found")
    dev_split = aligned_split(data, "dev")
    test_split = aligned_split(data, "test")
    if args.enable_dev and dev_split is None:
        print("Warning: --enable_dev requested but no aligned dev split was found; dev monitoring disabled")

    eval_splits = [s for s in [train_split, dev_split, test_split] if s is not None]
    primary_split = test_split or dev_split or train_split
    print(f"train samples: {len(train_split['meta'])}")
    if dev_split is not None:
        print(f"dev samples: {len(dev_split['meta'])}")
    if test_split is not None:
        print(f"test samples: {len(test_split['meta'])}")

    K = train_split["Y"].shape[1]
    model_mapping = {i: data["models"][i] for i in range(K)}
    model_costs = np.nanmean(train_split["C"], axis=0)

    all_results = {}
    summary_rows = []
    for lam in lambdas:
        tag = lambda_tag(lam)
        print("\n" + "=" * 80)
        print(f"Training CER lambda_cost={lam}")
        print("=" * 80)

        router = CERRouter(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            alpha_brier=args.alpha_brier,
            lambda_cost=lam,
            device=args.device,
            text_encoder=args.text_encoder,
            vision_encoder=args.vision_encoder,
            extra_feature_csv=args.extra_feature_csv,
            enable_dev=bool(args.enable_dev and dev_split is not None),
            patience=args.patience,
            random_state=args.seed,
            verbose=1,
        )
        monitor_dir = output_dir / f"training_monitor_lambda_{tag}" if args.enable_dev else None
        router.fit(
            train_split["Y"],
            train_split["C"],
            train_split["meta"],
            X_text=train_split["X_text"],
            X_vision=train_split["X_vision"],
            model_mapping=model_mapping,
            costs=model_costs,
            Y_dev=dev_split["Y"] if dev_split is not None else None,
            C_dev=dev_split["C"] if dev_split is not None else None,
            meta_dev=dev_split["meta"] if dev_split is not None else None,
            X_text_dev=dev_split["X_text"] if dev_split is not None else None,
            X_vision_dev=dev_split["X_vision"] if dev_split is not None else None,
            monitor_output_dir=monitor_dir,
            cmin=cmin,
            cmax=cmax,
            rank_score_beta=args.rank_score_beta,
        )

        checkpoint_path = output_dir / f"cer_lambda_{tag}.pt"
        router.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        lambda_results = {}
        pred_path = output_dir / f"pred_lambda_{tag}.csv"
        for split in eval_splits:
            split_pred_path = pred_path if split["name"] == primary_split["name"] else None
            metrics = evaluate_cer_router(
                router=router,
                split=split,
                models=data["models"],
                cmin=cmin,
                cmax=cmax,
                beta=args.rank_score_beta,
                pred_path=split_pred_path,
            )
            lambda_results[split["name"]] = metrics
            print(
                f"  {split['name']}: acc={metrics['accuracy']:.4f}, "
                f"cost={metrics['avg_cost']:.6f}, rank_score={metrics['rank_score']:.4f}, "
                f"auc={metrics['failure_auc']:.4f}, brier={metrics['brier']:.6f}"
            )

        primary_metrics = lambda_results[primary_split["name"]]
        summary_row = {
            "lambda_cost": lam,
            "split": primary_split["name"],
            "checkpoint": str(checkpoint_path),
            "pred_path": str(pred_path),
            **primary_metrics,
        }
        summary_rows.append(summary_row)
        all_results[str(lam)] = {
            "checkpoint": str(checkpoint_path),
            "pred_path": str(pred_path),
            "results": lambda_results,
            "training_history": router.training_history,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "cer_summary.csv"
    summary_json = output_dir / "cer_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    report = {
        "router": "cer",
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "alpha_brier": args.alpha_brier,
            "text_encoder": args.text_encoder,
            "vision_encoder": args.vision_encoder,
            "extra_feature_csv": args.extra_feature_csv,
            "enable_dev": args.enable_dev,
            "patience": args.patience,
            "rank_score_beta": args.rank_score_beta,
            "seed": args.seed,
        },
        "cost_bounds": {"cmin": cmin, "cmax": cmax, "source": cost_bounds_source},
        "lambda_list": lambdas,
        "primary_split": primary_split["name"],
        "summary": summary_rows,
        "by_lambda": all_results,
    }
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CER-Router training and evaluation complete")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    main()
