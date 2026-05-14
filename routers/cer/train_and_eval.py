#!/usr/bin/env python3
"""Train and evaluate CER-Router over a lambda_cost sweep."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.cer.experiment_splits import choose_calibration_indices, parse_list, resolve_experiment_splits
from routers.cer.features import ensure_meta_frame, parse_query_type
from routers.cer.metrics import accuracy_cost_summary, normalize_cost_matrix
from routers.cer.router import ABLATION_MODES, CERRouter
from routers.utils.rank_score import get_cost_bounds_from_config
from routers.utils.train_utils import align_train_data, load_data_for_training


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
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
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


def make_model_mapping(models: List[str], model_indices: List[int]) -> Dict[int, str]:
    return {relative_idx: models[source_idx] for relative_idx, source_idx in enumerate(model_indices)}


def aligned_split(data: Dict[str, Any], split_name: str, sample_ids: Iterable[Any], model_indices: List[int]):
    ids = set(sample_ids)
    if not ids:
        return None
    X_text, X_vision, Y, C, meta, meta_indices, embedding_indices = align_train_data(data, ids)
    if len(meta) == 0:
        return None

    model_indices_arr = np.asarray(model_indices, dtype=int)
    split = {
        "name": split_name,
        "X_text": X_text,
        "X_vision": X_vision,
        "Y": Y[:, model_indices_arr],
        "C": C[:, model_indices_arr],
        "Y_full": Y,
        "C_full": C,
        "meta": meta.reset_index(drop=True),
        "meta_indices": meta_indices,
        "embedding_indices": embedding_indices,
        "model_indices": list(model_indices),
    }
    return split


def build_eval_profiles(
    router: CERRouter,
    train_full_split: Dict[str, Any],
    train_model_indices: List[int],
    eval_model_indices: List[int],
    heldout_model_indices: List[int],
    profile_calibration_size: int,
    seed: int,
) -> np.ndarray:
    eval_model_indices_arr = np.asarray(eval_model_indices, dtype=int)
    Y_profile = train_full_split["Y_full"][:, eval_model_indices_arr]
    C_profile = train_full_split["C_full"][:, eval_model_indices_arr]
    meta_profile = train_full_split["meta"]
    eval_profiles = router.build_raw_model_profiles(Y_profile, C_profile, meta=meta_profile)

    if heldout_model_indices:
        eval_index_to_relative = {source_idx: i for i, source_idx in enumerate(eval_model_indices)}
        heldout_relative = [eval_index_to_relative[idx] for idx in heldout_model_indices if idx in eval_index_to_relative]
        seen_relative = [eval_index_to_relative[idx] for idx in train_model_indices if idx in eval_index_to_relative]

        if profile_calibration_size > 0:
            calibration_idx = choose_calibration_indices(len(meta_profile), profile_calibration_size, seed=seed)
            if calibration_idx.size:
                calibration_meta = meta_profile.iloc[calibration_idx].reset_index(drop=True)
                calibration_profiles = router.build_raw_model_profiles(
                    Y_profile[calibration_idx],
                    C_profile[calibration_idx],
                    meta=calibration_meta,
                )
                for relative_idx in heldout_relative:
                    eval_profiles[relative_idx] = calibration_profiles[relative_idx]
        elif seen_relative:
            mean_seen_profile = np.mean(eval_profiles[seen_relative], axis=0)
            for relative_idx in heldout_relative:
                eval_profiles[relative_idx] = mean_seen_profile
    return eval_profiles.astype(np.float32)


def evaluate_cer_router(
    router: CERRouter,
    split: Dict[str, Any],
    model_names: List[str],
    cmin: float,
    cmax: float,
    beta: float,
    router_overhead_cost: float,
    ece_bins: int,
    baseline_frontier: Optional[List[Dict[str, float]]] = None,
    pred_path: Optional[Path] = None,
    risk_matrix_path: Optional[Path] = None,
) -> Dict[str, float]:
    X_text = split["X_text"]
    X_vision = split["X_vision"]
    Y = np.asarray(split["Y"])
    C = np.asarray(split["C"])
    meta = ensure_meta_frame(split["meta"], n_samples=len(split["meta"]))

    risks = router.predict_risk_matrix(X_text=X_text, X_vision=X_vision, meta=meta)
    preds = router.predict(X_text=X_text, X_vision=X_vision, meta=meta, C=C)
    metrics = accuracy_cost_summary(
        preds=preds,
        Y=Y,
        C=C,
        risks=risks,
        cmin=cmin,
        cmax=cmax,
        beta=beta,
        router_overhead_cost=router_overhead_cost,
        ece_bins=ece_bins,
        baseline_frontier=baseline_frontier,
    )

    if risk_matrix_path is not None:
        risk_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(risk_matrix_path, risks)

    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        row_ids = np.arange(len(preds))
        correct = Y[row_ids, preds].astype(int)
        selected_costs = C[row_ids, preds].astype(float)
        norm_costs = normalize_cost_matrix(C, cmin=cmin, cmax=cmax)
        selected_risks = risks[row_ids, preds]
        selected_norm_costs = norm_costs[row_ids, preds]
        lambda_cost = router._effective_lambda()
        source_model_indices = [split["model_indices"][int(pred)] for pred in preds]
        pred_model_names = [model_names[source_idx] for source_idx in source_model_indices]
        datasets = meta["dataset"].tolist() if "dataset" in meta.columns else [""] * len(preds)
        query_types = [parse_query_type(row) for _, row in meta.iterrows()]
        pd.DataFrame(
            {
                "sample_id": meta["sample_id"].tolist(),
                "dataset": datasets,
                "query_type": query_types,
                "pred_model_idx": preds,
                "source_model_idx": source_model_indices,
                "pred_model": pred_model_names,
                "correct": correct,
                "cost": selected_costs,
                "cost_with_router_overhead": selected_costs + float(router_overhead_cost),
                "failure_risk": selected_risks,
                "normalized_cost": selected_norm_costs,
                "route_score": selected_risks + lambda_cost * selected_norm_costs,
            }
        ).to_csv(pred_path, index=False)

    return metrics


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_baseline_frontier(path: Optional[str | Path]) -> Optional[List[Dict[str, float]]]:
    if not path:
        return None
    baseline_path = Path(path)
    if not baseline_path.exists():
        print(f"Warning: baseline summary not found, skipping frontier comparison: {baseline_path}")
        return None
    try:
        df = pd.read_csv(baseline_path)
    except Exception as exc:
        print(f"Warning: failed to read baseline summary {baseline_path}: {exc}")
        return None

    required = {"accuracy", "avg_cost"}
    if not required.issubset(set(df.columns)):
        print(f"Warning: baseline summary lacks columns {sorted(required)}; skipping frontier comparison")
        return None

    if "router" in df.columns:
        df = df[df["router"].astype(str).str.lower() != "oracle"].copy()
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df = df[np.isfinite(df["accuracy"]) & np.isfinite(df["avg_cost"])]
    if df.empty:
        print("Warning: baseline summary has no finite non-oracle frontier rows")
        return None

    df = df.sort_values(["avg_cost", "accuracy"], ascending=[True, False])
    frontier_rows = []
    best_accuracy = -math.inf
    for _, row in df.iterrows():
        accuracy = float(row["accuracy"])
        avg_cost = float(row["avg_cost"])
        if accuracy > best_accuracy + 1e-12:
            try:
                rank_score_value = float(row.get("rank_score", float("nan")))
            except (TypeError, ValueError):
                rank_score_value = float("nan")
            frontier_rows.append(
                {
                    "router": str(row.get("router", "baseline")),
                    "accuracy": accuracy,
                    "avg_cost": avg_cost,
                    "rank_score": rank_score_value,
                }
            )
            best_accuracy = accuracy

    print(f"Loaded {len(frontier_rows)} baseline frontier points from {baseline_path}")
    return frontier_rows or None


def build_frontier_summary(
    summary_rows: List[Dict[str, Any]],
    baseline_frontier: Optional[List[Dict[str, float]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        rows.append(
            {
                "source": "cer",
                "router": row.get("router", "cer"),
                "seed": row.get("seed"),
                "lambda_cost": row.get("lambda_cost"),
                "ablation_mode": row.get("ablation_mode"),
                "split": row.get("split"),
                "accuracy": row.get("accuracy"),
                "avg_cost": row.get("avg_cost"),
                "total_cost": row.get("total_cost"),
                "rank_score": row.get("rank_score"),
                "oracle_gap": row.get("oracle_gap"),
            }
        )
    for row in baseline_frontier or []:
        rows.append(
            {
                "source": "baseline",
                "router": row.get("router", "baseline"),
                "seed": "",
                "lambda_cost": "",
                "ablation_mode": "",
                "split": "test",
                "accuracy": row.get("accuracy"),
                "avg_cost": row.get("avg_cost"),
                "total_cost": float("nan"),
                "rank_score": row.get("rank_score"),
                "oracle_gap": float("nan"),
            }
        )

    frontier_df = pd.DataFrame(rows)
    if frontier_df.empty:
        return frontier_df

    frontier_df = frontier_df.sort_values(["avg_cost", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    best_accuracy = -math.inf
    is_frontier = []
    for _, row in frontier_df.iterrows():
        try:
            accuracy = float(row["accuracy"])
        except (TypeError, ValueError):
            accuracy = float("nan")
        keep = np.isfinite(accuracy) and accuracy > best_accuracy + 1e-12
        is_frontier.append(bool(keep))
        if keep:
            best_accuracy = accuracy
    frontier_df["is_frontier"] = is_frontier
    return frontier_df


def run_one_seed(
    args,
    data: Dict[str, Any],
    cmin: float,
    cmax: float,
    cost_bounds_source: str,
    experiment_info: Dict[str, Any],
    baseline_frontier: Optional[List[Dict[str, float]]],
    seed: int,
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_splits = experiment_info["splits"]
    train_model_indices = list(experiment_info["train_model_indices"])
    eval_model_indices = list(experiment_info["eval_model_indices"])
    heldout_model_indices = list(experiment_info["heldout_model_indices"])
    models = list(data["models"])

    train_fit_split = aligned_split(data, "train", sample_splits.get("train", []), train_model_indices)
    train_full_split = aligned_split(data, "train", sample_splits.get("train", []), list(range(len(models))))
    if train_fit_split is None or train_full_split is None:
        raise ValueError("No aligned training samples found")

    dev_fit_split = aligned_split(data, "dev", sample_splits.get("dev", []), train_model_indices)
    train_eval_split = aligned_split(data, "train", sample_splits.get("train", []), eval_model_indices)
    dev_eval_split = aligned_split(data, "dev", sample_splits.get("dev", []), eval_model_indices)
    test_eval_split = aligned_split(data, "test", sample_splits.get("test", []), eval_model_indices)

    if args.enable_dev and dev_fit_split is None:
        print("Warning: --enable_dev requested but no aligned dev split was found; dev monitoring disabled")

    eval_splits = [split for split in [train_eval_split, dev_eval_split, test_eval_split] if split is not None]
    if not eval_splits:
        eval_splits = [train_eval_split] if train_eval_split is not None else []
    primary_split = test_eval_split or dev_eval_split or train_eval_split
    if primary_split is None:
        raise ValueError("No aligned evaluation samples found")

    print(f"seed={seed}")
    print(f"  train samples: {len(train_fit_split['meta'])}, train models: {len(train_model_indices)}")
    if dev_fit_split is not None:
        print(f"  dev samples: {len(dev_fit_split['meta'])}")
    if test_eval_split is not None:
        print(f"  test samples: {len(test_eval_split['meta'])}")
    if heldout_model_indices:
        heldout_names = [models[idx] for idx in heldout_model_indices]
        print(f"  heldout models: {heldout_names}")

    train_model_mapping = make_model_mapping(models, train_model_indices)
    eval_model_mapping = make_model_mapping(models, eval_model_indices)
    model_costs = np.nanmean(train_fit_split["C"], axis=0)
    lambdas = parse_lambda_list(args.lambda_list)

    seed_summary_rows: List[Dict[str, Any]] = []
    seed_report: Dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "cost_bounds": {"cmin": cmin, "cmax": cmax, "source": cost_bounds_source},
        "split_mode": args.split_mode,
        "ablation_mode": args.ablation_mode,
        "by_lambda": {},
    }

    for lambda_cost in lambdas:
        tag = lambda_tag(lambda_cost)
        print("\n" + "=" * 80)
        print(f"Training CER lambda_cost={lambda_cost} seed={seed}")
        print("=" * 80)

        router = CERRouter(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            alpha_brier=args.alpha_brier,
            beta_rank=args.beta_rank,
            rank_margin=args.rank_margin,
            rank_pairs_per_sample=args.rank_pairs_per_sample,
            beta_route_ce=args.beta_route_ce,
            lambda_cost=lambda_cost,
            device=args.device,
            text_encoder=args.text_encoder,
            vision_encoder=args.vision_encoder,
            extra_feature_csv=args.extra_feature_csv,
            ablation_mode=args.ablation_mode,
            enable_dev=bool(args.enable_dev and dev_fit_split is not None),
            patience=args.patience,
            monitor_metric=args.monitor_metric,
            random_state=seed,
            verbose=1,
        )
        monitor_dir = run_dir / f"training_monitor_lambda_{tag}" if args.enable_dev else None
        router.fit(
            train_fit_split["Y"],
            train_fit_split["C"],
            train_fit_split["meta"],
            X_text=train_fit_split["X_text"],
            X_vision=train_fit_split["X_vision"],
            model_mapping=train_model_mapping,
            costs=model_costs,
            Y_dev=dev_fit_split["Y"] if dev_fit_split is not None else None,
            C_dev=dev_fit_split["C"] if dev_fit_split is not None else None,
            meta_dev=dev_fit_split["meta"] if dev_fit_split is not None else None,
            X_text_dev=dev_fit_split["X_text"] if dev_fit_split is not None else None,
            X_vision_dev=dev_fit_split["X_vision"] if dev_fit_split is not None else None,
            monitor_output_dir=monitor_dir,
            cmin=cmin,
            cmax=cmax,
            rank_score_beta=args.rank_score_beta,
        )

        if eval_model_indices != train_model_indices or heldout_model_indices:
            eval_profiles = build_eval_profiles(
                router=router,
                train_full_split=train_full_split,
                train_model_indices=train_model_indices,
                eval_model_indices=eval_model_indices,
                heldout_model_indices=heldout_model_indices,
                profile_calibration_size=args.profile_calibration_size,
                seed=seed,
            )
            router.set_model_profiles(eval_profiles, model_mapping=eval_model_mapping)

        checkpoint_path = run_dir / f"cer_lambda_{tag}.pt"
        router.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        lambda_results: Dict[str, Dict[str, float]] = {}
        pred_path = run_dir / f"pred_lambda_{tag}.csv"
        for split in eval_splits:
            split_pred_path = pred_path if split["name"] == primary_split["name"] else None
            risk_matrix_path = None
            if args.save_risk_matrix and split["name"] == primary_split["name"]:
                risk_matrix_path = run_dir / f"risk_lambda_{tag}_{split['name']}.npy"
            metrics = evaluate_cer_router(
                router=router,
                split=split,
                model_names=models,
                cmin=cmin,
                cmax=cmax,
                beta=args.rank_score_beta,
                router_overhead_cost=args.router_overhead_cost,
                ece_bins=args.ece_bins,
                baseline_frontier=baseline_frontier,
                pred_path=split_pred_path,
                risk_matrix_path=risk_matrix_path,
            )
            lambda_results[split["name"]] = metrics
            print(
                f"  {split['name']}: acc={metrics['accuracy']:.4f}, "
                f"cost={metrics['avg_cost']:.6f}, total_cost={metrics['total_cost']:.6f}, "
                f"rank_score={metrics['rank_score']:.4f}, auc={metrics['failure_auc']:.4f}, "
                f"brier={metrics['brier']:.6f}, ece={metrics['ece']:.6f}, "
                f"rc_auc={metrics['risk_coverage_auc']:.4f}"
            )

        primary_metrics = lambda_results[primary_split["name"]]
        summary_row = {
            "router": "cer",
            "seed": seed,
            "lambda_cost": lambda_cost,
            "split": primary_split["name"],
            "ablation_mode": args.ablation_mode,
            "split_mode": args.split_mode,
            "checkpoint": str(checkpoint_path),
            "pred_path": str(pred_path),
            **primary_metrics,
        }
        seed_summary_rows.append(summary_row)
        seed_report["by_lambda"][str(lambda_cost)] = {
            "checkpoint": str(checkpoint_path),
            "pred_path": str(pred_path),
            "results": lambda_results,
            "training_history": router.training_history,
        }

    return seed_summary_rows, seed_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate CER-Router")
    parser.add_argument("--dataset_dir", "--dataset-dir", default=".", help="Dataset root directory")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/cer_router", help="Output directory")
    parser.add_argument(
        "--lambda_list",
        "--lambda-list",
        nargs="+",
        default=["0", "0.01", "0.03", "0.1", "0.3", "1.0"],
    )
    parser.add_argument("--text_encoder", "--text-encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision_encoder", "--vision-encoder", default="facebook/dinov2-base")
    parser.add_argument("--extra_feature_csv", "--extra-feature-csv", default=None)
    parser.add_argument("--hidden_dim", "--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha_brier", "--alpha-brier", type=float, default=0.3)
    parser.add_argument("--beta_rank", "--beta-rank", type=float, default=0.2)
    parser.add_argument("--rank_margin", "--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank_pairs_per_sample", "--rank-pairs-per-sample", type=int, default=4)
    parser.add_argument("--beta_route_ce", "--beta-route-ce", type=float, default=0.1)
    parser.add_argument("--pairwise_rank_weight", "--pairwise-rank-weight", type=float, default=None)
    parser.add_argument("--pairwise_rank_margin", "--pairwise-rank-margin", type=float, default=None)
    parser.add_argument("--enable_dev", "--enable-dev", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--monitor_metric",
        "--monitor-metric",
        default="rank_score",
        choices=["rank_score", "accuracy", "brier", "ECE", "ece"],
    )
    parser.add_argument("--rank_score_beta", "--rank-score-beta", type=float, default=0.1)
    parser.add_argument("--ece_bins", "--ece-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", "--num-seeds", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ablation_mode", "--ablation-mode", default="full", choices=list(ABLATION_MODES))
    parser.add_argument(
        "--split_mode",
        "--split-mode",
        default="standard",
        choices=["standard", "heldout_task", "heldout_dataset", "heldout_model"],
    )
    parser.add_argument("--heldout_tasks", "--heldout-tasks", nargs="*", default=[])
    parser.add_argument("--heldout_datasets", "--heldout-datasets", nargs="*", default=[])
    parser.add_argument("--heldout_models", "--heldout-models", nargs="*", default=[])
    parser.add_argument("--profile_calibration_size", "--profile-calibration-size", type=int, default=0)
    parser.add_argument("--router_overhead_cost", "--router-overhead-cost", type=float, default=0.0)
    parser.add_argument("--save_risk_matrix", "--save-risk-matrix", action="store_true")
    parser.add_argument("--baseline_summary", "--baseline-summary", default=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.pairwise_rank_weight is not None:
        args.beta_rank = float(args.pairwise_rank_weight)
    if args.pairwise_rank_margin is not None:
        args.rank_margin = float(args.pairwise_rank_margin)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambda_list(args.lambda_list)
    seeds = [int(args.seed) + offset for offset in range(max(1, int(args.num_seeds)))]

    print("=" * 80)
    print("CER-Router training and evaluation")
    print("=" * 80)
    print(f"dataset_dir: {dataset_dir}")
    print(f"output_dir: {output_dir}")
    print(f"lambda_list: {lambdas}")
    print(f"seeds: {seeds}")
    print(f"text_encoder: {args.text_encoder}")
    print(f"vision_encoder: {args.vision_encoder}")
    print(f"ablation_mode: {args.ablation_mode}")
    print(f"split_mode: {args.split_mode}")
    print(f"extra_feature_csv: {args.extra_feature_csv}")
    print(
        f"beta_rank: {args.beta_rank}, rank_margin: {args.rank_margin}, "
        f"rank_pairs_per_sample: {args.rank_pairs_per_sample}, beta_route_ce: {args.beta_route_ce}"
    )

    missing = find_missing_dataset_artifacts(dataset_dir, args.text_encoder, args.vision_encoder)
    if missing:
        print("Missing required dataset artifacts:")
        for item in missing:
            print(f"  - {item}")
        print("Build matrices and embeddings first; no data will be downloaded by this script.")
        raise SystemExit(2)

    data = load_data_for_training(
        dataset_dir,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
    )
    cmin, cmax, cost_bounds_source = load_cost_bounds(dataset_dir, data["C"])
    print(f"cost_bounds: cmin={cmin:.8f}, cmax={cmax:.8f}, source={cost_bounds_source}")
    baseline_frontier = load_baseline_frontier(args.baseline_summary)

    experiment_info = resolve_experiment_splits(
        data=data,
        split_mode=args.split_mode,
        heldout_tasks=parse_list(args.heldout_tasks),
        heldout_datasets=parse_list(args.heldout_datasets),
        heldout_models=parse_list(args.heldout_models),
    )
    for note in experiment_info.get("notes", []):
        print(f"split note: {note}")

    all_summary_rows: List[Dict[str, Any]] = []
    seed_reports: Dict[str, Any] = {}
    for seed in seeds:
        run_dir = output_dir if len(seeds) == 1 else output_dir / f"seed_{seed}"
        seed_rows, seed_report = run_one_seed(
            args=args,
            data=data,
            cmin=cmin,
            cmax=cmax,
            cost_bounds_source=cost_bounds_source,
            experiment_info=experiment_info,
            baseline_frontier=baseline_frontier,
            seed=seed,
            run_dir=run_dir,
        )
        all_summary_rows.extend(seed_rows)
        seed_reports[str(seed)] = seed_report

    summary_df = pd.DataFrame(all_summary_rows)
    required_summary_columns = [
        "lambda_cost",
        "accuracy",
        "avg_cost",
        "total_cost",
        "rank_score",
        "oracle_accuracy",
        "oracle_avg_cost",
        "oracle_gap",
        "failure_auc",
        "brier",
        "ece",
        "risk_coverage_auc",
        "num_samples",
        "num_correct",
    ]
    for column in required_summary_columns:
        if column not in summary_df.columns:
            summary_df[column] = np.nan
    leading_columns = [
        "router",
        "seed",
        "split",
        "ablation_mode",
        "split_mode",
        "checkpoint",
        "pred_path",
    ]
    ordered_columns = [col for col in leading_columns + required_summary_columns if col in summary_df.columns]
    ordered_columns.extend([col for col in summary_df.columns if col not in ordered_columns])
    summary_df = summary_df.loc[:, ordered_columns]
    summary_csv = output_dir / "cer_summary.csv"
    summary_json = output_dir / "cer_summary.json"
    risk_metrics_json = output_dir / "risk_metrics.json"
    frontier_csv = output_dir / "frontier_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    build_frontier_summary(all_summary_rows, baseline_frontier=baseline_frontier).to_csv(frontier_csv, index=False)

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
            "beta_rank": args.beta_rank,
            "rank_margin": args.rank_margin,
            "rank_pairs_per_sample": args.rank_pairs_per_sample,
            "beta_route_ce": args.beta_route_ce,
            "text_encoder": args.text_encoder,
            "vision_encoder": args.vision_encoder,
            "extra_feature_csv": args.extra_feature_csv,
            "baseline_summary": args.baseline_summary,
            "ablation_mode": args.ablation_mode,
            "split_mode": args.split_mode,
            "heldout_tasks": parse_list(args.heldout_tasks),
            "heldout_datasets": parse_list(args.heldout_datasets),
            "heldout_models": parse_list(args.heldout_models),
            "profile_calibration_size": args.profile_calibration_size,
            "router_overhead_cost": args.router_overhead_cost,
            "enable_dev": args.enable_dev,
            "patience": args.patience,
            "monitor_metric": args.monitor_metric,
            "rank_score_beta": args.rank_score_beta,
            "ece_bins": args.ece_bins,
            "seed": args.seed,
            "num_seeds": args.num_seeds,
        },
        "cost_bounds": {"cmin": cmin, "cmax": cmax, "source": cost_bounds_source},
        "lambda_list": lambdas,
        "seeds": seeds,
        "baseline_frontier": baseline_frontier,
        "experiment_splits": experiment_info,
        "summary": all_summary_rows,
        "by_seed": seed_reports,
    }
    summary_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    risk_metrics_json.write_text(json.dumps(to_jsonable(seed_reports), indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CER-Router training and evaluation complete")
    print(f"Summary CSV: {summary_csv}")
    print(f"Frontier CSV: {frontier_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Risk metrics JSON: {risk_metrics_json}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    main()
