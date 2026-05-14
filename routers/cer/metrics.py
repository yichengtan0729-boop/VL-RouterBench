#!/usr/bin/env python3
"""Metrics for CER routing experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from routers.utils.rank_score import rank_score


def normalize_cost_matrix(C: np.ndarray, cmin: float, cmax: float) -> np.ndarray:
    """Normalize costs to [0, 1] with robust NaN handling."""
    C_arr = np.asarray(C, dtype=np.float32)
    if cmax <= cmin:
        return np.zeros_like(C_arr, dtype=np.float32)
    normalized = (C_arr - float(cmin)) / (float(cmax) - float(cmin))
    return np.nan_to_num(np.clip(normalized, 0.0, 1.0), nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    finite = np.isfinite(labels) & np.isfinite(probabilities)
    if not finite.any():
        return float("nan")
    return float(np.mean((probabilities[finite] - labels[finite]) ** 2))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    finite = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[finite]
    probabilities = probabilities[finite]
    if labels.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for bin_idx in range(int(n_bins)):
        left = bins[bin_idx]
        right = bins[bin_idx + 1]
        if bin_idx == int(n_bins) - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not mask.any():
            continue
        confidence = float(np.mean(probabilities[mask]))
        empirical = float(np.mean(labels[mask]))
        ece += float(mask.mean()) * abs(confidence - empirical)
    return float(ece)


def failure_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """AUC for binary failure labels using averaged ranks for ties."""
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    finite = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[finite].astype(int)
    probabilities = probabilities[finite].astype(float)

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probabilities)
    sorted_scores = probabilities[order]
    ranks = np.empty(probabilities.size, dtype=np.float64)
    start = 0
    while start < probabilities.size:
        end = start + 1
        while end < probabilities.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = float(ranks[labels == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def risk_coverage_auc(correct: np.ndarray, selected_risks: np.ndarray) -> float:
    """Area under selective accuracy as low-risk predictions are admitted."""
    correct = np.asarray(correct, dtype=np.float32).reshape(-1)
    selected_risks = np.asarray(selected_risks, dtype=np.float32).reshape(-1)
    finite = np.isfinite(correct) & np.isfinite(selected_risks)
    correct = correct[finite]
    selected_risks = selected_risks[finite]
    if correct.size == 0:
        return float("nan")

    order = np.argsort(selected_risks, kind="mergesort")
    sorted_correct = correct[order]
    cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, sorted_correct.size + 1)
    coverage = np.arange(1, sorted_correct.size + 1, dtype=np.float32) / float(sorted_correct.size)
    coverage = np.concatenate([[0.0], coverage])
    cumulative_accuracy = np.concatenate([[float(cumulative_accuracy[0])], cumulative_accuracy])
    return float(np.trapz(cumulative_accuracy, coverage))


def oracle_routing(Y: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Route to the cheapest correct model, falling back to cheapest model."""
    Y = np.asarray(Y)
    C = np.asarray(C, dtype=np.float32)
    if Y.shape != C.shape or Y.ndim != 2:
        raise ValueError(f"Y and C must be 2D arrays with the same shape; got {Y.shape} and {C.shape}")

    preds = np.zeros(Y.shape[0], dtype=int)
    for row_idx in range(Y.shape[0]):
        row_cost = C[row_idx]
        correct = np.where(Y[row_idx] == 1)[0]
        finite = np.isfinite(row_cost)
        if correct.size:
            correct = correct[finite[correct]]
        if correct.size:
            preds[row_idx] = int(correct[np.argmin(row_cost[correct])])
        elif finite.any():
            finite_idx = np.where(finite)[0]
            preds[row_idx] = int(finite_idx[np.argmin(row_cost[finite])])
        else:
            preds[row_idx] = 0

    correct_values = Y[np.arange(Y.shape[0]), preds].astype(float)
    cost_values = C[np.arange(Y.shape[0]), preds].astype(float)
    return preds, float(np.nanmean(correct_values)), float(np.nanmean(cost_values))


def _coerce_frontier_arrays(baseline_frontier: Optional[Any]) -> Tuple[np.ndarray, np.ndarray]:
    if baseline_frontier is None:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    if hasattr(baseline_frontier, "to_dict"):
        try:
            rows = baseline_frontier.to_dict("records")
        except TypeError:
            rows = baseline_frontier.to_dict()
    else:
        rows = baseline_frontier

    if isinstance(rows, dict):
        cost_values = rows.get("avg_cost", rows.get("cost", []))
        acc_values = rows.get("accuracy", rows.get("acc", []))
        rows = [{"avg_cost": cost, "accuracy": acc} for cost, acc in zip(cost_values, acc_values)]

    costs = []
    accuracies = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        cost = row.get("avg_cost", row.get("cost", row.get("model_avg_cost")))
        acc = row.get("accuracy", row.get("acc"))
        try:
            cost_f = float(cost)
            acc_f = float(acc)
        except (TypeError, ValueError):
            continue
        if np.isfinite(cost_f) and np.isfinite(acc_f):
            costs.append(cost_f)
            accuracies.append(acc_f)

    return np.asarray(costs, dtype=np.float32), np.asarray(accuracies, dtype=np.float32)


def baseline_frontier_match_metrics(
    accuracy: float,
    avg_cost: float,
    baseline_frontier: Optional[Any] = None,
) -> Dict[str, float]:
    """Compare a router point against a baseline accuracy/cost frontier."""
    costs, accuracies = _coerce_frontier_arrays(baseline_frontier)
    if costs.size == 0:
        return {
            "matched_cost_accuracy": float("nan"),
            "matched_accuracy_cost_reduction": float("nan"),
        }

    same_cost = costs <= float(avg_cost) + 1e-12
    if same_cost.any():
        matched_cost_accuracy = float(np.max(accuracies[same_cost]))
    else:
        matched_cost_accuracy = float("nan")

    same_accuracy = accuracies >= float(accuracy) - 1e-12
    if same_accuracy.any():
        baseline_cost = float(np.min(costs[same_accuracy]))
        if baseline_cost > 0:
            cost_reduction = float((baseline_cost - float(avg_cost)) / baseline_cost)
        else:
            cost_reduction = float("nan")
    else:
        cost_reduction = float("nan")

    return {
        "matched_cost_accuracy": matched_cost_accuracy,
        "matched_accuracy_cost_reduction": cost_reduction,
    }


def accuracy_cost_summary(
    preds: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    risks: Optional[np.ndarray] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    beta: float = 0.1,
    router_overhead_cost: float = 0.0,
    ece_bins: int = 15,
    baseline_frontier: Optional[Any] = None,
) -> Dict[str, float]:
    preds = np.asarray(preds, dtype=int)
    Y = np.asarray(Y)
    C = np.asarray(C, dtype=np.float32)
    if preds.ndim != 1:
        raise ValueError("preds must be a 1D array")
    if Y.shape != C.shape or Y.ndim != 2:
        raise ValueError(f"Y and C must be 2D arrays with the same shape; got {Y.shape} and {C.shape}")
    if len(preds) != Y.shape[0]:
        raise ValueError(f"preds length {len(preds)} does not match Y rows {Y.shape[0]}")

    row_ids = np.arange(len(preds))
    correct = Y[row_ids, preds].astype(float)
    selected_model_costs = C[row_ids, preds].astype(float)
    selected_total_costs = selected_model_costs + float(router_overhead_cost)
    accuracy = float(np.nanmean(correct)) if correct.size else 0.0
    avg_cost = float(np.nanmean(selected_total_costs)) if selected_total_costs.size else 0.0
    total_cost = float(np.nansum(selected_total_costs))

    if cmin is None:
        cmin = float(np.nanmin(C))
    if cmax is None:
        cmax = float(np.nanmax(C))
    if cmax <= cmin:
        cmax = cmin + 1.0

    _, oracle_accuracy, oracle_avg_cost = oracle_routing(Y, C)
    result: Dict[str, float] = {
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "model_avg_cost": float(np.nanmean(selected_model_costs)) if selected_model_costs.size else 0.0,
        "total_cost": total_cost,
        "rank_score": float(rank_score(accuracy, avg_cost, float(cmin), float(cmax), beta=beta)),
        "oracle_accuracy": float(oracle_accuracy),
        "oracle_avg_cost": float(oracle_avg_cost),
        "oracle_gap": float(oracle_accuracy - accuracy),
        "num_samples": int(len(preds)),
        "num_correct": int(np.nansum(correct)),
        "router_overhead_cost": float(router_overhead_cost),
    }
    result.update(baseline_frontier_match_metrics(accuracy, avg_cost, baseline_frontier))

    if risks is not None:
        fail_labels = (1.0 - Y).astype(np.float32)
        risks_arr = np.asarray(risks, dtype=np.float32)
        if risks_arr.shape == Y.shape:
            selected_risks = risks_arr[row_ids, preds]
        else:
            selected_risks = np.array([], dtype=np.float32)
        ece = expected_calibration_error(fail_labels, risks_arr, n_bins=ece_bins)
        result.update(
            {
                "failure_auc": failure_auc(fail_labels, risks_arr),
                "brier": brier_score(fail_labels, risks_arr),
                "ECE": ece,
                "ece": ece,
                "risk_coverage_auc": risk_coverage_auc(correct, selected_risks),
                "risk_mean": float(np.nanmean(risks_arr)) if risks_arr.size else float("nan"),
                "risk_std": float(np.nanstd(risks_arr)) if risks_arr.size else float("nan"),
            }
        )
    else:
        result.update(
            {
                "failure_auc": float("nan"),
                "brier": float("nan"),
                "ECE": float("nan"),
                "ece": float("nan"),
                "risk_coverage_auc": float("nan"),
                "risk_mean": float("nan"),
                "risk_std": float("nan"),
            }
        )
    return result
