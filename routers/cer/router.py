#!/usr/bin/env python3
"""CER-Router: Evidence-Conditioned Risk Routing."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on runtime env
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    HAS_TORCH = False

try:
    from routers.common import RouterBase
except Exception:  # pragma: no cover - fallback for isolated imports
    class RouterBase:  # type: ignore[no-redef]
        pass

from routers.cer.features import (
    QUERY_TYPES,
    build_model_profiles,
    build_sample_features,
    parse_query_type,
)
from routers.cer.metrics import accuracy_cost_summary, normalize_cost_matrix


ABLATION_MODES: Tuple[str, ...] = (
    "full",
    "no_query_type",
    "no_model_profile",
    "no_calibration",
    "no_cost",
    "no_extra_features",
    "no_ranking",
    "no_route_ce",
    "sample_difficulty_only",
    "random_extra_features",
)


if HAS_TORCH:

    class PairwiseRiskNet(nn.Module):
        """Pairwise risk estimator with model-profile projection."""

        def __init__(self, sample_dim: int, model_dim: int, hidden_dim: int = 512, dropout: float = 0.15):
            super().__init__()
            self.sample_dim = int(sample_dim)
            self.model_dim = int(model_dim)
            self.hidden_dim = int(hidden_dim)
            self.dropout = float(dropout)

            self.profile_proj = nn.Sequential(
                nn.Linear(self.model_dim, self.sample_dim),
                nn.LayerNorm(self.sample_dim),
                nn.GELU(),
            )
            second_dim = max(32, self.hidden_dim // 2)
            self.mlp = nn.Sequential(
                nn.Linear(self.sample_dim * 3, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, second_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(second_dim, 1),
            )

        def forward(self, sample_features, model_profiles):
            projected_profile = self.profile_proj(model_profiles)
            interaction = sample_features * projected_profile
            pair_features = torch.cat([sample_features, projected_profile, interaction], dim=-1)
            return self.mlp(pair_features).squeeze(-1)

else:

    class PairwiseRiskNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("CER-Router requires PyTorch. Please install torch.")


class _PairIndexDataset(Dataset):
    """Dataset of flattened (sample, model) pair ids."""

    def __init__(self, n_samples: int, n_models: int):
        self.n_samples = int(n_samples)
        self.n_models = int(n_models)
        self.n_pairs = self.n_samples * self.n_models

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return int(idx)


def _fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X, dtype=np.float32).mean(axis=0).astype(np.float32)
    std = np.asarray(X, dtype=np.float32).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)


def _torch_load_compat(path: str | Path, map_location=None):
    if not HAS_TORCH:
        raise ImportError("CER-Router requires PyTorch. Please install torch.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CERRouter(RouterBase):
    """
    Evidence-Conditioned Risk Router.

    The model learns R(x, m) = P(model m fails on sample x). Inference routes
    with argmin_m R(x, m) + lambda_cost * normalized_cost(x, m), and falls back
    to risk-only routing when no cost matrix is supplied.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8192,
        epochs: int = 30,
        alpha_brier: float = 0.3,
        pairwise_rank_weight: Optional[float] = None,
        pairwise_rank_margin: Optional[float] = None,
        beta_rank: float = 0.2,
        rank_margin: float = 0.05,
        rank_pairs_per_sample: int = 4,
        beta_route_ce: float = 0.1,
        lambda_cost: float = 0.0,
        device: Optional[str] = None,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        extra_feature_csv: Optional[str | Path] = None,
        ablation_mode: str = "full",
        enable_dev: bool = False,
        patience: int = 5,
        monitor_metric: str = "rank_score",
        random_state: int = 42,
        verbose: int = 1,
    ):
        if not HAS_TORCH:
            raise ImportError("CER-Router requires PyTorch. Please install torch.")
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation_mode={ablation_mode}; expected one of {ABLATION_MODES}")

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.alpha_brier = float(alpha_brier)
        if pairwise_rank_weight is not None:
            beta_rank = float(pairwise_rank_weight)
        if pairwise_rank_margin is not None:
            rank_margin = float(pairwise_rank_margin)
        self.beta_rank = float(beta_rank)
        self.rank_margin = float(rank_margin)
        self.rank_pairs_per_sample = int(rank_pairs_per_sample)
        self.beta_route_ce = float(beta_route_ce)
        self.pairwise_rank_weight = self.beta_rank
        self.pairwise_rank_margin = self.rank_margin
        self.lambda_cost = float(lambda_cost)
        self.text_encoder = str(text_encoder)
        self.vision_encoder = str(vision_encoder)
        self.extra_feature_csv = str(extra_feature_csv) if extra_feature_csv else None
        self.ablation_mode = str(ablation_mode)
        self.enable_dev = bool(enable_dev)
        self.patience = int(patience)
        self.monitor_metric = str(monitor_metric)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = str(device)

        self.model: Optional[PairwiseRiskNet] = None
        self.model_mapping: Optional[Dict[int, str]] = None
        self.reverse_mapping: Optional[Dict[str, int]] = None
        self.costs: Optional[np.ndarray] = None
        self.sample_mean: Optional[np.ndarray] = None
        self.sample_std: Optional[np.ndarray] = None
        self.profile_mean: Optional[np.ndarray] = None
        self.profile_std: Optional[np.ndarray] = None
        self.model_profiles: Optional[np.ndarray] = None
        self.model_profiles_raw: Optional[np.ndarray] = None
        self.extra_feature_columns: List[str] = []
        self.cost_min: Optional[float] = None
        self.cost_max: Optional[float] = None
        self.sample_dim: Optional[int] = None
        self.model_dim: Optional[int] = None
        self.training_history: List[Dict[str, float]] = []

    def name(self) -> str:
        return "CER-Router"

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _feature_kwargs(self) -> Dict[str, Any]:
        return {
            "include_embeddings": self.ablation_mode != "sample_difficulty_only",
            "include_query_type": self.ablation_mode != "no_query_type",
            "include_extra_features": self.ablation_mode not in ("no_extra_features",),
            "random_extra_features": self.ablation_mode == "random_extra_features",
            "random_state": self.random_state,
        }

    def _profile_kwargs(self) -> Dict[str, Any]:
        return {
            "include_query_stats": self.ablation_mode != "no_query_type",
            "include_cost": self.ablation_mode != "no_cost",
            "mode": "constant" if self.ablation_mode == "no_model_profile" else "profile",
        }

    def _effective_alpha_brier(self) -> float:
        if self.ablation_mode == "no_calibration":
            return 0.0
        return float(self.alpha_brier)

    def _effective_lambda(self, lambda_cost: Optional[float] = None) -> float:
        if self.ablation_mode == "no_cost":
            return 0.0
        return float(self.lambda_cost if lambda_cost is None else lambda_cost)

    def _effective_beta_rank(self) -> float:
        if self.ablation_mode == "no_ranking":
            return 0.0
        return float(self.beta_rank)

    def _effective_beta_route_ce(self) -> float:
        if self.ablation_mode == "no_route_ce":
            return 0.0
        return float(self.beta_route_ce)

    def _prepare_sample_features(self, X_text, X_vision, meta, fit: bool = False):
        features, query_types, extra_columns = build_sample_features(
            X_text=X_text,
            X_vision=X_vision,
            meta=meta,
            extra_feature_csv=self.extra_feature_csv,
            extra_feature_columns=None if fit else self.extra_feature_columns,
            return_query_types=True,
            **self._feature_kwargs(),
        )
        if fit:
            self.extra_feature_columns = list(extra_columns)
        return features, query_types

    def build_raw_model_profiles(self, Y: np.ndarray, C: np.ndarray, meta=None, query_types=None) -> np.ndarray:
        return build_model_profiles(
            Y=Y,
            C=C,
            meta=meta,
            query_types=query_types,
            **self._profile_kwargs(),
        )

    def _transform_samples(self, features: np.ndarray) -> np.ndarray:
        if self.sample_mean is None or self.sample_std is None:
            raise RuntimeError("CERRouter is not fitted: missing sample standardizer")
        return _apply_standardizer(features, self.sample_mean, self.sample_std)

    def _transform_profiles(self, profiles: np.ndarray) -> np.ndarray:
        if self.profile_mean is None or self.profile_std is None:
            raise RuntimeError("CERRouter is not fitted: missing profile standardizer")
        return _apply_standardizer(profiles, self.profile_mean, self.profile_std)

    def set_model_profiles(
        self,
        model_profiles: np.ndarray,
        model_mapping: Optional[Dict[int, str]] = None,
        profiles_are_standardized: bool = False,
    ):
        """Replace model profiles, used by heldout/unseen-model experiments."""
        profiles = np.asarray(model_profiles, dtype=np.float32)
        if profiles.ndim != 2:
            raise ValueError("model_profiles must be a 2D array")
        if self.model_dim is not None and profiles.shape[1] != self.model_dim:
            raise ValueError(f"profile dim mismatch: got {profiles.shape[1]}, expected {self.model_dim}")
        if profiles_are_standardized:
            self.model_profiles = profiles.astype(np.float32)
            self.model_profiles_raw = None
        else:
            self.model_profiles_raw = profiles.astype(np.float32)
            self.model_profiles = self._transform_profiles(profiles)
        if model_mapping is not None:
            self.model_mapping = dict(model_mapping)
            self.reverse_mapping = {v: k for k, v in self.model_mapping.items()}
        return self

    def fit(
        self,
        Y: np.ndarray,
        C: np.ndarray,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        model_mapping: Optional[Dict[int, str]] = None,
        costs: Optional[np.ndarray] = None,
        model_profiles: Optional[np.ndarray] = None,
        Y_dev: Optional[np.ndarray] = None,
        C_dev: Optional[np.ndarray] = None,
        X_dev: Optional[np.ndarray] = None,
        X_text_dev: Optional[np.ndarray] = None,
        X_vision_dev: Optional[np.ndarray] = None,
        meta_dev=None,
        monitor_output_dir: Optional[Path] = None,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        rank_score_beta: float = 0.1,
        **kwargs,
    ):
        del X, X_dev, kwargs
        if X_text is None or X_vision is None:
            raise ValueError("CERRouter.fit requires X_text and X_vision")

        self._set_seed()
        Y = np.asarray(Y, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)
        if Y.ndim != 2 or C.ndim != 2 or Y.shape != C.shape:
            raise ValueError(f"Y and C must be 2D arrays with the same shape; got {Y.shape} and {C.shape}")

        n_samples, n_models = Y.shape
        self.model_mapping = dict(model_mapping or {i: f"model_{i}" for i in range(n_models)})
        self.reverse_mapping = {v: k for k, v in self.model_mapping.items()}
        self.costs = np.asarray(costs, dtype=np.float32) if costs is not None else np.nanmean(C, axis=0).astype(np.float32)

        finite_costs = C[np.isfinite(C)]
        self.cost_min = float(cmin) if cmin is not None else (float(np.min(finite_costs)) if finite_costs.size else 0.0)
        self.cost_max = float(cmax) if cmax is not None else (float(np.max(finite_costs)) if finite_costs.size else 1.0)
        if self.cost_max <= self.cost_min:
            self.cost_max = self.cost_min + 1.0

        sample_features_raw, query_types = self._prepare_sample_features(X_text, X_vision, meta, fit=True)
        if model_profiles is None:
            model_profiles_raw = self.build_raw_model_profiles(Y, C, meta=meta, query_types=query_types)
        else:
            model_profiles_raw = np.asarray(model_profiles, dtype=np.float32)
        if model_profiles_raw.shape[0] != n_models:
            raise ValueError(
                f"model_profiles rows ({model_profiles_raw.shape[0]}) must match number of models ({n_models})"
            )

        self.sample_mean, self.sample_std = _fit_standardizer(sample_features_raw)
        self.profile_mean, self.profile_std = _fit_standardizer(model_profiles_raw)
        sample_features = self._transform_samples(sample_features_raw)
        model_profiles_std = self._transform_profiles(model_profiles_raw)
        self.model_profiles_raw = model_profiles_raw.astype(np.float32)
        self.model_profiles = model_profiles_std.astype(np.float32)
        self.sample_dim = int(sample_features.shape[1])
        self.model_dim = int(model_profiles_std.shape[1])

        pair_rows = int(n_samples * n_models)
        if self.verbose:
            print(
                f"  CER train shape: N={n_samples}, K={n_models}, pair_rows={pair_rows}, "
                f"sample_dim={self.sample_dim}, model_dim={self.model_dim}, ablation={self.ablation_mode}"
            )

        self.model = PairwiseRiskNet(
            sample_dim=self.sample_dim,
            model_dim=self.model_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        sample_tensor_cpu = torch.from_numpy(sample_features.astype(np.float32))
        profile_tensor_cpu = torch.from_numpy(model_profiles_std.astype(np.float32))
        fail_tensor_cpu = torch.from_numpy((1.0 - Y).astype(np.float32))
        norm_cost_tensor_cpu = torch.from_numpy(self._normalize_cost_matrix(C).astype(np.float32))
        route_target_cpu = torch.from_numpy(self._build_route_targets(Y, C).astype(np.int64))

        dataset = _PairIndexDataset(n_samples, n_models)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        use_dev = (
            self.enable_dev
            and Y_dev is not None
            and C_dev is not None
            and X_text_dev is not None
            and X_vision_dev is not None
            and len(Y_dev) > 0
        )
        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        stale_epochs = 0
        alpha_brier = self._effective_alpha_brier()
        if monitor_output_dir is not None:
            Path(monitor_output_dir).mkdir(parents=True, exist_ok=True)
        beta_rank = self._effective_beta_rank()
        beta_route_ce = self._effective_beta_route_ce()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_brier = 0.0
            total_rank = 0.0
            total_route_ce = 0.0
            total_seen = 0

            for pair_ids in loader:
                pair_ids = pair_ids.long()
                sample_idx = torch.div(pair_ids, n_models, rounding_mode="floor")
                model_idx = pair_ids.remainder(n_models)
                labels = fail_tensor_cpu[sample_idx, model_idx].to(self.device, non_blocking=True)
                sample_batch = sample_tensor_cpu[sample_idx].to(self.device, non_blocking=True)
                profile_batch = profile_tensor_cpu[model_idx].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(sample_batch, profile_batch)
                bce_loss = bce(logits, labels)
                probs = torch.sigmoid(logits)
                brier_loss = torch.mean((probs - labels) ** 2)
                rank_loss, route_ce_loss = self._auxiliary_losses_for_batch(
                    sample_idx=sample_idx,
                    sample_tensor_cpu=sample_tensor_cpu,
                    profile_tensor_cpu=profile_tensor_cpu,
                    fail_tensor_cpu=fail_tensor_cpu,
                    norm_cost_tensor_cpu=norm_cost_tensor_cpu,
                    route_target_cpu=route_target_cpu,
                    n_models=n_models,
                )
                loss = bce_loss + alpha_brier * brier_loss + beta_rank * rank_loss + beta_route_ce * route_ce_loss
                loss.backward()
                optimizer.step()

                batch_n = int(labels.numel())
                total_seen += batch_n
                total_loss += float(loss.item()) * batch_n
                total_bce += float(bce_loss.item()) * batch_n
                total_brier += float(brier_loss.item()) * batch_n
                total_rank += float(rank_loss.item()) * batch_n
                total_route_ce += float(route_ce_loss.item()) * batch_n

            row = {
                "epoch": float(epoch),
                "loss": total_loss / max(total_seen, 1),
                "bce": total_bce / max(total_seen, 1),
                "brier": total_brier / max(total_seen, 1),
                "rank_loss": total_rank / max(total_seen, 1),
                "route_ce": total_route_ce / max(total_seen, 1),
            }

            if use_dev:
                dev_metrics = self._evaluate_dev(
                    X_text_dev=X_text_dev,
                    X_vision_dev=X_vision_dev,
                    meta_dev=meta_dev,
                    Y_dev=Y_dev,
                    C_dev=C_dev,
                    rank_score_beta=rank_score_beta,
                )
                row.update(
                    {
                        f"dev_{key}": float(value)
                        for key, value in dev_metrics.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    }
                )
                metric = self._monitor_value(dev_metrics)
            else:
                metric = -row["loss"]

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
                if monitor_output_dir is not None:
                    torch.save(best_state, Path(monitor_output_dir) / "best_risk_net_state.pt")
            else:
                stale_epochs += 1

            self.training_history.append(row)
            if self.verbose:
                msg = (
                    f"  epoch {epoch:03d}: loss={row['loss']:.6f}, "
                    f"bce={row['bce']:.6f}, brier={row['brier']:.6f}, "
                    f"rank_loss={row['rank_loss']:.6f}"
                )
                if beta_route_ce > 0:
                    msg += f", route_ce={row['route_ce']:.6f}"
                if use_dev:
                    msg += (
                        f", dev_acc={row.get('dev_accuracy', 0.0):.4f}, "
                        f"dev_cost={row.get('dev_avg_cost', 0.0):.6f}, "
                        f"dev_rs={row.get('dev_rank_score', 0.0):.4f}"
                    )
                print(msg)

            if use_dev and self.patience > 0 and stale_epochs >= self.patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _build_route_targets(self, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float32)
        C = np.asarray(C, dtype=np.float32)
        if Y.shape != C.shape or Y.ndim != 2:
            raise ValueError(f"Y and C must be 2D arrays with the same shape; got {Y.shape} and {C.shape}")

        targets = np.zeros(Y.shape[0], dtype=np.int64)
        for row_idx in range(Y.shape[0]):
            row_cost = C[row_idx]
            correct = np.where(Y[row_idx] > 0.5)[0]
            finite = np.isfinite(row_cost)
            if correct.size:
                correct = correct[finite[correct]]
            if correct.size:
                targets[row_idx] = int(correct[np.argmin(row_cost[correct])])
            elif finite.any():
                targets[row_idx] = int(np.where(finite)[0][np.argmin(row_cost[finite])])
            else:
                targets[row_idx] = 0
        return targets

    def _score_matrix_for_samples(
        self,
        unique_samples,
        sample_tensor_cpu,
        profile_tensor_cpu,
        norm_cost_tensor_cpu,
        n_models: int,
    ):
        sample_block = sample_tensor_cpu[unique_samples].to(self.device, non_blocking=True)
        profile_block = profile_tensor_cpu.to(self.device, non_blocking=True)
        sample_expanded = sample_block[:, None, :].expand(-1, n_models, -1).reshape(-1, sample_block.shape[1])
        profile_expanded = profile_block[None, :, :].expand(sample_block.shape[0], -1, -1).reshape(
            -1,
            profile_block.shape[1],
        )
        logits = self.model(sample_expanded, profile_expanded).reshape(sample_block.shape[0], n_models)
        risks = torch.sigmoid(logits)
        norm_costs = norm_cost_tensor_cpu[unique_samples].to(self.device, non_blocking=True)
        scores = risks + self._effective_lambda() * norm_costs
        return scores, logits, norm_costs

    def _auxiliary_losses_for_batch(
        self,
        sample_idx,
        sample_tensor_cpu,
        profile_tensor_cpu,
        fail_tensor_cpu,
        norm_cost_tensor_cpu,
        route_target_cpu,
        n_models: int,
        max_samples: int = 128,
    ):
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        beta_rank = self._effective_beta_rank()
        beta_route_ce = self._effective_beta_route_ce()
        if beta_rank <= 0 and beta_route_ce <= 0:
            return zero, zero

        unique_samples = torch.unique(sample_idx)[:max_samples]
        if unique_samples.numel() == 0:
            return zero, zero

        scores, _, norm_costs = self._score_matrix_for_samples(
            unique_samples=unique_samples,
            sample_tensor_cpu=sample_tensor_cpu,
            profile_tensor_cpu=profile_tensor_cpu,
            norm_cost_tensor_cpu=norm_cost_tensor_cpu,
            n_models=n_models,
        )
        labels = fail_tensor_cpu[unique_samples].to(self.device, non_blocking=True)

        rank_loss = zero
        if beta_rank > 0 and self.rank_pairs_per_sample > 0:
            rank_terms = []
            for row_idx in range(labels.shape[0]):
                correct_choices = torch.nonzero(labels[row_idx] <= 0.5, as_tuple=False).flatten()
                if correct_choices.numel() == 0:
                    continue
                for _ in range(int(self.rank_pairs_per_sample)):
                    a_pos = torch.randint(correct_choices.numel(), (1,), device=self.device)
                    model_a = int(correct_choices[a_pos].item())
                    b_mask = (labels[row_idx] > 0.5) | (norm_costs[row_idx] > norm_costs[row_idx, model_a] + 1e-8)
                    b_mask[model_a] = False
                    model_b_choices = torch.nonzero(b_mask, as_tuple=False).flatten()
                    if model_b_choices.numel() == 0:
                        continue
                    b_pos = torch.randint(model_b_choices.numel(), (1,), device=self.device)
                    model_b = int(model_b_choices[b_pos].item())
                    rank_terms.append(torch.relu(scores[row_idx, model_a] + self.rank_margin - scores[row_idx, model_b]))
            if rank_terms:
                rank_loss = torch.stack(rank_terms).mean()

        route_ce_loss = zero
        if beta_route_ce > 0:
            route_targets = route_target_cpu[unique_samples].to(self.device, non_blocking=True)
            route_ce_loss = nn.functional.cross_entropy(-scores, route_targets)

        return rank_loss, route_ce_loss

    def _monitor_value(self, dev_metrics: Dict[str, float]) -> float:
        metric = self.monitor_metric
        if metric == "loss":
            return -float(dev_metrics.get("brier", math.inf))
        if metric in ("brier", "ECE", "ece", "failure_auc"):
            value = float(dev_metrics.get(metric, math.inf))
            return -value if metric in ("brier", "ECE", "ece") else value
        return float(dev_metrics.get(metric, -math.inf))

    def _evaluate_dev(self, X_text_dev, X_vision_dev, meta_dev, Y_dev, C_dev, rank_score_beta: float = 0.1):
        risks = self.predict_risk_matrix(X_text=X_text_dev, X_vision=X_vision_dev, meta=meta_dev)
        preds = self.predict(X_text=X_text_dev, X_vision=X_vision_dev, meta=meta_dev, C=C_dev)
        return accuracy_cost_summary(
            preds=preds,
            Y=np.asarray(Y_dev),
            C=np.asarray(C_dev),
            risks=risks,
            cmin=float(self.cost_min if self.cost_min is not None else np.nanmin(C_dev)),
            cmax=float(self.cost_max if self.cost_max is not None else np.nanmax(C_dev)),
            beta=rank_score_beta,
            router_overhead_cost=0.0,
        )

    def _normalize_cost_matrix(self, C: np.ndarray) -> np.ndarray:
        cmin = float(self.cost_min if self.cost_min is not None else np.nanmin(C))
        cmax = float(self.cost_max if self.cost_max is not None else np.nanmax(C))
        return normalize_cost_matrix(C, cmin=cmin, cmax=cmax)

    def predict_risk_matrix(
        self,
        X_text: np.ndarray,
        X_vision: np.ndarray,
        meta=None,
        batch_size: Optional[int] = None,
        model_profiles: Optional[np.ndarray] = None,
        profiles_are_standardized: bool = False,
        **kwargs,
    ) -> np.ndarray:
        del kwargs
        if self.model is None or self.model_profiles is None:
            raise RuntimeError("CERRouter is not fitted")

        sample_features_raw, _ = self._prepare_sample_features(X_text, X_vision, meta, fit=False)
        sample_features = self._transform_samples(sample_features_raw)
        if model_profiles is None:
            profiles = self.model_profiles
        elif profiles_are_standardized:
            profiles = np.asarray(model_profiles, dtype=np.float32)
        else:
            profiles = self._transform_profiles(np.asarray(model_profiles, dtype=np.float32))

        n_samples = sample_features.shape[0]
        n_models = profiles.shape[0]
        if profiles.shape[1] != self.model_dim:
            raise ValueError(f"profile dim mismatch: got {profiles.shape[1]}, expected {self.model_dim}")

        batch_size = int(batch_size or max(1, min(512, self.batch_size // max(n_models, 1))))
        risks = np.zeros((n_samples, n_models), dtype=np.float32)
        profile_tensor = torch.as_tensor(profiles, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                sample_tensor = torch.as_tensor(sample_features[start:end], dtype=torch.float32, device=self.device)
                sample_expanded = sample_tensor[:, None, :].expand(-1, n_models, -1).reshape(-1, sample_tensor.shape[1])
                profile_expanded = profile_tensor[None, :, :].expand(end - start, -1, -1).reshape(-1, profile_tensor.shape[1])
                logits = self.model(sample_expanded, profile_expanded).reshape(end - start, n_models)
                risks[start:end] = torch.sigmoid(logits).detach().cpu().numpy()
        return risks

    def predict(
        self,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
        C: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if X_text is None or X_vision is None:
            raise ValueError("CERRouter.predict requires X_text and X_vision")

        lambda_cost = self._effective_lambda(kwargs.pop("lambda_cost", None))
        batch_size = kwargs.pop("batch_size", None)
        risks = self.predict_risk_matrix(X_text=X_text, X_vision=X_vision, meta=meta, batch_size=batch_size)
        scores = risks
        if C is not None:
            C_arr = np.asarray(C, dtype=np.float32)
            if C_arr.ndim == 1:
                C_arr = np.broadcast_to(C_arr.reshape(1, -1), risks.shape).astype(np.float32)
            if C_arr.shape != risks.shape:
                raise ValueError(f"C shape must match risk matrix {risks.shape}, got {C_arr.shape}")
            scores = risks + lambda_cost * self._normalize_cost_matrix(C_arr)
        return np.argmin(scores, axis=1).astype(int)

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted CERRouter")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hyperparameters": {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "alpha_brier": self.alpha_brier,
                "pairwise_rank_weight": self.pairwise_rank_weight,
                "pairwise_rank_margin": self.pairwise_rank_margin,
                "beta_rank": self.beta_rank,
                "rank_margin": self.rank_margin,
                "rank_pairs_per_sample": self.rank_pairs_per_sample,
                "beta_route_ce": self.beta_route_ce,
                "lambda_cost": self.lambda_cost,
                "device": self.device,
                "text_encoder": self.text_encoder,
                "vision_encoder": self.vision_encoder,
                "extra_feature_csv": self.extra_feature_csv,
                "ablation_mode": self.ablation_mode,
                "enable_dev": self.enable_dev,
                "patience": self.patience,
                "monitor_metric": self.monitor_metric,
                "random_state": self.random_state,
                "verbose": self.verbose,
            },
            "state_dict": self.model.state_dict(),
            "model_mapping": self.model_mapping,
            "reverse_mapping": self.reverse_mapping,
            "costs": self.costs,
            "sample_mean": self.sample_mean,
            "sample_std": self.sample_std,
            "profile_mean": self.profile_mean,
            "profile_std": self.profile_std,
            "model_profiles": self.model_profiles,
            "model_profiles_raw": self.model_profiles_raw,
            "extra_feature_columns": self.extra_feature_columns,
            "cost_min": self.cost_min,
            "cost_max": self.cost_max,
            "sample_dim": self.sample_dim,
            "model_dim": self.model_dim,
            "training_history": self.training_history,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Optional[str] = None):
        payload = _torch_load_compat(path, map_location=map_location)
        hp = dict(payload.get("hyperparameters", {}))
        if map_location is not None:
            hp["device"] = map_location
        elif str(hp.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
            hp["device"] = "cpu"

        router = cls(**hp)
        router.model_mapping = payload.get("model_mapping")
        router.reverse_mapping = payload.get("reverse_mapping")
        if router.reverse_mapping is None and router.model_mapping:
            router.reverse_mapping = {v: k for k, v in router.model_mapping.items()}
        router.costs = payload.get("costs")
        router.sample_mean = payload.get("sample_mean")
        router.sample_std = payload.get("sample_std")
        router.profile_mean = payload.get("profile_mean")
        router.profile_std = payload.get("profile_std")
        router.model_profiles = payload.get("model_profiles")
        router.model_profiles_raw = payload.get("model_profiles_raw")
        router.extra_feature_columns = list(payload.get("extra_feature_columns") or [])
        router.cost_min = payload.get("cost_min")
        router.cost_max = payload.get("cost_max")
        router.sample_dim = int(payload.get("sample_dim"))
        router.model_dim = int(payload.get("model_dim"))
        router.training_history = list(payload.get("training_history") or [])

        router.model = PairwiseRiskNet(
            sample_dim=router.sample_dim,
            model_dim=router.model_dim,
            hidden_dim=router.hidden_dim,
            dropout=router.dropout,
        ).to(router.device)
        router.model.load_state_dict(payload["state_dict"])
        router.model.eval()
        return router

    def __repr__(self):
        return (
            f"CERRouter(hidden_dim={self.hidden_dim}, lambda_cost={self.lambda_cost}, "
            f"alpha_brier={self.alpha_brier}, ablation_mode='{self.ablation_mode}')"
        )
