#!/usr/bin/env python3
"""
CER-Router: sample-model-specific failure risk estimation.

The router learns R(x, m) = P(model m fails on sample x). At inference it
selects argmin_m R(x, m) + lambda_cost * normalized_cost(x, m) when a cost
matrix is available, and falls back to risk-only routing otherwise.
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
    _PANDAS_IMPORT_ERROR = None
except Exception as exc:
    pd = None
    HAS_PANDAS = False
    _PANDAS_IMPORT_ERROR = exc

try:
    from routers.common import RouterBase
except Exception:
    class RouterBase:  # type: ignore[no-redef]
        pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    HAS_TORCH = False


QUERY_TYPES: Tuple[str, ...] = (
    "ocr",
    "counting",
    "spatial",
    "chart",
    "fine_grained",
    "stem",
    "general",
)

QUERY_TYPE_TO_ID = {name: idx for idx, name in enumerate(QUERY_TYPES)}


def _require_pandas():
    if not HAS_PANDAS:
        raise ImportError(f"CER metadata handling requires pandas: {_PANDAS_IMPORT_ERROR}")


def _coerce_meta(meta, n_samples: Optional[int] = None):
    n = int(n_samples or 0)
    if HAS_PANDAS:
        if meta is None:
            meta_df = pd.DataFrame(index=np.arange(n))
        elif isinstance(meta, pd.DataFrame):
            meta_df = meta.reset_index(drop=True).copy()
        else:
            try:
                meta_df = pd.DataFrame(meta).reset_index(drop=True)
            except Exception:
                meta_df = pd.DataFrame(index=np.arange(n))

        if n and len(meta_df) != n:
            meta_df = meta_df.iloc[:n].copy() if len(meta_df) > n else meta_df.reindex(range(n))
        rows = [row.to_dict() for _, row in meta_df.iterrows()]
        return rows, meta_df

    if meta is None:
        return [{} for _ in range(n)], None
    if isinstance(meta, list):
        rows = []
        for item in meta[:n]:
            try:
                rows.append(dict(item))
            except Exception:
                rows.append({})
    elif isinstance(meta, dict):
        rows = []
        for i in range(n):
            row = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > i:
                    row[key] = value[i]
                else:
                    row[key] = value
            rows.append(row)
    else:
        rows = [{} for _ in range(n)]

    if len(rows) < n:
        rows.extend({} for _ in range(n - len(rows)))
    return rows[:n], None


def _safe_text(value) -> str:
    if value is None:
        return ""
    if HAS_PANDAS:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
    return str(value)


def parse_query_type(meta_row) -> str:
    """
    Infer a coarse query type from metadata.

    The parser is intentionally heuristic and robust: it checks common text
    fields plus the dataset name, and returns "general" when there is not
    enough signal.
    """
    if meta_row is None:
        return "general"
    if HAS_PANDAS and isinstance(meta_row, pd.Series):
        row = meta_row.to_dict()
    elif isinstance(meta_row, dict):
        row = meta_row
    else:
        try:
            row = dict(meta_row)
        except Exception:
            row = {}

    text_fields = ("question", "prompt", "query", "text", "instruction", "problem", "input")
    pieces = [_safe_text(row.get(field, "")) for field in text_fields]
    dataset = _safe_text(row.get("dataset", ""))
    haystack = " ".join(pieces + [dataset]).lower()
    dataset_l = dataset.lower()

    chart_datasets = ("chart", "plotqa", "dvqa", "figureqa", "tabmwp")
    ocr_datasets = ("ocr", "textvqa", "docvqa", "st-vqa", "stvqa", "infographic", "receipt")
    count_datasets = ("count", "tallyqa", "howmany")
    spatial_datasets = ("spatial", "relation", "vcr", "gqa")
    fine_datasets = ("fgvc", "fine", "birds", "cub", "cars", "aircraft", "food", "logo")
    stem_datasets = ("science", "math", "ai2d", "mathvista", "geometry", "chemistry", "physics")

    if any(key in dataset_l for key in chart_datasets):
        return "chart"
    if any(key in dataset_l for key in ocr_datasets):
        return "ocr"
    if any(key in dataset_l for key in count_datasets):
        return "counting"
    if any(key in dataset_l for key in spatial_datasets):
        return "spatial"
    if any(key in dataset_l for key in fine_datasets):
        return "fine_grained"
    if any(key in dataset_l for key in stem_datasets):
        return "stem"

    chart_terms = (
        "chart",
        "graph",
        "plot",
        "bar chart",
        "line chart",
        "pie chart",
        "axis",
        "legend",
        "trend",
        "table",
        "x-axis",
        "y-axis",
    )
    ocr_terms = (
        "ocr",
        "read the text",
        "text in the image",
        "what does the sign say",
        "document",
        "receipt",
        "license plate",
        "handwritten",
        "transcribe",
    )
    counting_terms = (
        "how many",
        "count",
        "number of",
        "total number",
        "many objects",
        "how much",
        "quantity",
    )
    spatial_terms = (
        "left",
        "right",
        "above",
        "below",
        "under",
        "over",
        "behind",
        "in front",
        "between",
        "next to",
        "beside",
        "relative position",
        "where is",
        "closer",
        "farther",
    )
    fine_terms = (
        "fine-grained",
        "fine grained",
        "species",
        "breed",
        "brand",
        "logo",
        "model of",
        "specific type",
        "exact type",
        "identify the",
        "subtle",
    )
    stem_terms = (
        "solve",
        "calculate",
        "equation",
        "formula",
        "math",
        "geometry",
        "physics",
        "chemistry",
        "biology",
        "science",
        "which of the following",
    )

    if any(term in haystack for term in chart_terms):
        return "chart"
    if any(term in haystack for term in ocr_terms):
        return "ocr"
    if any(term in haystack for term in counting_terms):
        return "counting"
    if any(term in haystack for term in spatial_terms):
        return "spatial"
    if any(term in haystack for term in fine_terms):
        return "fine_grained"
    if any(term in haystack for term in stem_terms):
        return "stem"
    return "general"


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
    return X / np.maximum(norms, eps), norms.squeeze(1)


def _load_extra_features(
    extra_feature_csv: Optional[str | Path],
    meta_df,
    n_samples: int,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if not extra_feature_csv:
        return np.zeros((n_samples, 0), dtype=np.float32), []

    _require_pandas()
    if meta_df is None:
        warnings.warn("extra features require pandas metadata with sample_id; filled with zeros")
        cols = list(feature_columns or [])
        return np.zeros((n_samples, len(cols)), dtype=np.float32), cols

    path = Path(extra_feature_csv)
    if not path.exists():
        warnings.warn(f"extra_feature_csv not found, ignoring: {path}")
        cols = list(feature_columns or [])
        return np.zeros((n_samples, len(cols)), dtype=np.float32), cols

    extra_df = pd.read_csv(path)
    if "sample_id" not in extra_df.columns:
        warnings.warn("extra_feature_csv has no sample_id column; extra features are filled with zeros")
        cols = list(feature_columns or [])
        if feature_columns is None:
            numeric_cols = extra_df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in numeric_cols if c != "sample_id"]
        return np.zeros((n_samples, len(cols)), dtype=np.float32), cols

    if feature_columns is None:
        numeric_cols = extra_df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if c != "sample_id"]
    else:
        cols = list(feature_columns)
        for col in cols:
            if col not in extra_df.columns:
                extra_df[col] = 0.0

    if not cols:
        return np.zeros((n_samples, 0), dtype=np.float32), []

    if "sample_id" not in meta_df.columns:
        warnings.warn("meta has no sample_id column; extra features are filled with zeros")
        return np.zeros((n_samples, len(cols)), dtype=np.float32), cols

    aligned = (
        meta_df[["sample_id"]]
        .merge(extra_df[["sample_id"] + cols], on="sample_id", how="left")
        .loc[:, cols]
        .fillna(0.0)
    )
    return aligned.to_numpy(dtype=np.float32), cols


def build_sample_features(
    X_text: np.ndarray,
    X_vision: np.ndarray,
    meta=None,
    extra_feature_csv: Optional[str | Path] = None,
    extra_feature_columns: Optional[Sequence[str]] = None,
    return_query_types: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[str], List[str]]:
    """
    Build CER sample features from embeddings, metadata, and optional evidence CSV.

    Features:
      - L2-normalized text embedding
      - L2-normalized vision embedding
      - image-text cosine similarity over the shared prefix dimension
      - original text/vision embedding norms and their gap
      - query_type one-hot
      - optional numeric extra features aligned by sample_id
    """
    X_text = np.asarray(X_text, dtype=np.float32)
    X_vision = np.asarray(X_vision, dtype=np.float32)
    if X_text.ndim != 2 or X_vision.ndim != 2:
        raise ValueError("X_text and X_vision must be 2D arrays")
    if X_text.shape[0] != X_vision.shape[0]:
        raise ValueError(f"X_text and X_vision row counts differ: {X_text.shape[0]} vs {X_vision.shape[0]}")

    n_samples = X_text.shape[0]
    meta_rows, meta_df = _coerce_meta(meta, n_samples=n_samples)

    X_text_norm, text_norm = _l2_normalize(X_text)
    X_vision_norm, vision_norm = _l2_normalize(X_vision)

    shared_dim = min(X_text_norm.shape[1], X_vision_norm.shape[1])
    if shared_dim > 0:
        cosine = np.sum(X_text_norm[:, :shared_dim] * X_vision_norm[:, :shared_dim], axis=1)
    else:
        cosine = np.zeros(n_samples, dtype=np.float32)

    query_types = [parse_query_type(row) for row in meta_rows]
    query_one_hot = np.zeros((n_samples, len(QUERY_TYPES)), dtype=np.float32)
    for i, query_type in enumerate(query_types):
        query_one_hot[i, QUERY_TYPE_TO_ID.get(query_type, QUERY_TYPE_TO_ID["general"])] = 1.0

    scalars = np.column_stack(
        [
            cosine.astype(np.float32),
            text_norm.astype(np.float32),
            vision_norm.astype(np.float32),
            (text_norm - vision_norm).astype(np.float32),
        ]
    )
    extra_matrix, extra_cols = _load_extra_features(
        extra_feature_csv=extra_feature_csv,
        meta_df=meta_df,
        n_samples=n_samples,
        feature_columns=extra_feature_columns,
    )

    features = np.concatenate(
        [X_text_norm, X_vision_norm, scalars, query_one_hot, extra_matrix],
        axis=1,
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if return_query_types:
        return features, query_types, extra_cols
    return features


def build_model_profiles(
    Y: np.ndarray,
    C: np.ndarray,
    meta=None,
    query_types: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Build per-model profiles from train-set outcomes and costs.

    Each model profile contains global accuracy, global cost mean/std, plus
    query-type-conditioned accuracy and mean cost for every QUERY_TYPES bucket.
    """
    Y = np.asarray(Y, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    if Y.ndim != 2 or C.ndim != 2:
        raise ValueError("Y and C must be 2D arrays")
    if Y.shape != C.shape:
        raise ValueError(f"Y and C shapes differ: {Y.shape} vs {C.shape}")

    n_samples, n_models = Y.shape
    if query_types is None:
        meta_rows, _ = _coerce_meta(meta, n_samples=n_samples)
        query_types = [parse_query_type(row) for row in meta_rows]
    query_types = list(query_types)
    if len(query_types) != n_samples:
        query_types = (query_types + ["general"] * n_samples)[:n_samples]

    profiles = []
    for model_idx in range(n_models):
        y_col = Y[:, model_idx]
        c_col = C[:, model_idx]
        finite_cost = np.isfinite(c_col)
        global_acc = float(np.nanmean(y_col)) if len(y_col) else 0.0
        mean_cost = float(np.nanmean(c_col[finite_cost])) if finite_cost.any() else 0.0
        std_cost = float(np.nanstd(c_col[finite_cost])) if finite_cost.any() else 0.0

        values = [global_acc, mean_cost, std_cost]
        for query_type in QUERY_TYPES:
            mask = np.array([qt == query_type for qt in query_types], dtype=bool)
            if mask.any():
                qt_acc = float(np.nanmean(y_col[mask]))
                qt_cost_values = c_col[mask]
                qt_finite = np.isfinite(qt_cost_values)
                qt_cost = float(np.nanmean(qt_cost_values[qt_finite])) if qt_finite.any() else mean_cost
            else:
                qt_acc = global_acc
                qt_cost = mean_cost
            values.extend([qt_acc, qt_cost])
        profiles.append(values)

    profiles_arr = np.asarray(profiles, dtype=np.float32)
    return np.nan_to_num(profiles_arr, nan=0.0, posinf=0.0, neginf=0.0)


if HAS_TORCH:

    class PairwiseRiskNet(nn.Module):
        """Pairwise risk estimator with model-profile projection and interaction."""

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
    def __init__(self, n_samples: int, n_models: int):
        self.n_samples = int(n_samples)
        self.n_models = int(n_models)
        self.n_pairs = self.n_samples * self.n_models

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return int(idx)


def _fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)


def _torch_load_compat(path: str | Path, map_location=None):
    if not HAS_TORCH:
        raise ImportError("CER-Router requires PyTorch. Please install torch.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CERRouter(RouterBase):
    """Counterfactual Evidence Risk router."""

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8192,
        epochs: int = 30,
        alpha_brier: float = 0.3,
        lambda_cost: float = 0.0,
        device: Optional[str] = None,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        extra_feature_csv: Optional[str | Path] = None,
        enable_dev: bool = False,
        patience: int = 5,
        random_state: int = 42,
        verbose: int = 1,
    ):
        if not HAS_TORCH:
            raise ImportError("CER-Router requires PyTorch. Please install torch.")

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.alpha_brier = float(alpha_brier)
        self.lambda_cost = float(lambda_cost)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.extra_feature_csv = str(extra_feature_csv) if extra_feature_csv else None
        self.enable_dev = bool(enable_dev)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.verbose = int(verbose)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[PairwiseRiskNet] = None
        self.model_mapping: Optional[Dict[int, str]] = None
        self.reverse_mapping: Optional[Dict[str, int]] = None
        self.costs: Optional[np.ndarray] = None
        self.sample_mean: Optional[np.ndarray] = None
        self.sample_std: Optional[np.ndarray] = None
        self.profile_mean: Optional[np.ndarray] = None
        self.profile_std: Optional[np.ndarray] = None
        self.model_profiles: Optional[np.ndarray] = None
        self.extra_feature_columns: List[str] = []
        self.cost_min: Optional[float] = None
        self.cost_max: Optional[float] = None
        self.sample_dim: Optional[int] = None
        self.model_dim: Optional[int] = None
        self.training_history: List[Dict[str, float]] = []

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _prepare_sample_features(self, X_text, X_vision, meta, fit: bool = False):
        features, query_types, extra_cols = build_sample_features(
            X_text=X_text,
            X_vision=X_vision,
            meta=meta,
            extra_feature_csv=self.extra_feature_csv,
            extra_feature_columns=None if fit else self.extra_feature_columns,
            return_query_types=True,
        )
        if fit:
            self.extra_feature_columns = list(extra_cols)
        return features, query_types

    def _transform_samples(self, features: np.ndarray) -> np.ndarray:
        if self.sample_mean is None or self.sample_std is None:
            raise RuntimeError("CERRouter is not fitted: missing sample standardizer")
        return _apply_standardizer(features, self.sample_mean, self.sample_std)

    def _transform_profiles(self, profiles: np.ndarray) -> np.ndarray:
        if self.profile_mean is None or self.profile_std is None:
            raise RuntimeError("CERRouter is not fitted: missing profile standardizer")
        return _apply_standardizer(profiles, self.profile_mean, self.profile_std)

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
        if Y.shape != C.shape or Y.ndim != 2:
            raise ValueError(f"Y and C must be 2D arrays with the same shape; got {Y.shape} and {C.shape}")

        n_samples, n_models = Y.shape
        if model_mapping is None:
            model_mapping = {i: f"model_{i}" for i in range(n_models)}
        self.model_mapping = dict(model_mapping)
        self.reverse_mapping = {v: k for k, v in self.model_mapping.items()}
        self.costs = np.asarray(costs, dtype=np.float32) if costs is not None else np.nanmean(C, axis=0).astype(np.float32)
        finite_costs = C[np.isfinite(C)]
        self.cost_min = float(cmin) if cmin is not None else (float(np.min(finite_costs)) if finite_costs.size else 0.0)
        self.cost_max = float(cmax) if cmax is not None else (float(np.max(finite_costs)) if finite_costs.size else 1.0)
        if self.cost_max <= self.cost_min:
            self.cost_max = self.cost_min + 1.0

        sample_features_raw, query_types = self._prepare_sample_features(X_text, X_vision, meta, fit=True)
        model_profiles_raw = build_model_profiles(Y, C, meta=meta, query_types=query_types)

        self.sample_mean, self.sample_std = _fit_standardizer(sample_features_raw)
        self.profile_mean, self.profile_std = _fit_standardizer(model_profiles_raw)
        sample_features = self._transform_samples(sample_features_raw)
        model_profiles = self._transform_profiles(model_profiles_raw)
        self.model_profiles = model_profiles.astype(np.float32)

        self.sample_dim = int(sample_features.shape[1])
        self.model_dim = int(model_profiles.shape[1])
        pair_rows = int(n_samples * n_models)
        print(
            f"  CER train shape: N={n_samples}, K={n_models}, pair_rows={pair_rows}, "
            f"sample_dim={self.sample_dim}, model_dim={self.model_dim}"
        )

        self.model = PairwiseRiskNet(
            sample_dim=self.sample_dim,
            model_dim=self.model_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        sample_tensor = torch.as_tensor(sample_features, dtype=torch.float32, device=self.device)
        profile_tensor = torch.as_tensor(model_profiles, dtype=torch.float32, device=self.device)
        fail_tensor = torch.as_tensor(1.0 - Y, dtype=torch.float32, device=self.device)

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
        if monitor_output_dir is not None:
            Path(monitor_output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_brier = 0.0
            total_seen = 0

            for pair_ids in loader:
                pair_ids = pair_ids.to(self.device, non_blocking=True)
                sample_idx = torch.div(pair_ids, n_models, rounding_mode="floor")
                model_idx = pair_ids.remainder(n_models)
                labels = fail_tensor[sample_idx, model_idx]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(sample_tensor[sample_idx], profile_tensor[model_idx])
                bce_loss = bce(logits, labels)
                probs = torch.sigmoid(logits)
                brier_loss = torch.mean((probs - labels) ** 2)
                loss = bce_loss + self.alpha_brier * brier_loss
                loss.backward()
                optimizer.step()

                batch_n = int(labels.numel())
                total_seen += batch_n
                total_loss += float(loss.item()) * batch_n
                total_bce += float(bce_loss.item()) * batch_n
                total_brier += float(brier_loss.item()) * batch_n

            row = {
                "epoch": float(epoch),
                "loss": total_loss / max(total_seen, 1),
                "bce": total_bce / max(total_seen, 1),
                "brier": total_brier / max(total_seen, 1),
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
                row.update({f"dev_{k}": float(v) for k, v in dev_metrics.items() if isinstance(v, (int, float, np.floating))})
                metric = float(dev_metrics.get("rank_score", -dev_metrics.get("brier", math.inf)))
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if monitor_output_dir is not None:
                        torch.save(best_state, Path(monitor_output_dir) / "best_risk_net_state.pt")
                else:
                    stale_epochs += 1
            else:
                metric = -row["loss"]
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            self.training_history.append(row)
            if self.verbose:
                msg = (
                    f"  epoch {epoch:03d}: loss={row['loss']:.6f}, "
                    f"bce={row['bce']:.6f}, brier={row['brier']:.6f}"
                )
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

    def _evaluate_dev(self, X_text_dev, X_vision_dev, meta_dev, Y_dev, C_dev, rank_score_beta: float = 0.1):
        from routers.utils.rank_score import rank_score

        risks = self.predict_risk_matrix(X_text=X_text_dev, X_vision=X_vision_dev, meta=meta_dev)
        preds = self.predict(X_text=X_text_dev, X_vision=X_vision_dev, meta=meta_dev, C=C_dev)
        Y_dev = np.asarray(Y_dev)
        C_dev = np.asarray(C_dev)
        correct = Y_dev[np.arange(len(preds)), preds].astype(float)
        selected_costs = C_dev[np.arange(len(preds)), preds].astype(float)
        accuracy = float(np.nanmean(correct)) if len(correct) else 0.0
        avg_cost = float(np.nanmean(selected_costs)) if len(selected_costs) else 0.0
        cmin = float(self.cost_min if self.cost_min is not None else np.nanmin(C_dev))
        cmax = float(self.cost_max if self.cost_max is not None else np.nanmax(C_dev))
        rs = float(rank_score(accuracy, avg_cost, cmin, cmax, beta=rank_score_beta))
        labels = (1.0 - Y_dev).astype(np.float32)
        brier = float(np.mean((risks - labels) ** 2)) if labels.size else 0.0
        return {"accuracy": accuracy, "avg_cost": avg_cost, "rank_score": rs, "brier": brier}

    def _normalize_cost_matrix(self, C: np.ndarray) -> np.ndarray:
        C_arr = np.asarray(C, dtype=np.float32)
        if C_arr.ndim == 1:
            C_arr = np.broadcast_to(C_arr.reshape(1, -1), (1, C_arr.shape[0])).astype(np.float32)
        cmin = float(self.cost_min if self.cost_min is not None else np.nanmin(C_arr))
        cmax = float(self.cost_max if self.cost_max is not None else np.nanmax(C_arr))
        if cmax <= cmin:
            return np.zeros_like(C_arr, dtype=np.float32)
        normed = (C_arr - cmin) / (cmax - cmin)
        return np.nan_to_num(np.clip(normed, 0.0, 1.0), nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def predict_risk_matrix(
        self,
        X_text: np.ndarray,
        X_vision: np.ndarray,
        meta=None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        del kwargs
        if self.model is None or self.model_profiles is None:
            raise RuntimeError("CERRouter is not fitted")

        sample_features_raw, _ = self._prepare_sample_features(X_text, X_vision, meta, fit=False)
        sample_features = self._transform_samples(sample_features_raw)
        model_profiles = self.model_profiles
        n_samples = sample_features.shape[0]
        n_models = model_profiles.shape[0]
        batch_size = int(batch_size or max(1, min(512, self.batch_size // max(n_models, 1))))

        risks = np.zeros((n_samples, n_models), dtype=np.float32)
        profile_tensor = torch.as_tensor(model_profiles, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                sample_tensor = torch.as_tensor(sample_features[start:end], dtype=torch.float32, device=self.device)
                for model_idx in range(n_models):
                    model_tensor = profile_tensor[model_idx].unsqueeze(0).expand(end - start, -1)
                    logits = self.model(sample_tensor, model_tensor)
                    risks[start:end, model_idx] = torch.sigmoid(logits).detach().cpu().numpy()
        return risks

    def predict(
        self,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
        C: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict selected model indices.

        If C is provided, route by risk + lambda_cost * normalized_cost. If C is
        absent, route risk-only so existing evaluators that do not pass C still
        work.
        """
        if X_text is None or X_vision is None:
            raise ValueError("CERRouter.predict requires X_text and X_vision")

        lambda_cost = float(kwargs.pop("lambda_cost", self.lambda_cost))
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
                "lambda_cost": self.lambda_cost,
                "device": self.device,
                "text_encoder": self.text_encoder,
                "vision_encoder": self.vision_encoder,
                "extra_feature_csv": self.extra_feature_csv,
                "enable_dev": self.enable_dev,
                "patience": self.patience,
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
            f"alpha_brier={self.alpha_brier})"
        )
