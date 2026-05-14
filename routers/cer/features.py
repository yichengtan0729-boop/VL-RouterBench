#!/usr/bin/env python3
"""Feature builders for CER-Router."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
    PANDAS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in broken envs
    pd = None
    HAS_PANDAS = False
    PANDAS_IMPORT_ERROR = exc


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

TEXT_FIELDS: Tuple[str, ...] = (
    "question",
    "prompt",
    "query",
    "text",
    "instruction",
    "problem",
    "input",
)


def require_pandas():
    if not HAS_PANDAS:
        raise ImportError(f"CER metadata handling requires pandas: {PANDAS_IMPORT_ERROR}")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if HAS_PANDAS:
        try:
            return bool(pd.isna(value))
        except Exception:
            return False
    return False


def safe_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _fallback_sample_ids(n_samples: int) -> List[str]:
    return [f"sample_{i}" for i in range(int(n_samples))]


def coerce_meta(meta: Any, n_samples: int) -> Tuple[List[Dict[str, Any]], Any]:
    """
    Return metadata rows plus a pandas DataFrame when pandas is available.

    Missing or short metadata is padded conservatively. If sample_id is absent,
    deterministic fallback ids are added so downstream joins and prediction
    files keep working.
    """
    n_samples = int(n_samples)

    if HAS_PANDAS:
        if meta is None:
            meta_df = pd.DataFrame(index=np.arange(n_samples))
        elif isinstance(meta, pd.DataFrame):
            meta_df = meta.reset_index(drop=True).copy()
        else:
            try:
                meta_df = pd.DataFrame(meta).reset_index(drop=True)
            except Exception:
                meta_df = pd.DataFrame(index=np.arange(n_samples))

        if len(meta_df) != n_samples:
            if len(meta_df) > n_samples:
                meta_df = meta_df.iloc[:n_samples].copy()
            else:
                meta_df = meta_df.reindex(range(n_samples)).copy()

        if "sample_id" not in meta_df.columns:
            meta_df["sample_id"] = _fallback_sample_ids(n_samples)
        else:
            fallback_ids = _fallback_sample_ids(n_samples)
            meta_df["sample_id"] = [
                fallback_ids[i] if _is_missing(value) or safe_text(value) == "" else value
                for i, value in enumerate(meta_df["sample_id"].tolist())
            ]

        if "dataset" not in meta_df.columns:
            meta_df["dataset"] = ""

        rows = [row.to_dict() for _, row in meta_df.iterrows()]
        return rows, meta_df

    rows: List[Dict[str, Any]] = []
    if isinstance(meta, list):
        for item in meta[:n_samples]:
            try:
                rows.append(dict(item))
            except Exception:
                rows.append({})
    elif isinstance(meta, dict):
        for i in range(n_samples):
            row = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > i:
                    row[key] = value[i]
                else:
                    row[key] = value
            rows.append(row)

    if len(rows) < n_samples:
        rows.extend({} for _ in range(n_samples - len(rows)))

    fallback_ids = _fallback_sample_ids(n_samples)
    for i, row in enumerate(rows):
        if "sample_id" not in row or safe_text(row.get("sample_id")) == "":
            row["sample_id"] = fallback_ids[i]
        if "dataset" not in row:
            row["dataset"] = ""
    return rows[:n_samples], None


def ensure_meta_frame(meta: Any, n_samples: int):
    """Return a pandas DataFrame with sample_id/dataset fallbacks."""
    require_pandas()
    _, meta_df = coerce_meta(meta, n_samples=n_samples)
    return meta_df


def parse_query_type(meta_row: Any) -> str:
    """
    Infer a coarse query type from common text fields and dataset names.

    Supported labels are: ocr, counting, spatial, chart, fine_grained, stem,
    and general. The parser is heuristic by design and falls back to general.
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

    pieces = [safe_text(row.get(field, "")) for field in TEXT_FIELDS]
    dataset = safe_text(row.get("dataset", ""))
    haystack = " ".join(pieces + [dataset]).lower()
    dataset_l = dataset.lower()

    dataset_rules = [
        ("chart", ("chart", "plotqa", "dvqa", "figureqa", "tabmwp")),
        ("ocr", ("ocr", "textvqa", "docvqa", "st-vqa", "stvqa", "infographic", "receipt")),
        ("counting", ("count", "tallyqa", "howmany")),
        ("spatial", ("spatial", "relation", "vcr", "gqa", "nlvr")),
        ("fine_grained", ("fgvc", "fine", "birds", "cub", "cars", "aircraft", "food", "logo")),
        ("stem", ("science", "math", "ai2d", "mathvista", "geometry", "chemistry", "physics")),
    ]
    for query_type, keys in dataset_rules:
        if any(key in dataset_l for key in keys):
            return query_type

    text_rules = [
        (
            "chart",
            (
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
            ),
        ),
        (
            "ocr",
            (
                "ocr",
                "read the text",
                "text in the image",
                "what does the sign say",
                "document",
                "receipt",
                "license plate",
                "handwritten",
                "transcribe",
            ),
        ),
        (
            "counting",
            (
                "how many",
                "count",
                "number of",
                "total number",
                "many objects",
                "how much",
                "quantity",
            ),
        ),
        (
            "spatial",
            (
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
            ),
        ),
        (
            "fine_grained",
            (
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
            ),
        ),
        (
            "stem",
            (
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
            ),
        ),
    ]
    for query_type, terms in text_rules:
        if any(term in haystack for term in terms):
            return query_type
    return "general"


def get_query_types(meta: Any, n_samples: int) -> List[str]:
    rows, _ = coerce_meta(meta, n_samples=n_samples)
    return [parse_query_type(row) for row in rows]


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).astype(np.float32)
    normalized = matrix / np.maximum(norms, eps)
    return normalized.astype(np.float32), norms.squeeze(1).astype(np.float32)


def _load_extra_features(
    extra_feature_csv: Optional[str | Path],
    meta_df: Any,
    n_samples: int,
    feature_columns: Optional[Sequence[str]] = None,
    enabled: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if not enabled or not extra_feature_csv:
        return np.zeros((n_samples, 0), dtype=np.float32), []

    if not HAS_PANDAS:
        warnings.warn("extra_feature_csv requires pandas; extra features are ignored")
        return np.zeros((n_samples, 0), dtype=np.float32), []

    path = Path(extra_feature_csv)
    if not path.exists():
        warnings.warn(f"extra_feature_csv not found, ignoring: {path}")
        columns = list(feature_columns or [])
        return np.zeros((n_samples, len(columns)), dtype=np.float32), columns

    extra_df = pd.read_csv(path)
    if "sample_id" not in extra_df.columns:
        warnings.warn("extra_feature_csv has no sample_id column; filling extra features with zeros")
        if feature_columns is None:
            columns = [col for col in extra_df.select_dtypes(include=[np.number]).columns if col != "sample_id"]
        else:
            columns = list(feature_columns)
        return np.zeros((n_samples, len(columns)), dtype=np.float32), columns

    if feature_columns is None:
        columns = [col for col in extra_df.select_dtypes(include=[np.number]).columns if col != "sample_id"]
    else:
        columns = list(feature_columns)
        for col in columns:
            if col not in extra_df.columns:
                extra_df[col] = 0.0

    if not columns:
        return np.zeros((n_samples, 0), dtype=np.float32), []

    if meta_df is None or "sample_id" not in meta_df.columns:
        warnings.warn("meta has no sample_id column; filling extra features with zeros")
        return np.zeros((n_samples, len(columns)), dtype=np.float32), columns

    aligned = (
        meta_df[["sample_id"]]
        .merge(extra_df[["sample_id"] + columns], on="sample_id", how="left")
        .loc[:, columns]
        .fillna(0.0)
    )
    return aligned.to_numpy(dtype=np.float32), columns


def build_sample_features(
    X_text: np.ndarray,
    X_vision: np.ndarray,
    meta: Any = None,
    extra_feature_csv: Optional[str | Path] = None,
    extra_feature_columns: Optional[Sequence[str]] = None,
    include_embeddings: bool = True,
    include_query_type: bool = True,
    include_extra_features: bool = True,
    random_extra_features: bool = False,
    random_state: int = 42,
    return_query_types: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[str], List[str]]:
    """
    Build sample-side CER features.

    Full mode includes normalized text embeddings, normalized vision
    embeddings, image-text cosine similarity, text norm, vision norm, norm gap,
    query_type one-hot, and optional sample_id-aligned extra features.
    """
    X_text = np.asarray(X_text, dtype=np.float32)
    X_vision = np.asarray(X_vision, dtype=np.float32)
    if X_text.ndim != 2 or X_vision.ndim != 2:
        raise ValueError("X_text and X_vision must be 2D arrays")
    if X_text.shape[0] != X_vision.shape[0]:
        raise ValueError(f"X_text and X_vision row counts differ: {X_text.shape[0]} vs {X_vision.shape[0]}")

    n_samples = X_text.shape[0]
    rows, meta_df = coerce_meta(meta, n_samples=n_samples)
    X_text_norm, text_norm = _l2_normalize(X_text)
    X_vision_norm, vision_norm = _l2_normalize(X_vision)

    shared_dim = min(X_text_norm.shape[1], X_vision_norm.shape[1])
    if shared_dim:
        cosine = np.sum(X_text_norm[:, :shared_dim] * X_vision_norm[:, :shared_dim], axis=1)
    else:
        cosine = np.zeros(n_samples, dtype=np.float32)

    blocks: List[np.ndarray] = []
    if include_embeddings:
        blocks.extend([X_text_norm, X_vision_norm])

    scalars = np.column_stack(
        [
            cosine.astype(np.float32),
            text_norm.astype(np.float32),
            vision_norm.astype(np.float32),
            (text_norm - vision_norm).astype(np.float32),
        ]
    )
    blocks.append(scalars)

    query_types = [parse_query_type(row) for row in rows]
    if include_query_type:
        query_one_hot = np.zeros((n_samples, len(QUERY_TYPES)), dtype=np.float32)
        for i, query_type in enumerate(query_types):
            query_one_hot[i, QUERY_TYPE_TO_ID.get(query_type, QUERY_TYPE_TO_ID["general"])] = 1.0
        blocks.append(query_one_hot)

    extra_matrix, extra_columns = _load_extra_features(
        extra_feature_csv=extra_feature_csv,
        meta_df=meta_df,
        n_samples=n_samples,
        feature_columns=extra_feature_columns,
        enabled=include_extra_features,
    )
    if random_extra_features and extra_matrix.shape[1] > 0:
        rng = np.random.default_rng(int(random_state))
        extra_matrix = rng.standard_normal(extra_matrix.shape).astype(np.float32)
    if extra_matrix.shape[1] > 0:
        blocks.append(extra_matrix)

    features = np.concatenate(blocks, axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if return_query_types:
        return features, query_types, extra_columns
    return features


def build_model_profiles(
    Y: np.ndarray,
    C: np.ndarray,
    meta: Any = None,
    query_types: Optional[Sequence[str]] = None,
    include_query_stats: bool = True,
    include_cost: bool = True,
    mode: str = "profile",
) -> np.ndarray:
    """
    Build per-model profiles from training-set outcomes and costs.

    Full profiles contain global accuracy, global cost mean/std, and
    query-type-conditioned accuracy/cost statistics. The no_model_profile
    ablation uses a constant profile with the same interface.
    """
    Y = np.asarray(Y, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    if Y.ndim != 2 or C.ndim != 2:
        raise ValueError("Y and C must be 2D arrays")
    if Y.shape != C.shape:
        raise ValueError(f"Y and C shapes differ: {Y.shape} vs {C.shape}")

    n_samples, n_models = Y.shape
    if mode == "constant":
        return np.ones((n_models, 1), dtype=np.float32)

    if query_types is None:
        query_types = get_query_types(meta, n_samples=n_samples)
    query_types = list(query_types)
    if len(query_types) != n_samples:
        query_types = (query_types + ["general"] * n_samples)[:n_samples]

    profiles: List[List[float]] = []
    for model_idx in range(n_models):
        y_col = Y[:, model_idx]
        c_col = C[:, model_idx]
        finite_cost = np.isfinite(c_col)

        global_acc = float(np.nanmean(y_col)) if y_col.size else 0.0
        mean_cost = float(np.nanmean(c_col[finite_cost])) if finite_cost.any() else 0.0
        std_cost = float(np.nanstd(c_col[finite_cost])) if finite_cost.any() else 0.0
        if not include_cost:
            mean_cost = 0.0
            std_cost = 0.0

        values = [global_acc, mean_cost, std_cost]
        if include_query_stats:
            query_type_arr = np.asarray(query_types)
            for query_type in QUERY_TYPES:
                mask = query_type_arr == query_type
                if mask.any():
                    qt_acc = float(np.nanmean(y_col[mask]))
                    qt_cost_values = c_col[mask]
                    qt_finite = np.isfinite(qt_cost_values)
                    qt_cost = float(np.nanmean(qt_cost_values[qt_finite])) if qt_finite.any() else mean_cost
                else:
                    qt_acc = global_acc
                    qt_cost = mean_cost
                if not include_cost:
                    qt_cost = 0.0
                values.extend([qt_acc, qt_cost])
        profiles.append(values)

    profiles_arr = np.asarray(profiles, dtype=np.float32)
    return np.nan_to_num(profiles_arr, nan=0.0, posinf=0.0, neginf=0.0)
