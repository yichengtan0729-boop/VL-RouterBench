#!/usr/bin/env python3
"""Split helpers for CER experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from routers.cer.features import coerce_meta, get_query_types, safe_text


def parse_list(values: Optional[Iterable[str] | str]) -> List[str]:
    """Parse comma- or whitespace-separated command line lists."""
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    parsed: List[str] = []
    for value in raw_values:
        for piece in str(value).replace(",", " ").split():
            clean = piece.strip()
            if clean:
                parsed.append(clean)
    return parsed


def _standard_sample_splits(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    splits = data.get("splits", {}) or {}
    meta = data.get("meta")
    n_samples = len(meta) if meta is not None else len(data.get("sample_ids", []))
    rows, _ = coerce_meta(meta, n_samples=n_samples)
    all_ids = [row["sample_id"] for row in rows]

    train_ids = list(splits.get("train", [])) or all_ids
    dev_ids = list(splits.get("dev", []))
    test_ids = list(splits.get("test", []))
    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def _ids_matching_query_type(meta: Any, wanted: List[str]) -> List[Any]:
    n_samples = len(meta)
    rows, _ = coerce_meta(meta, n_samples=n_samples)
    query_types = get_query_types(meta, n_samples=n_samples)
    wanted_set = {item.lower() for item in wanted}
    return [row["sample_id"] for row, query_type in zip(rows, query_types) if query_type.lower() in wanted_set]


def _ids_matching_dataset(meta: Any, wanted: List[str]) -> List[Any]:
    n_samples = len(meta)
    rows, _ = coerce_meta(meta, n_samples=n_samples)
    wanted_set = {item.lower() for item in wanted}
    matched = []
    for row in rows:
        dataset = safe_text(row.get("dataset", "")).lower()
        if dataset in wanted_set or any(token in dataset for token in wanted_set):
            matched.append(row["sample_id"])
    return matched


def make_sample_splits(
    data: Dict[str, Any],
    split_mode: str = "standard",
    heldout_tasks: Optional[Iterable[str] | str] = None,
    heldout_datasets: Optional[Iterable[str] | str] = None,
) -> Dict[str, Any]:
    """Create sample split ids for standard and heldout modes."""
    split_mode = str(split_mode or "standard")
    standard = _standard_sample_splits(data)
    notes: List[str] = []

    if split_mode in ("standard", "heldout_model"):
        return {"splits": standard, "notes": notes}

    meta = data.get("meta")
    if meta is None or len(meta) == 0:
        notes.append(f"{split_mode}: missing metadata; using standard splits")
        return {"splits": standard, "notes": notes}

    if split_mode == "heldout_task":
        wanted = parse_list(heldout_tasks)
        if not wanted:
            notes.append("heldout_task requested without --heldout_tasks; using standard splits")
            return {"splits": standard, "notes": notes}
        heldout_ids = set(_ids_matching_query_type(meta, wanted))
    elif split_mode == "heldout_dataset":
        wanted = parse_list(heldout_datasets)
        if not wanted:
            notes.append("heldout_dataset requested without --heldout_datasets; using standard splits")
            return {"splits": standard, "notes": notes}
        heldout_ids = set(_ids_matching_dataset(meta, wanted))
    else:
        notes.append(f"unknown split_mode={split_mode}; using standard splits")
        return {"splits": standard, "notes": notes}

    if not heldout_ids:
        notes.append(f"{split_mode}: no heldout samples matched; using standard splits")
        return {"splits": standard, "notes": notes}

    train_ids = [sid for sid in standard["train"] if sid not in heldout_ids]
    dev_ids = [sid for sid in standard["dev"] if sid not in heldout_ids]
    test_ids = [sid for sid in standard["test"] if sid in heldout_ids]

    if not test_ids:
        test_ids = list(heldout_ids)
        notes.append(f"{split_mode}: no heldout ids in standard test; using all matched heldout ids as test")
    if not train_ids:
        train_ids = standard["train"]
        notes.append(f"{split_mode}: filtering emptied train; using standard train split")
    if standard["dev"] and not dev_ids:
        dev_ids = standard["dev"]
        notes.append(f"{split_mode}: filtering emptied dev; using standard dev split")

    return {"splits": {"train": train_ids, "dev": dev_ids, "test": test_ids}, "notes": notes}


def make_model_split(models: List[str], heldout_models: Optional[Iterable[str] | str] = None) -> Dict[str, Any]:
    """Create train/eval model index lists."""
    wanted = parse_list(heldout_models)
    n_models = len(models)
    if not wanted:
        return {
            "train_model_indices": list(range(n_models)),
            "eval_model_indices": list(range(n_models)),
            "heldout_model_indices": [],
            "notes": [],
        }

    wanted_l = {item.lower() for item in wanted}
    heldout = []
    for idx, model in enumerate(models):
        basename = str(model).split("/")[-1]
        candidates = {str(idx).lower(), str(model).lower(), basename.lower()}
        if candidates & wanted_l:
            heldout.append(idx)

    notes: List[str] = []
    if not heldout:
        notes.append("heldout_model requested but no models matched; using all models for training")

    train_indices = [idx for idx in range(n_models) if idx not in set(heldout)]
    if not train_indices:
        train_indices = list(range(n_models))
        notes.append("heldout_model filtering removed every model; using all models for training")
        heldout = []

    return {
        "train_model_indices": train_indices,
        "eval_model_indices": list(range(n_models)),
        "heldout_model_indices": heldout,
        "notes": notes,
    }


def resolve_experiment_splits(
    data: Dict[str, Any],
    split_mode: str = "standard",
    heldout_tasks: Optional[Iterable[str] | str] = None,
    heldout_datasets: Optional[Iterable[str] | str] = None,
    heldout_models: Optional[Iterable[str] | str] = None,
) -> Dict[str, Any]:
    """Resolve sample and model splits for one experiment."""
    sample_info = make_sample_splits(
        data=data,
        split_mode=split_mode,
        heldout_tasks=heldout_tasks,
        heldout_datasets=heldout_datasets,
    )
    if split_mode == "heldout_model":
        model_info = make_model_split(list(data.get("models", [])), heldout_models=heldout_models)
    else:
        model_info = make_model_split(list(data.get("models", [])), heldout_models=None)

    notes = list(sample_info.get("notes", [])) + list(model_info.get("notes", []))
    return {
        "splits": sample_info["splits"],
        "train_model_indices": model_info["train_model_indices"],
        "eval_model_indices": model_info["eval_model_indices"],
        "heldout_model_indices": model_info["heldout_model_indices"],
        "notes": notes,
    }


def choose_calibration_indices(n_samples: int, calibration_size: int, seed: int = 42) -> np.ndarray:
    """Choose deterministic calibration rows for heldout-model profiles."""
    n_samples = int(n_samples)
    calibration_size = int(calibration_size)
    if calibration_size <= 0 or n_samples <= 0:
        return np.array([], dtype=int)
    size = min(calibration_size, n_samples)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(np.arange(n_samples), size=size, replace=False)).astype(int)
