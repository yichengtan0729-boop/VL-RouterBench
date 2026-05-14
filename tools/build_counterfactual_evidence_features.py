#!/usr/bin/env python3
"""Build query-conditioned counterfactual evidence features for CER-Full."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFilter

    HAS_PIL = True
except Exception:  # pragma: no cover - depends on runtime environment
    Image = None
    ImageDraw = None
    ImageFilter = None
    HAS_PIL = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from routers.cer.features import TEXT_FIELDS, parse_query_type, safe_text


REQUIRED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "generic_blur_dist",
    "generic_crop_dist",
    "lowres_dist",
    "query_cond_mean",
    "query_cond_max",
    "query_cond_std",
    "random_perturb_mean",
    "ocr_like_dist",
    "chart_like_dist",
    "counting_like_dist",
    "spatial_like_dist",
)


def model_name_to_filename(model_name: str) -> str:
    return str(model_name).split("/")[-1]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def stable_unit(value: Any, salt: str = "") -> float:
    payload = f"{value}|{salt}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return float(integer / float(2**64 - 1))


def clamp(value: float, low: float = 0.0, high: float = 2.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, low), high))


def parse_jsonish(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return text


def load_benchmark_metadata(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    benchmarks_dir = dataset_dir / "BENCHMARKS"
    records: Dict[str, Dict[str, Any]] = {}
    if not benchmarks_dir.exists():
        return records

    for samples_file in benchmarks_dir.rglob("*_samples.jsonl"):
        with samples_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = safe_text(row.get("sample_id", ""))
                if sample_id:
                    records[sample_id] = row
    return records


def merge_meta_row(meta_row: Dict[str, Any], benchmark_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sample_id = safe_text(meta_row.get("sample_id", ""))
    merged = dict(meta_row)
    benchmark_row = benchmark_records.get(sample_id, {})
    for key, value in benchmark_row.items():
        if key not in merged or is_missing(merged.get(key)) or safe_text(merged.get(key)) == "":
            merged[key] = value
    return merged


def resolve_embedding_path(
    dataset_dir: Path,
    kind: str,
    model_name: str,
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return dataset_dir / "EMBEDDINGS" / kind / f"{model_name_to_filename(model_name)}.parquet"


def load_embedding_map(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"Warning: failed to read embeddings from {path}: {exc}")
        return {}
    if "sample_id" not in df.columns or "embedding" not in df.columns:
        return {}

    mapping: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        sample_id = safe_text(row.get("sample_id", ""))
        if not sample_id:
            continue
        try:
            vector = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if vector.size:
            mapping[sample_id] = vector
    return mapping


def resolve_local_path(uri: str, dataset_dir: Path) -> Optional[Path]:
    text = str(uri).strip()
    if not text or text.lower().startswith(("http://", "https://")):
        return None
    candidates = [
        Path(text),
        dataset_dir / text,
        dataset_dir / "TSV_images" / text,
        Path.cwd() / text,
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


class TsvImageCache:
    """Small TSV image loader with frame-level caching."""

    def __init__(self) -> None:
        self.frames: Dict[str, pd.DataFrame] = {}

    def _load_frame(self, path: Path) -> pd.DataFrame:
        key = str(path)
        if key not in self.frames:
            frame = pd.read_csv(path, sep="\t")
            if "image" in frame.columns and "index" in frame.columns:
                image_map = {str(idx): str(img) for idx, img in zip(frame["index"], frame["image"])}
                for idx, image_value in list(image_map.items()):
                    if len(image_value) <= 64 and image_value in image_map and len(image_map[image_value]) > 64:
                        image_map[idx] = image_map[image_value]
                frame = frame.copy()
                frame["image"] = frame["index"].map(lambda idx: image_map.get(str(idx), ""))
            self.frames[key] = frame
        return self.frames[key]

    def load_image(self, tsv_file: str, index: Any) -> Optional[Any]:
        if not HAS_PIL:
            return None
        path = Path(tsv_file)
        if not path.exists():
            return None
        try:
            row_idx = int(index)
            frame = self._load_frame(path)
            if row_idx < 0 or row_idx >= len(frame) or "image" not in frame.columns:
                return None
            image_text = str(frame.iloc[row_idx]["image"])
            if "," in image_text and image_text.lower().startswith("data:image"):
                image_text = image_text.split(",", 1)[1]
            image_data = base64.b64decode(image_text)
            return Image.open(BytesIO(image_data)).convert("RGB")
        except Exception:
            return None


def iter_assets(row: Dict[str, Any]) -> Iterable[Any]:
    assets = parse_jsonish(row.get("assets"))
    if isinstance(assets, dict):
        yield assets
    elif isinstance(assets, list):
        for asset in assets:
            yield asset

    for key in ("image_path", "img_path", "image", "Image", "asset", "uri"):
        value = row.get(key)
        if is_missing(value):
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                yield {"type": "image", "uri": item}
            continue
        for piece in str(value).replace(",", ";").split(";"):
            piece = piece.strip()
            if piece:
                yield {"type": "image", "uri": piece}


def load_image_for_row(row: Dict[str, Any], dataset_dir: Path, tsv_cache: TsvImageCache) -> Optional[Any]:
    if not HAS_PIL:
        return None
    for asset in iter_assets(row):
        if isinstance(asset, str):
            asset = {"type": "image", "uri": asset}
        if not isinstance(asset, dict):
            continue
        asset_type = safe_text(asset.get("type", "image"))
        if asset_type == "image_tsv":
            image = tsv_cache.load_image(safe_text(asset.get("tsv_file", "")), asset.get("index", 0))
            if image is not None:
                return image
        else:
            uri = asset.get("uri", asset.get("path", asset.get("image_path", "")))
            local_path = resolve_local_path(safe_text(uri), dataset_dir)
            if local_path is None:
                continue
            try:
                return Image.open(local_path).convert("RGB")
            except Exception:
                continue
    return None


def resize_square(image: Any, size: int) -> Any:
    return image.convert("RGB").resize((int(size), int(size)))


def center_crop_resize(image: Any, fraction: float, size: Tuple[int, int]) -> Any:
    width, height = image.size
    crop_width = max(1, int(width * float(fraction)))
    crop_height = max(1, int(height * float(fraction)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(size)


def lowres_image(image: Any, factor: int = 4) -> Any:
    width, height = image.size
    small = image.resize((max(8, width // factor), max(8, height // factor)))
    return small.resize((width, height))


def text_like_mask(image: Any, use_ocr_boxes: bool = False, row: Optional[Dict[str, Any]] = None) -> Any:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    width, height = masked.size
    fill = tuple(int(value) for value in np.asarray(masked).reshape(-1, 3).mean(axis=0))

    boxes = []
    if use_ocr_boxes and row is not None:
        parsed = parse_jsonish(row.get("ocr_boxes"))
        if isinstance(parsed, list):
            boxes = parsed

    if boxes:
        for box in boxes:
            try:
                x0, y0, x1, y1 = [int(float(value)) for value in box[:4]]
                draw.rectangle((x0, y0, x1, y1), fill=fill)
            except Exception:
                continue
    else:
        band_h = max(2, height // 18)
        for offset in (0.18, 0.34, 0.52, 0.70):
            y0 = int(height * offset)
            draw.rectangle((int(width * 0.08), y0, int(width * 0.92), y0 + band_h), fill=fill)
    return masked


def chart_region_mask(image: Any) -> Any:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    width, height = masked.size
    fill = tuple(int(value) for value in np.asarray(masked).reshape(-1, 3).mean(axis=0))
    draw.rectangle((0, int(height * 0.82), width, height), fill=fill)
    draw.rectangle((0, 0, int(width * 0.16), height), fill=fill)
    return masked


def object_density_proxy(image: Any) -> Any:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    return Image.blend(image.convert("RGB"), edges, 0.45)


def background_perturb(image: Any) -> Any:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=4))
    width, height = image.size
    left = int(width * 0.22)
    top = int(height * 0.22)
    right = int(width * 0.78)
    bottom = int(height * 0.78)
    blurred.paste(image.crop((left, top, right, bottom)), (left, top))
    return blurred


def random_mask(image: Any, seed: int) -> Any:
    rng = np.random.default_rng(int(seed))
    perturbed = image.copy()
    draw = ImageDraw.Draw(perturbed)
    width, height = image.size
    fill = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
    for _ in range(3):
        x0 = int(rng.uniform(0.0, 0.75) * width)
        y0 = int(rng.uniform(0.0, 0.75) * height)
        x1 = min(width, x0 + int(rng.uniform(0.08, 0.28) * width))
        y1 = min(height, y0 + int(rng.uniform(0.08, 0.28) * height))
        draw.rectangle((x0, y0, x1, y1), fill=fill)
    return perturbed


def image_descriptor(image: Any, size: int = 128) -> np.ndarray:
    original_width, original_height = image.size
    image = resize_square(image, size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)

    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_hist, _ = np.histogram(edge, bins=8, range=(0.0, max(float(edge.max()), 1e-6)))
    edge_hist = edge_hist.astype(np.float32) / max(float(edge_hist.sum()), 1.0)

    center = gray[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = False
    border = gray[border_mask]
    aspect = math.log((float(original_width) + 1.0) / (float(original_height) + 1.0))

    scalars = np.asarray(
        [
            arr[..., 0].mean(),
            arr[..., 1].mean(),
            arr[..., 2].mean(),
            arr[..., 0].std(),
            arr[..., 1].std(),
            arr[..., 2].std(),
            edge.mean(),
            edge.std(),
            float((edge > edge.mean() + edge.std()).mean()),
            gray.mean(),
            gray.std(),
            center.mean() if center.size else gray.mean(),
            border.mean() if border.size else gray.mean(),
            gray.mean(axis=0).std(),
            gray.mean(axis=1).std(),
            aspect,
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist, edge_hist, scalars]).astype(np.float32)


def descriptor_distance(base: np.ndarray, other: np.ndarray) -> float:
    if base.shape != other.shape:
        return 0.0
    return float(np.linalg.norm(base - other) / math.sqrt(max(1, base.size)))


def embedding_distance(base: np.ndarray, other: np.ndarray) -> float:
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    other = np.asarray(other, dtype=np.float32).reshape(-1)
    dim = min(base.size, other.size)
    if dim == 0:
        return 0.0
    base = base[:dim] / max(float(np.linalg.norm(base[:dim])), 1e-6)
    other = other[:dim] / max(float(np.linalg.norm(other[:dim])), 1e-6)
    return float(np.linalg.norm(base - other) / math.sqrt(float(dim)))


def query_conditioned_stats(query_type: str, features: Dict[str, float]) -> Tuple[float, float, float]:
    if query_type == "ocr":
        values = [features["lowres_dist"], features["ocr_like_dist"], features["generic_blur_dist"]]
    elif query_type == "chart":
        values = [features["lowres_dist"], features["chart_like_dist"], features["generic_crop_dist"]]
    elif query_type == "counting":
        values = [features["counting_like_dist"], features["generic_crop_dist"], features["generic_blur_dist"]]
    elif query_type == "spatial":
        values = [features["spatial_like_dist"], features["generic_crop_dist"], features["random_perturb_mean"]]
    else:
        values = [features["generic_blur_dist"], features["generic_crop_dist"], features["random_perturb_mean"]]
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.max()), float(arr.std())


def image_counterfactual_features(
    row: Dict[str, Any],
    image: Any,
    image_size: int,
    use_ocr_boxes: bool,
    perturb_encoder: Optional[Any] = None,
) -> Dict[str, float]:
    image = resize_square(image, image_size)
    blur = image.filter(ImageFilter.GaussianBlur(radius=1.4))
    crop = center_crop_resize(image, 0.84, image.size)
    lowres = lowres_image(image, factor=4)
    ocr_like = text_like_mask(lowres_image(image, factor=5).filter(ImageFilter.GaussianBlur(radius=0.8)), use_ocr_boxes, row)
    chart_like = chart_region_mask(center_crop_resize(lowres_image(image, factor=3), 0.90, image.size))
    counting_like = object_density_proxy(center_crop_resize(image.filter(ImageFilter.GaussianBlur(radius=0.7)), 0.78, image.size))
    spatial_like = background_perturb(center_crop_resize(image, 0.82, image.size))

    sample_id = row.get("sample_id", "")
    random_images = []
    for salt in ("r0", "r1", "r2"):
        seed = int(stable_unit(sample_id, salt) * (2**32 - 1))
        random_images.append(random_mask(image, seed))

    if perturb_encoder is not None:
        try:
            encoded = perturb_encoder.extract(
                [image, blur, crop, lowres, ocr_like, chart_like, counting_like, spatial_like, *random_images]
            )
            base_vec = encoded[0]
            random_values = [embedding_distance(base_vec, vector) for vector in encoded[8:]]
            features = {
                "generic_blur_dist": embedding_distance(base_vec, encoded[1]),
                "generic_crop_dist": embedding_distance(base_vec, encoded[2]),
                "lowres_dist": embedding_distance(base_vec, encoded[3]),
                "random_perturb_mean": float(np.mean(random_values)) if random_values else 0.0,
                "ocr_like_dist": embedding_distance(base_vec, encoded[4]),
                "chart_like_dist": embedding_distance(base_vec, encoded[5]),
                "counting_like_dist": embedding_distance(base_vec, encoded[6]),
                "spatial_like_dist": embedding_distance(base_vec, encoded[7]),
            }
        except Exception:
            features = {}
    else:
        features = {}

    if not features:
        base = image_descriptor(image)
        random_values = [descriptor_distance(base, image_descriptor(random_image)) for random_image in random_images]
        features = {
            "generic_blur_dist": descriptor_distance(base, image_descriptor(blur)),
            "generic_crop_dist": descriptor_distance(base, image_descriptor(crop)),
            "lowres_dist": descriptor_distance(base, image_descriptor(lowres)),
            "random_perturb_mean": float(np.mean(random_values)) if random_values else 0.0,
            "ocr_like_dist": descriptor_distance(base, image_descriptor(ocr_like)),
            "chart_like_dist": descriptor_distance(base, image_descriptor(chart_like)),
            "counting_like_dist": descriptor_distance(base, image_descriptor(counting_like)),
            "spatial_like_dist": descriptor_distance(base, image_descriptor(spatial_like)),
        }

    query_type = parse_query_type(row)
    mean_value, max_value, std_value = query_conditioned_stats(query_type, features)
    features.update(
        {
            "query_cond_mean": mean_value,
            "query_cond_max": max_value,
            "query_cond_std": std_value,
        }
    )
    return {key: clamp(value) for key, value in features.items()}


def text_proxy_stats(row: Dict[str, Any]) -> Dict[str, float]:
    text = " ".join(safe_text(row.get(field, "")) for field in TEXT_FIELDS).lower()
    length = max(len(text), 1)
    digit_ratio = sum(ch.isdigit() for ch in text) / float(length)
    alpha_ratio = sum(ch.isalpha() for ch in text) / float(length)
    spatial_terms = sum(term in text for term in ("left", "right", "above", "below", "behind", "front", "between"))
    chart_terms = sum(term in text for term in ("chart", "graph", "axis", "legend", "table", "plot"))
    counting_terms = sum(term in text for term in ("how many", "count", "number of", "total"))
    ocr_terms = sum(term in text for term in ("text", "read", "sign", "document", "ocr", "word"))
    return {
        "digit_ratio": digit_ratio,
        "alpha_ratio": alpha_ratio,
        "length_score": min(math.log1p(length) / 8.0, 1.0),
        "spatial_terms": min(spatial_terms / 3.0, 1.0),
        "chart_terms": min(chart_terms / 3.0, 1.0),
        "counting_terms": min(counting_terms / 3.0, 1.0),
        "ocr_terms": min(ocr_terms / 3.0, 1.0),
    }


def embedding_proxy_features(
    row: Dict[str, Any],
    text_embedding: Optional[np.ndarray],
    vision_embedding: Optional[np.ndarray],
) -> Dict[str, float]:
    sample_id = row.get("sample_id", "")
    query_type = parse_query_type(row)
    text_stats = text_proxy_stats(row)
    vector = vision_embedding if vision_embedding is not None else text_embedding

    if vector is None or vector.size == 0:
        spread = 0.05 + 0.10 * stable_unit(sample_id, "spread")
        mean_abs = 0.05 + 0.10 * stable_unit(sample_id, "mean_abs")
        diff = 0.05 + 0.10 * stable_unit(sample_id, "diff")
        pos_ratio = 0.5
    else:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        normalized = vector / max(norm, 1e-6)
        spread = float(np.std(normalized) * math.sqrt(normalized.size))
        mean_abs = float(np.mean(np.abs(normalized)) * math.sqrt(normalized.size))
        diff = float(np.mean(np.abs(np.diff(normalized))) * math.sqrt(max(1, normalized.size)))
        pos_ratio = float((normalized > 0).mean())

    cross_gap = 0.0
    if text_embedding is not None and vision_embedding is not None:
        dim = min(text_embedding.size, vision_embedding.size)
        if dim > 0:
            text_vec = text_embedding[:dim] / max(float(np.linalg.norm(text_embedding[:dim])), 1e-6)
            vision_vec = vision_embedding[:dim] / max(float(np.linalg.norm(vision_embedding[:dim])), 1e-6)
            cross_gap = float(1.0 - np.dot(text_vec, vision_vec))

    jitter = 0.02 * stable_unit(sample_id, "jitter")
    features = {
        "generic_blur_dist": 0.08 + 0.18 * spread + 0.05 * text_stats["length_score"] + jitter,
        "generic_crop_dist": 0.10 + 0.12 * mean_abs + 0.04 * abs(pos_ratio - 0.5) + jitter,
        "lowres_dist": 0.12 + 0.15 * diff + 0.05 * text_stats["digit_ratio"] + jitter,
        "random_perturb_mean": 0.07 + 0.10 * stable_unit(sample_id, "random") + 0.05 * cross_gap,
        "ocr_like_dist": 0.10 + 0.20 * text_stats["ocr_terms"] + 0.12 * text_stats["alpha_ratio"] + 0.05 * cross_gap,
        "chart_like_dist": 0.10 + 0.22 * text_stats["chart_terms"] + 0.10 * text_stats["digit_ratio"] + 0.04 * cross_gap,
        "counting_like_dist": 0.10 + 0.22 * text_stats["counting_terms"] + 0.08 * text_stats["digit_ratio"] + 0.05 * spread,
        "spatial_like_dist": 0.10 + 0.22 * text_stats["spatial_terms"] + 0.08 * cross_gap + 0.04 * mean_abs,
    }
    mean_value, max_value, std_value = query_conditioned_stats(query_type, features)
    features.update(
        {
            "query_cond_mean": mean_value,
            "query_cond_max": max_value,
            "query_cond_std": std_value,
        }
    )
    return {key: clamp(value) for key, value in features.items()}


def build_row_features(
    row: Dict[str, Any],
    dataset_dir: Path,
    tsv_cache: TsvImageCache,
    text_embeddings: Dict[str, np.ndarray],
    vision_embeddings: Dict[str, np.ndarray],
    image_size: int,
    use_ocr_boxes: bool,
    perturb_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    sample_id = safe_text(row.get("sample_id", ""))
    query_type = parse_query_type(row)
    image = load_image_for_row(row, dataset_dir, tsv_cache)
    text_embedding = text_embeddings.get(sample_id)
    vision_embedding = vision_embeddings.get(sample_id)

    if image is not None:
        features = image_counterfactual_features(
            row=row,
            image=image,
            image_size=image_size,
            use_ocr_boxes=use_ocr_boxes,
            perturb_encoder=perturb_encoder,
        )
        source = "image"
    else:
        features = embedding_proxy_features(row, text_embedding=text_embedding, vision_embedding=vision_embedding)
        source = "embedding" if text_embedding is not None or vision_embedding is not None else "metadata"

    output = {
        "sample_id": sample_id,
        "query_type": query_type,
        "evidence_source": source,
        "has_image": 1.0 if image is not None else 0.0,
        "has_embedding": 1.0 if text_embedding is not None or vision_embedding is not None else 0.0,
    }
    output.update(features)
    return output


def complete_cache_rows(cache_df: pd.DataFrame) -> pd.Series:
    if cache_df.empty or "sample_id" not in cache_df.columns:
        return pd.Series([], dtype=bool)
    for column in REQUIRED_FEATURE_COLUMNS:
        if column not in cache_df.columns:
            cache_df[column] = np.nan
    numeric = cache_df.loc[:, list(REQUIRED_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return cache_df["sample_id"].notna() & np.isfinite(numeric).all(axis=1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CER counterfactual evidence features")
    parser.add_argument("--dataset-dir", "--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--meta-path", "--meta_path", default=None, help="Metadata parquet path")
    parser.add_argument(
        "--output-csv",
        "--output_csv",
        default="outputs/features/counterfactual_evidence_features.csv",
        help="Output feature CSV",
    )
    parser.add_argument("--text-encoder", "--text_encoder", default="BAAI/bge-m3")
    parser.add_argument("--vision-encoder", "--vision_encoder", default="facebook/dinov2-base")
    parser.add_argument("--text-embedding-path", "--text_embedding_path", default=None)
    parser.add_argument("--vision-embedding-path", "--vision_embedding_path", default=None)
    parser.add_argument(
        "--perturb-encoder",
        "--perturb_encoder",
        default="descriptor",
        choices=["descriptor", "vision"],
        help="Use fast descriptors or the configured vision encoder for perturbation distances",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", "--max_samples", type=int, default=None)
    parser.add_argument("--image-size", "--image_size", type=int, default=224)
    parser.add_argument("--use-ocr-boxes", "--use_ocr_boxes", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute selected samples even if cached")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    meta_path = Path(args.meta_path) if args.meta_path else dataset_dir / "data" / "registry" / "meta.parquet"
    output_csv = Path(args.output_csv)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.parquet not found: {meta_path}")

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    if "sample_id" not in meta.columns:
        raise ValueError(f"Metadata must contain sample_id: {meta_path}")
    if args.max_samples is not None:
        meta = meta.iloc[: max(0, int(args.max_samples))].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        cache_df = pd.read_csv(output_csv)
    else:
        cache_df = pd.DataFrame(columns=["sample_id", "query_type", "evidence_source", *REQUIRED_FEATURE_COLUMNS])

    complete_mask = complete_cache_rows(cache_df)
    cached_ids = set(cache_df.loc[complete_mask, "sample_id"].astype(str).tolist()) if not args.force else set()
    selected_ids = meta["sample_id"].astype(str).tolist()
    todo_ids = [sample_id for sample_id in selected_ids if sample_id not in cached_ids]

    print("=" * 80)
    print("Building CER counterfactual evidence features")
    print("=" * 80)
    print(f"metadata: {meta_path}")
    print(f"output_csv: {output_csv}")
    print(f"selected_samples: {len(selected_ids)}")
    print(f"cached_complete: {len(cached_ids & set(selected_ids))}")
    print(f"to_compute: {len(todo_ids)}")
    print(f"use_ocr_boxes: {bool(args.use_ocr_boxes)}")

    if not todo_ids:
        for column in REQUIRED_FEATURE_COLUMNS:
            if column not in cache_df.columns:
                cache_df[column] = 0.0
        cache_df.to_csv(output_csv, index=False)
        print("All selected samples already cached.")
        return

    text_path = resolve_embedding_path(dataset_dir, "text", args.text_encoder, args.text_embedding_path)
    vision_path = resolve_embedding_path(dataset_dir, "vision", args.vision_encoder, args.vision_embedding_path)
    text_embeddings = load_embedding_map(text_path)
    vision_embeddings = load_embedding_map(vision_path)
    print(f"text_embeddings: {len(text_embeddings)} from {text_path}")
    print(f"vision_embeddings: {len(vision_embeddings)} from {vision_path}")

    perturb_encoder = None
    if args.perturb_encoder == "vision" and HAS_PIL:
        try:
            from routers.features import VisionEncoder

            perturb_encoder = VisionEncoder(model_name=args.vision_encoder, device=args.device)
            print(f"perturb_encoder: vision ({args.vision_encoder})")
        except Exception as exc:
            print(f"Warning: failed to initialize vision perturb encoder; using descriptor fallback: {exc}")
            perturb_encoder = None
    else:
        print("perturb_encoder: descriptor")

    if not HAS_PIL:
        print("Warning: pillow is unavailable; falling back to embedding/metadata proxy features.")

    benchmark_records = load_benchmark_metadata(dataset_dir)
    tsv_cache = TsvImageCache()
    todo_set = set(todo_ids)
    new_rows: List[Dict[str, Any]] = []
    for _, meta_row in meta.iterrows():
        sample_id = safe_text(meta_row.get("sample_id", ""))
        if sample_id not in todo_set:
            continue
        row = merge_meta_row(meta_row.to_dict(), benchmark_records)
        new_rows.append(
            build_row_features(
                row=row,
                dataset_dir=dataset_dir,
                tsv_cache=tsv_cache,
                text_embeddings=text_embeddings,
                vision_embeddings=vision_embeddings,
                image_size=int(args.image_size),
                use_ocr_boxes=bool(args.use_ocr_boxes),
                perturb_encoder=perturb_encoder,
            )
        )
        if len(new_rows) % 100 == 0:
            print(f"  computed {len(new_rows)}/{len(todo_ids)}")

    new_df = pd.DataFrame(new_rows)
    if args.force and not cache_df.empty and "sample_id" in cache_df.columns:
        keep_cache = cache_df[~cache_df["sample_id"].astype(str).isin(todo_set)].copy()
    else:
        keep_cache = cache_df.copy()
    combined = pd.concat([keep_cache, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    for column in REQUIRED_FEATURE_COLUMNS:
        if column not in combined.columns:
            combined[column] = 0.0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)

    order = {sample_id: idx for idx, sample_id in enumerate(selected_ids)}
    combined["_order"] = combined["sample_id"].astype(str).map(order).fillna(len(order)).astype(int)
    combined = combined.sort_values(["_order", "sample_id"]).drop(columns=["_order"])
    combined.to_csv(output_csv, index=False)
    print(f"Wrote {len(combined)} rows to {output_csv}")


if __name__ == "__main__":
    main()
