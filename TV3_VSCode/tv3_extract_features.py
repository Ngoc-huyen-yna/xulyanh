"""Tính và lưu đặc trưng cho một phiên bản ảnh TV2 đã xuất."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tv3_features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TV2_OUTPUTS = PROJECT_ROOT / "NGUYENTHIANHHONG_2045240092_TV2" / "outputs"
VARIANTS = ("original", "contrast", "clahe", "denoise", "sharpen", "combo")
OPTICAL_NODATA = -1
MIN_OPTICAL_VALID_FRAC = 0.5


def optical_valid_fractions(cache: np.ndarray, cache_indices: np.ndarray) -> np.ndarray:
    """Compute the fraction of resized pixels with valid values in every optical band."""
    fractions = np.empty(len(cache_indices), dtype=np.float32)
    chunk_size = 64
    for start in range(0, len(cache_indices), chunk_size):
        stop = start + chunk_size
        images = np.asarray(cache[cache_indices[start:stop]])
        valid_pixels = np.all(images != OPTICAL_NODATA, axis=-1)
        fractions[start:stop] = valid_pixels.mean(axis=(1, 2))
    return fractions


def make_variant_image(raw_image: np.ndarray, variant: str, norm: dict, transforms: dict) -> np.ndarray:
    """Build one RGB uint8 variant using the normalization and parameters exported by TV2 Hồng."""
    p_lo = np.asarray(norm["p_lo"], dtype=np.float32)
    p_hi = np.asarray(norm["p_hi"], dtype=np.float32)
    rgb_idx = list(norm["rgb_idx"])
    rgb01 = np.clip((np.asarray(raw_image, dtype=np.float32) - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)
    image = (rgb01[..., rgb_idx] * 255).astype(np.uint8)
    params = transforms["params"]

    if variant == "original":
        return image
    if variant == "contrast":
        lo, hi = np.percentile(image, (params["contrast"]["lo"], params["contrast"]["hi"]))
        return np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
    if variant == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(
            clipLimit=params["clahe"]["clip"],
            tileGridSize=(params["clahe"]["grid"],) * 2,
        ).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if variant == "denoise":
        return cv2.medianBlur(image, params["denoise"]["k"])
    if variant == "sharpen":
        sigma = params["sharpen"]["sigma"]
        amount = params["sharpen"]["amount"]
        blur = cv2.GaussianBlur(image, (0, 0), sigma)
        return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
    if variant == "combo":
        denoised = cv2.medianBlur(image, params["denoise"]["k"])
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(
            clipLimit=params["clahe"]["clip"],
            tileGridSize=(params["clahe"]["grid"],) * 2,
        ).apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        sigma = params["sharpen"]["sigma"]
        amount = params["sharpen"]["amount"]
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        return cv2.addWeighted(enhanced, 1 + amount, blur, -amount, 0)
    raise ValueError(f"Biến thể không hỗ trợ: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất Histogram, HOG, GLCM và Edge Detection")
    parser.add_argument("--variant", choices=VARIANTS, default="original", help="Phiên bản ảnh cần chạy")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "TV3_VSCode" / "feature_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = TV2_OUTPUTS / "index.csv"
    cache_path = TV2_OUTPUTS / "cache" / "x_raw_128.npy"
    stats_path = TV2_OUTPUTS / "stats.csv"
    norm_path = TV2_OUTPUTS / "configs" / "norm_stats.json"
    transforms_path = TV2_OUTPUTS / "configs" / "transforms.json"
    required_paths = (index_path, stats_path, cache_path, norm_path, transforms_path)
    if any(not path.exists() for path in required_paths):
        raise FileNotFoundError(
            "Thiếu đầu vào TV2 của Hồng: "
            + ", ".join(str(path) for path in required_paths if not path.exists())
        )
    index = pd.read_csv(index_path)
    if index["sample_id"].duplicated().any():
        raise ValueError(f"{index_path} có sample_id bị trùng.")
    cache = np.load(cache_path, mmap_mode="r")
    if cache.ndim != 4 or cache.shape[-1] != 7:
        raise ValueError(f"Mong đợi cache ảnh TV2 dạng (N,H,W,7), nhận được {cache.shape}")
    stats = pd.read_csv(stats_path)
    if not {"sample_id", "split", "cache_idx", "ok"}.issubset(stats.columns):
        raise ValueError(f"{stats_path} thiếu cột sample_id/split/cache_idx/ok.")
    with norm_path.open(encoding="utf-8") as f:
        norm = json.load(f)
    with transforms_path.open(encoding="utf-8") as f:
        transforms = json.load(f)
    if norm.get("img_size") != 128 or transforms.get("img_size") != 128:
        raise ValueError("Cấu hình TV2 của Hồng không phải ảnh 128x128; cần thống nhất trước khi trích xuất.")
    if list(norm.get("rgb_idx", [])) != [2, 1, 0] or list(transforms.get("rgb_idx", [])) != [2, 1, 0]:
        raise ValueError("Thứ tự RGB trong cấu hình TV2 không khớp [2, 1, 0].")
    cache_indices_by_id = index.set_index("sample_id")["cache_idx"]

    for split in ("train", "val", "test"):
        split_stats = stats[(stats["split"] == split) & stats["ok"].fillna(False).astype(bool)].copy()
        if "valid_frac" in split_stats.columns:
            split_stats = split_stats[split_stats["valid_frac"] >= 0.0]
        sample_ids = split_stats["sample_id"].astype(str).to_numpy()
        cache_indices = cache_indices_by_id.reindex(sample_ids)
        if cache_indices.isna().any():
            raise ValueError(f"Có sample_id trong stats.csv không khớp với index.csv ({split}).")
        cache_indices = cache_indices.to_numpy(dtype=np.int64)
        if np.any(cache_indices < 0) or np.any(cache_indices >= len(cache)):
            raise ValueError(f"cache_idx trong {index_path.name} nằm ngoài phạm vi cache.")
        valid_fractions = optical_valid_fractions(cache, cache_indices)
        keep = valid_fractions >= MIN_OPTICAL_VALID_FRAC
        kept_count = int(keep.sum())
        if kept_count == 0:
            raise ValueError(
                f"Không còn mẫu ở split {split} sau khi yêu cầu ít nhất "
                f"{MIN_OPTICAL_VALID_FRAC:.0%} pixel ảnh hợp lệ."
            )
        print(
            f"{split}: giữ {kept_count}/{len(sample_ids)} mẫu có ít nhất "
            f"{MIN_OPTICAL_VALID_FRAC:.0%} pixel ảnh quang học hợp lệ; "
            f"loại {len(sample_ids) - kept_count} mẫu."
        )

        collected: dict[str, list[np.ndarray]] = {name: [] for name in ("hist", "hog", "glcm", "edge")}
        kept_positions = np.flatnonzero(keep)
        for position in tqdm(kept_positions, desc=f"{args.variant} / {split}"):
            image = make_variant_image(cache[cache_indices[position]], args.variant, norm, transforms)
            result = extract_features(image)
            for name, vector in result.items():
                collected[name].append(vector)

        arrays = {name: np.stack(vectors).astype(np.float32) for name, vectors in collected.items()}
        # Store IDs as fixed-width Unicode so loading the NPZ never needs pickle.
        arrays["sample_id"] = np.asarray(sample_ids[keep], dtype=str)
        arrays["optical_valid_frac"] = valid_fractions[keep]
        save_path = output_dir / f"{args.variant}_{split}.npz"
        np.savez_compressed(save_path, **arrays)
        print(f"Đã lưu: {save_path}")
        print("Kích thước feature:", {name: arrays[name].shape for name in collected})


if __name__ == "__main__":
    main()
