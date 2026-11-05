"""Các hàm trích xuất đặc trưng cho ảnh RGB uint8 do TV2 xuất."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog


def extract_features(image: np.ndarray) -> dict[str, np.ndarray]:
    """Nhận một ảnh (H, W, 3) RGB uint8, trả về bốn vector đặc trưng."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Cần ảnh có dạng (H,W,3), nhận được {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Histogram màu: 16 bins cho mỗi kênh RGB, chuẩn hóa tổng mỗi kênh = 1.
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=16, range=(0, 256))
        histograms.append(hist.astype(np.float32) / max(hist.sum(), 1))
    color_hist = np.concatenate(histograms)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray01 = gray.astype(np.float32) / 255.0

    # HOG: hướng gradient theo các ô 16x16 pixel.
    hog_vector = hog(
        gray01,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # GLCM: lượng tử ảnh xám về 8 mức rồi lấy 6 thống kê ở 4 hướng, 2 khoảng.
    gray8 = np.minimum(gray // 32, 7).astype(np.uint8)
    matrix = graycomatrix(
        gray8,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    glcm_values = [
        graycoprops(matrix, prop).ravel()
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    ]
    texture = np.nan_to_num(np.concatenate(glcm_values), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Edge: mật độ pixel biên Canny và histogram 8 hướng gradient có trọng số.
    edges = cv2.Canny(gray, threshold1=60, threshold2=120)
    edge_density = np.array([np.mean(edges > 0)], dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    selected = edges > 0
    if selected.any():
        orientation_hist, _ = np.histogram(
            angle[selected], bins=8, range=(0, 2 * np.pi), weights=magnitude[selected]
        )
    else:
        orientation_hist = np.zeros(8, dtype=np.float32)
    orientation_hist = orientation_hist.astype(np.float32)
    orientation_hist /= max(float(orientation_hist.sum()), 1e-8)
    edge_features = np.concatenate([edge_density, orientation_hist])

    return {"hist": color_hist, "hog": hog_vector, "glcm": texture, "edge": edge_features}
