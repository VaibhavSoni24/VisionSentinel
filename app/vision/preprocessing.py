"""Frame preprocessing, filtering, and normalization utilities using OpenCV 5."""

from typing import Tuple

import cv2
import numpy as np


def resize_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resizes frame to target dimensions."""
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Applies Gaussian smoothing to suppress high-frequency sensor noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=sigma)


def normalize_illumination(frame: np.ndarray) -> np.ndarray:
    """Normalizes illumination using CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel."""
    if len(frame.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(frame)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def extract_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extracts a padded region of interest (ROI) from a frame.

    Args:
        frame: BGR or grayscale image array.
        bbox: (x, y, w, h) bounding box.
        padding: Fractional padding to expand around the bounding box.

    Returns:
        (cropped_image, (padded_x, padded_y, padded_w, padded_h))
    """
    x, y, w, h = bbox
    img_h, img_w = frame.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    crop = frame[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)
