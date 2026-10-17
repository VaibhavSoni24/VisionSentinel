"""Fixed-camera motion extraction and candidate segmentation using OpenCV 5."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app.vision.preprocessing import apply_gaussian_blur


@dataclass
class MotionCandidate:
    """Represents a detected moving object region."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]  # (cx, cy)
    area: float
    aspect_ratio: float  # h / w
    extent: float  # contour_area / (w * h)
    contour: np.ndarray


class MotionExtractor:
    """OpenCV 5 Background Subtraction pipeline for fixed-camera surveillance."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_contour_area: int = 250,
        max_contour_area_fraction: float = 0.65,
        morph_kernel_size: int = 3,
        subtractor_type: str = "MOG2",
    ):
        self.min_contour_area = min_contour_area
        self.max_contour_area_fraction = max_contour_area_fraction
        self.morph_kernel_size = morph_kernel_size

        if subtractor_type.upper() == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=var_threshold * 10, detectShadows=detect_shadows
            )
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows
            )

        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size * 2 + 1, morph_kernel_size * 2 + 1)
        )

    def extract(
        self,
        frame: np.ndarray,
        learning_rate: float = -1,
    ) -> Tuple[np.ndarray, List[MotionCandidate]]:
        """Processes a single frame and extracts moving candidate regions.

        Returns:
            (clean_foreground_mask, list_of_motion_candidates)
        """
        # 1. High-frequency noise suppression
        blurred = apply_gaussian_blur(frame, kernel_size=3)

        # 2. Background subtraction
        fg_mask = self.subtractor.apply(blurred, learningRate=learning_rate)

        # 3. Suppress shadows: In OpenCV MOG2, shadow pixels are marked 127, foreground 255
        _, clean_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # 4. Morphological noise reduction: Open removes noise, Close bridges gaps
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, self.kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 5. Extract contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[MotionCandidate] = []
        frame_area = frame.shape[0] * frame.shape[1]
        max_area = frame_area * self.max_contour_area_fraction

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            # Centroid calculation via image moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            aspect_ratio = float(h) / float(w)
            bbox_area = float(w * h)
            extent = area / bbox_area if bbox_area > 0 else 0.0

            candidates.append(
                MotionCandidate(
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    area=area,
                    aspect_ratio=aspect_ratio,
                    extent=extent,
                    contour=cnt,
                )
            )

        return clean_mask, candidates
