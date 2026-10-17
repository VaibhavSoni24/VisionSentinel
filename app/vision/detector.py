"""Local object detector wrapper compatible with OpenCV 5 DNN (Mode A vs Mode B)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.api.schemas import CandidateCategory
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SemanticDetection:
    """Detection result produced by the local neural network."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    category: str


class LocalDetector:
    """Manages OpenCV 5 DNN execution for local person detection."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.model_path
        self.net: Optional[cv2.dnn.Net] = None
        self._is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load ONNX model via OpenCV 5 DNN module."""
        if not settings.local_model_enabled:
            logger.info("Local detector disabled via configuration.")
            return

        if not self.model_path.is_file():
            logger.info(
                "Local model weights not found at %s. Operating in Mode B (Classical OpenCV Motion Candidate).",
                self.model_path,
            )
            return

        try:
            # OpenCV 5 DNN load from ONNX
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            # Set preferable backend and target to CPU (portable and zero external dependencies)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._is_loaded = True
            logger.info(
                "Successfully loaded local ONNX detector from %s (Mode A).", self.model_path
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize OpenCV DNN model from %s: %s. Falling back to Mode B.",
                self.model_path,
                e,
            )
            self.net = None
            self._is_loaded = False

    @property
    def is_available(self) -> bool:
        """Returns True if local model is active and loaded."""
        return self._is_loaded and self.net is not None

    def detect_person(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.50,
    ) -> List[SemanticDetection]:
        """Runs forward inference on an image or crop to detect person objects.

        If model is absent (Mode B), returns an empty list without error.
        """
        if not self.is_available:
            return []

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []

        try:
            # Standard pre-processing blob
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=1.0 / 255.0,
                size=(256, 256),
                mean=(0, 0, 0),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            output = self.net.forward()

            detections: List[SemanticDetection] = []
            # Parse standard ONNX detection tensor output format
            if isinstance(output, np.ndarray) and output.ndim >= 2:
                # Format: [batch, num_detections, 7] or [num_detections, 6]
                flat_out = output.reshape(-1, output.shape[-1])
                for row in flat_out:
                    conf = float(row[2]) if len(row) > 2 else float(row[1])
                    if conf >= confidence_threshold:
                        # Extract coordinates normalized to image
                        if len(row) >= 6:
                            bx = int(row[3] * w)
                            by = int(row[4] * h)
                            bw = int((row[5] - row[3]) * w)
                            bh = int((row[6] - row[4]) * h)
                            detections.append(
                                SemanticDetection(
                                    bbox=(max(0, bx), max(0, by), max(1, bw), max(1, bh)),
                                    confidence=conf,
                                    category=CandidateCategory.PERSON.value,
                                )
                            )

            return detections
        except Exception as e:
            logger.warning("Error during local DNN inference: %s", e)
            return []


# Global detector instance singleton
local_detector = LocalDetector()
