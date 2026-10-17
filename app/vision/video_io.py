"""Video I/O and frame acquisition module powered by OpenCV 5."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when video file is corrupt, unreadable, or violates size/duration limits."""

    pass


@dataclass
class VideoMetadata:
    """Detailed metadata for an ingested video stream."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    fourcc: str


class VideoReader:
    """Safely opens, validates, and iterates video frames using OpenCV 5."""

    def __init__(
        self,
        video_path: Path,
        max_duration_seconds: Optional[int] = None,
        max_dimension: int = 1920,
    ):
        self.video_path = Path(video_path)
        self.max_duration = max_duration_seconds or settings.max_video_seconds
        self.max_dimension = max_dimension
        self.metadata = self._probe()

    def _probe(self) -> VideoMetadata:
        """Probes video stream to ensure valid dimensions, frame count, and duration."""
        if not self.video_path.is_file():
            raise VideoProcessingError(f"Video file does not exist: {self.video_path}")

        file_size_mb = self.video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_upload_mb:
            raise VideoProcessingError(
                f"File size {file_size_mb:.1f} MB exceeds limit of {settings.max_upload_mb} MB"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"OpenCV could not open video: {self.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        if width <= 0 or height <= 0:
            raise VideoProcessingError(f"Invalid frame dimensions: {width}x{height}")

        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Sensible default fallback

        duration = total_frames / fps if total_frames > 0 else 0.0
        if duration > self.max_duration:
            logger.warning(
                "Video duration (%.1fs) exceeds limit (%.1fs); will truncate reading.",
                duration,
                self.max_duration,
            )

        return VideoMetadata(
            path=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            fourcc=fourcc,
        )

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields (frame_index, timestamp_seconds, frame_bgr) with bounded processing."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video stream: {self.video_path}")

        frame_idx = 0
        max_allowed_frames = int(self.max_duration * self.metadata.fps)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx >= max_allowed_frames:
                    logger.info("Reached maximum allowed frame limit: %d", max_allowed_frames)
                    break

                # Resize if exceeding maximum permissible dimension
                h, w = frame.shape[:2]
                if max(h, w) > self.max_dimension:
                    scale = self.max_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                timestamp = frame_idx / self.metadata.fps
                yield frame_idx, timestamp, frame
                frame_idx += 1
        finally:
            cap.release()

        if frame_idx == 0:
            raise VideoProcessingError("Video contained 0 valid frames.")


class VideoWriter:
    """Writes annotated frames into an MP4 video file using OpenCV 5."""

    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps > 0 else 25.0
        self.width = width
        self.height = height

        # Attempt standard MP4 codecs ('mp4v' then 'avc1')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            logger.warning("mp4v codec failed, attempting avc1 codec")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self.writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (self.width, self.height)
            )

    def write_frame(self, frame: np.ndarray) -> None:
        """Writes a single BGR frame."""
        if self.writer and self.writer.isOpened():
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def release(self) -> None:
        """Closes the video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
