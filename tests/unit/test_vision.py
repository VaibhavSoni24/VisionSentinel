"""Unit tests for OpenCV 5 video I/O, motion extraction, and preprocessing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.api.schemas import ZonePolygon
from app.vision.annotation import render_annotated_frame
from app.vision.motion import MotionExtractor
from app.vision.preprocessing import (
    apply_gaussian_blur,
    extract_crop,
    normalize_illumination,
    resize_frame,
)
from app.vision.tracking import CandidateTracker
from app.vision.video_io import VideoReader
from app.vision.zones import RestrictedZone


def test_preprocessing_utilities():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)

    # 1. Resize
    resized = resize_frame(img, 50, 50)
    assert resized.shape == (50, 50, 3)

    # 2. Gaussian blur
    blurred = apply_gaussian_blur(img, kernel_size=5)
    assert blurred.shape == img.shape

    # 3. Illumination normalization
    norm = normalize_illumination(img)
    assert norm.shape == img.shape

    # 4. Crop extraction
    crop, pad_bbox = extract_crop(img, (40, 40, 20, 20), padding=0.2)
    assert crop.shape[0] > 20 and crop.shape[1] > 20
    assert pad_bbox[2] > 20


def test_motion_extraction_on_moving_square():
    extractor = MotionExtractor(history=10, var_threshold=16.0, min_contour_area=50)

    # Feed 5 identical black frames (background model learns black)
    for _ in range(5):
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        mask, candidates = extractor.extract(bg)
        assert len(candidates) == 0

    # Introduce bright moving square at (80, 80)
    moving_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(moving_frame, (80, 80), (120, 120), (255, 255, 255), -1)

    mask, candidates = extractor.extract(moving_frame)
    assert len(candidates) >= 1
    c = candidates[0]
    # Check centroid is approximately (100, 100)
    assert 90 <= c.centroid[0] <= 110
    assert 90 <= c.centroid[1] <= 110
    assert c.area >= 50


def test_video_reader_on_generated_demo():
    demo_video = Path("./data/generated/demo_synthetic.mp4")
    if not demo_video.exists():
        pytest.skip("Synthetic demo video not yet generated")

    reader = VideoReader(demo_video)
    assert reader.metadata.width == 640
    assert reader.metadata.height == 360
    assert reader.metadata.total_frames >= 300
    assert reader.metadata.fps == 25.0

    # Read first 10 frames
    count = 0
    for idx, ts, frame in reader.frames():
        assert frame.shape == (360, 640, 3)
        assert idx == count
        count += 1
        if count >= 10:
            break
    assert count == 10


def test_annotation_rendering():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    config = ZonePolygon(
        name="Security Zone",
        points=[(0.4, 0.2), (0.8, 0.2), (0.8, 0.8), (0.4, 0.8)],
    )
    zone = RestrictedZone(config, frame_width=640, frame_height=360)
    tracker = CandidateTracker(zone)

    annotated = render_annotated_frame(
        frame=frame,
        zone=zone,
        tracks=list(tracker.active_tracks.values()),
        frame_idx=1,
        timestamp=0.04,
        operating_mode="Mode B (Classical OpenCV Motion Candidate)",
    )
    assert annotated.shape == (360, 640, 3)
    assert np.any(annotated > 0)  # Has HUD and zone drawn
