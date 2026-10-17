"""Unit tests for multi-object tracking and association logic."""

import numpy as np
import pytest

from app.api.schemas import ZonePolygon
from app.vision.motion import MotionCandidate
from app.vision.tracking import CandidateTracker, compute_iou
from app.vision.zones import RestrictedZone


@pytest.fixture
def test_tracker():
    config = ZonePolygon(
        name="Track Zone",
        points=[(0.4, 0.2), (0.8, 0.2), (0.8, 0.8), (0.4, 0.8)],
    )
    zone = RestrictedZone(config, frame_width=500, frame_height=500)
    return CandidateTracker(zone, max_distance_threshold=60.0, max_missing_frames=5)


def test_compute_iou():
    box1 = (50, 50, 50, 50)
    box2 = (50, 50, 50, 50)
    assert compute_iou(box1, box2) == 1.0

    box3 = (200, 200, 50, 50)
    assert compute_iou(box1, box3) == 0.0

    box4 = (75, 50, 50, 50)  # 50% horizontal overlap
    iou = compute_iou(box1, box4)
    assert 0.3 <= iou <= 0.4


def test_tracker_association_and_lifecycle(test_tracker):
    # Frame 1: Detection at (100, 100)
    cnt1 = np.array([[[90, 80]], [[110, 80]], [[110, 120]], [[90, 120]]], dtype=np.int32)
    det1 = MotionCandidate(
        bbox=(90, 80, 20, 40),
        centroid=(100, 100),
        area=800.0,
        aspect_ratio=2.0,
        extent=1.0,
        contour=cnt1,
    )
    active = test_tracker.update(1, 0.04, [det1])
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].total_observed_frames == 1

    # Frame 2: Candidate moves to (110, 105) (small displacement, should match)
    cnt2 = np.array([[[100, 85]], [[120, 85]], [[120, 125]], [[100, 125]]], dtype=np.int32)
    det2 = MotionCandidate(
        bbox=(100, 85, 20, 40),
        centroid=(110, 105),
        area=800.0,
        aspect_ratio=2.0,
        extent=1.0,
        contour=cnt2,
    )
    active = test_tracker.update(2, 0.08, [det2])
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].total_observed_frames == 2
    assert active[0].current_centroid == (110, 105)

    # Frame 3-8: Missing candidate
    for f in range(3, 9):
        active = test_tracker.update(f, f * 0.04, [])

    # Exceeded max_missing_frames (5), should be purged from active tracks and moved to completed
    assert len(active) == 0
    assert len(test_tracker.completed_tracks) == 1
    assert test_tracker.completed_tracks[0].track_id == 1
