"""Unit tests for restricted-zone geometry, denormalization, and boundary crossing."""

import pytest

from app.api.schemas import ZonePolygon
from app.vision.zones import RestrictedZone


@pytest.fixture
def test_zone():
    # Square zone occupying center 50%: x from 0.25 to 0.75, y from 0.25 to 0.75
    config = ZonePolygon(
        name="Test Zone",
        points=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
    )
    # 400x400 frame -> pixel coords: [100, 100] to [300, 300]
    return RestrictedZone(config, frame_width=400, frame_height=400)


def test_point_strictly_inside(test_zone):
    # Point (200, 200) is the exact center
    inside, dist = test_zone.point_test((200, 200))
    assert inside is True
    assert dist > 0  # Inside has positive distance to edge (approx 100px)
    assert abs(dist - 100.0) < 2.0


def test_point_strictly_outside(test_zone):
    # Point (50, 200) is 50px to the left of x=100 boundary
    inside, dist = test_zone.point_test((50, 200))
    assert inside is False
    assert dist < 0  # Outside has negative signed distance
    assert abs(abs(dist) - 50.0) < 2.0


def test_bbox_overlap_ratio(test_zone):
    # Box entirely inside: (150, 150, 40, 40)
    overlap_inside = test_zone.bbox_overlap_ratio((150, 150, 40, 40))
    assert overlap_inside == pytest.approx(1.0, 0.05)

    # Box entirely outside: (10, 10, 30, 30)
    overlap_outside = test_zone.bbox_overlap_ratio((10, 10, 30, 30))
    assert overlap_outside == 0.0

    # Box straddling boundary: (80, 150, 40, 40) -> x from 80 to 120 (half inside x=100)
    overlap_straddle = test_zone.bbox_overlap_ratio((80, 150, 40, 40))
    assert 0.4 <= overlap_straddle <= 0.6


def test_trajectory_motion_evaluation(test_zone):
    # Moving from (50, 200) to (80, 200) -> approaching
    delta, disp, state = test_zone.evaluate_trajectory_motion((50, 200), (80, 200))
    assert state == "APPROACHING"
    assert delta > 0

    # Moving from (80, 200) to (120, 200) -> crossed inside
    delta, disp, state = test_zone.evaluate_trajectory_motion((80, 200), (120, 200))
    assert state == "CROSSED_INSIDE"

    # Moving from (80, 200) to (50, 200) -> receding
    delta, disp, state = test_zone.evaluate_trajectory_motion((80, 200), (50, 200))
    assert state == "RECEDING"
    assert delta < 0
