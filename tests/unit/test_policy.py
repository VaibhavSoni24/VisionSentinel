"""Unit tests for the deterministic agent policy engine."""

import pytest

from app.agent.policy import AgentPolicy
from app.api.schemas import ZonePolygon
from app.vision.tracking import TrackedCandidate, TrackObservation
from app.vision.zones import RestrictedZone


@pytest.fixture
def policy_fixture():
    policy = AgentPolicy(persistence_threshold=4, approach_threshold=30.0)
    config = ZonePolygon(points=[(0.4, 0.2), (0.8, 0.2), (0.8, 0.8), (0.4, 0.8)])
    zone = RestrictedZone(config, frame_width=500, frame_height=500)
    return policy, zone


def test_policy_empty_track_dismisses(policy_fixture):
    policy, zone = policy_fixture
    track = TrackedCandidate(track_id=1)
    tool_name, tool_args, condition = policy.select_next_tool(track, zone, [], step_index=1)
    assert tool_name == "dismiss_candidate"
    assert condition == "EMPTY_OBSERVATIONS"


def test_policy_max_step_budget_inconclusive(policy_fixture):
    policy, zone = policy_fixture
    track = TrackedCandidate(track_id=1)
    track.observations.append(
        TrackObservation(
            frame_index=10,
            timestamp=0.4,
            bbox=(200, 200, 40, 40),
            centroid=(220, 220),
            signed_dist=-10.0,
            inside_zone=False,
            bbox_overlap=0.1,
        )
    )
    # If step_index >= max_steps, should mark inconclusive
    tool_name, tool_args, condition = policy.select_next_tool(
        track, zone, [{"tool_name": "inspect_roi"}], step_index=5, max_steps=4
    )
    assert tool_name == "mark_inconclusive"
    assert condition == "MAX_BUDGET_INCONCLUSIVE"
