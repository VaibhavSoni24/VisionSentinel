"""Unit tests proving the agentic feedback loop, dynamic tool execution, and policy branching."""

import numpy as np
import pytest

import app.agent.tools  # noqa: F401
from app.agent.orchestrator import AgentOrchestrator
from app.agent.registry import registry
from app.api.schemas import EventOutcome, ZonePolygon
from app.vision.tracking import TrackedCandidate, TrackObservation
from app.vision.zones import RestrictedZone


@pytest.fixture
def test_setup():
    config = ZonePolygon(
        name="Security Zone",
        points=[(0.5, 0.2), (0.9, 0.2), (0.9, 0.8), (0.5, 0.8)],
    )
    zone = RestrictedZone(config, frame_width=640, frame_height=360)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    fg_mask = np.zeros((360, 640), dtype=np.uint8)
    return zone, frame, fg_mask


def test_tool_registry_validation():
    # Verify tools are registered
    tool_names = [t["name"] for t in registry.list_tools()]
    assert "inspect_roi" in tool_names
    assert "extract_foreground" in tool_names
    assert "evaluate_zone_geometry" in tool_names
    assert "recheck_temporal_window" in tool_names
    assert "confirm_candidate" in tool_names
    assert "dismiss_candidate" in tool_names
    assert "mark_inconclusive" in tool_names

    # Invalid arguments should raise error
    with pytest.raises(Exception):
        registry.execute("confirm_candidate", {"evidence_score": 1.5, "rationale": "short"}, {})


def test_agent_confirms_sustained_entry(test_setup):
    zone, frame, fg_mask = test_setup
    orchestrator = AgentOrchestrator(max_tool_budget=4)

    # Candidate inside zone with 6 consecutive inside frames (exceeds persistence threshold 4)
    track = TrackedCandidate(
        track_id=101,
        consecutive_inside_frames=6,
        state=EventOutcome.CANDIDATE,
    )
    # Centroid at (400, 200) inside zone
    track.observations.append(
        TrackObservation(
            frame_index=50,
            timestamp=2.0,
            bbox=(380, 170, 40, 60),
            centroid=(400, 200),
            signed_dist=50.0,
            inside_zone=True,
            bbox_overlap=1.0,
        )
    )

    final_state = orchestrator.evaluate_candidate(
        track=track,
        frame=frame,
        fg_mask=fg_mask,
        zone=zone,
        frame_idx=50,
        timestamp=2.0,
    )

    assert final_state == EventOutcome.CONFIRMED
    assert track.evidence_score >= 0.70
    assert len(orchestrator.trace.steps) >= 1
    # Check that confirm_candidate tool was chosen
    last_step = orchestrator.trace.steps[-1]
    assert last_step.chosen_tool == "confirm_candidate"
    assert last_step.human_review_required is True


def test_agent_dismisses_candidate_turning_away(test_setup):
    zone, frame, fg_mask = test_setup
    orchestrator = AgentOrchestrator(max_tool_budget=4)

    # Candidate approached boundary, then moved backwards
    track = TrackedCandidate(
        track_id=102,
        consecutive_inside_frames=0,
        state=EventOutcome.APPROACHING,
    )
    # Observation 1: near boundary at -10px
    track.observations.append(
        TrackObservation(
            frame_index=40,
            timestamp=1.6,
            bbox=(290, 170, 40, 60),
            centroid=(310, 200),
            signed_dist=-10.0,
            inside_zone=False,
            bbox_overlap=0.1,
        )
    )
    # Observation 2: retreating back to -25px (delta = -15px)
    track.observations.append(
        TrackObservation(
            frame_index=41,
            timestamp=1.64,
            bbox=(275, 170, 40, 60),
            centroid=(295, 200),
            signed_dist=-25.0,
            inside_zone=False,
            bbox_overlap=0.0,
        )
    )

    final_state = orchestrator.evaluate_candidate(
        track=track,
        frame=frame,
        fg_mask=fg_mask,
        zone=zone,
        frame_idx=41,
        timestamp=1.64,
    )

    assert final_state == EventOutcome.DISMISSED
    assert len(orchestrator.trace.steps) >= 1
    last_step = orchestrator.trace.steps[-1]
    assert last_step.chosen_tool == "dismiss_candidate"
    assert "turned away" in last_step.rationale.lower()


def test_agent_inspects_ambiguous_candidate_roi(test_setup):
    zone, frame, fg_mask = test_setup
    orchestrator = AgentOrchestrator(max_tool_budget=4)

    # Small candidate with tiny bbox (area < 500)
    track = TrackedCandidate(
        track_id=103,
        state=EventOutcome.CANDIDATE,
    )
    track.observations.append(
        TrackObservation(
            frame_index=20,
            timestamp=0.8,
            bbox=(300, 180, 15, 20),  # Area 300 (<500)
            centroid=(307, 190),
            signed_dist=-5.0,
            inside_zone=False,
            bbox_overlap=0.2,
        )
    )

    final_state = orchestrator.evaluate_candidate(
        track=track,
        frame=frame,
        fg_mask=fg_mask,
        zone=zone,
        frame_idx=20,
        timestamp=0.8,
    )
    assert final_state is not None

    # Verify that inspect_roi was called during the trace
    tools_invoked = [s.chosen_tool for s in orchestrator.trace.steps]
    assert "inspect_roi" in tools_invoked
