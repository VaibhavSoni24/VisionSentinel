"""Restricted-zone event state machine and incident recording."""

from typing import Dict, List, Optional, Tuple

from app.api.schemas import EventIncident, EventOutcome, TimelineEntry
from app.vision.tracking import TrackedCandidate


class EventPolicyEngine:
    """Evaluates track observations against explicit restricted-zone verification rules."""

    def __init__(
        self,
        persistence_frame_threshold: int = 5,
        approach_distance_threshold: float = 35.0,
    ):
        self.persistence_threshold = persistence_frame_threshold
        self.approach_threshold = approach_distance_threshold

    def calculate_evidence_score(self, track: TrackedCandidate) -> float:
        """Calculates a documented evidence score [0.0 - 1.0] from visual measurements.

        NOTE: This represents a weighted combination of measurements, NOT a calibrated probability.
        Weights:
        - Bounding box zone overlap (up to 0.35)
        - Spatial boundary proximity / crossing (up to 0.35)
        - Temporal persistence duration (up to 0.30)
        """
        if not track.observations:
            return 0.0

        last_obs = track.observations[-1]

        # 1. Overlap component
        overlap_weight = min(1.0, last_obs.bbox_overlap) * 0.35

        # 2. Signed distance component: positive (inside) yields full score, close yields partial
        if last_obs.signed_dist >= 0:
            dist_weight = 0.35
        else:
            dist_weight = max(
                0.0, 0.35 * (1.0 - abs(last_obs.signed_dist) / self.approach_threshold)
            )

        # 3. Persistence component
        persist_ratio = min(
            1.0, float(track.consecutive_inside_frames) / float(self.persistence_threshold)
        )
        persist_weight = persist_ratio * 0.30

        return round(float(overlap_weight + dist_weight + persist_weight), 3)

    def evaluate_transition(
        self,
        track: TrackedCandidate,
        dist_delta: float,
    ) -> Tuple[EventOutcome, str]:
        """Evaluates deterministic event state transitions based on track geometry and history."""
        if not track.observations:
            return EventOutcome.OUTSIDE, "No observations available"

        last_obs = track.observations[-1]
        signed_dist = last_obs.signed_dist
        is_inside = last_obs.inside_zone
        consecutive_inside = track.consecutive_inside_frames
        total_observed = track.total_observed_frames

        # Condition 1: Sustained confirmed entry
        if is_inside and consecutive_inside >= self.persistence_threshold:
            return (
                EventOutcome.CONFIRMED,
                f"Candidate sustained inside restricted zone for {consecutive_inside} frames",
            )

        # Condition 2: Inside zone but awaiting persistence threshold
        if is_inside or last_obs.bbox_overlap > 0.4:
            return (
                EventOutcome.CANDIDATE,
                f"Candidate entered zone boundary; evaluating temporal persistence ({consecutive_inside}/{self.persistence_threshold})",
            )

        # Condition 3: Candidate was close to boundary but is now moving away (turned away)
        if (
            signed_dist < 0
            and dist_delta < -2.0
            and track.state in (EventOutcome.APPROACHING, EventOutcome.NEEDS_RECHECK)
        ):
            return (
                EventOutcome.DISMISSED,
                f"Candidate receded from boundary (distance delta: {dist_delta:.1f}px)",
            )

        # Condition 4: Candidate close to boundary, ambiguous or small
        if -self.approach_threshold <= signed_dist < 0:
            if total_observed < 3:
                return (
                    EventOutcome.NEEDS_RECHECK,
                    "Candidate near boundary with short track history; secondary OpenCV inspection recommended",
                )
            if dist_delta > 1.0:
                return (
                    EventOutcome.APPROACHING,
                    f"Candidate approaching zone boundary at distance {abs(signed_dist):.1f}px",
                )

        # Condition 5: Candidate far outside
        if signed_dist < -self.approach_threshold:
            return (
                EventOutcome.OUTSIDE,
                f"Candidate outside zone perimeter ({abs(signed_dist):.1f}px from boundary)",
            )

        # Default fallback
        return track.state, "Maintaining current state"


class EventIncidentManager:
    """Manages active incidents and generates structured timeline entries."""

    def __init__(self, policy_engine: Optional[EventPolicyEngine] = None):
        self.policy = policy_engine or EventPolicyEngine()
        self.timeline: List[TimelineEntry] = []
        self.incidents: Dict[int, EventIncident] = {}

    def process_track(
        self,
        track: TrackedCandidate,
        frame_idx: int,
        timestamp: float,
    ) -> EventOutcome:
        """Evaluates track, updates evidence score, records state changes to timeline."""
        # Calculate distance delta from previous observation
        dist_delta = 0.0
        if len(track.observations) >= 2:
            dist_delta = track.observations[-1].signed_dist - track.observations[-2].signed_dist

        prev_state = track.state
        new_state, rationale = self.policy.evaluate_transition(track, dist_delta)
        evidence_score = self.policy.calculate_evidence_score(track)

        track.state = new_state
        track.evidence_score = evidence_score
        track.rationale = rationale

        # Record timeline event on state transition or confirmation
        if prev_state != new_state or new_state == EventOutcome.CONFIRMED:
            self.timeline.append(
                TimelineEntry(
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    event_type=f"STATE_{new_state.value}",
                    candidate_id=track.track_id,
                    state=new_state,
                    evidence_score=evidence_score,
                    message=rationale,
                )
            )

        # Update or record incident
        first_time = track.observations[0].timestamp if track.observations else timestamp
        zone_crossed = any(obs.inside_zone for obs in track.observations)
        human_review = new_state in (EventOutcome.CONFIRMED, EventOutcome.INCONCLUSIVE)

        self.incidents[track.track_id] = EventIncident(
            event_id=f"EVT-{track.track_id:04d}",
            candidate_id=track.track_id,
            category=track.category,
            final_state=new_state,
            evidence_score=evidence_score,
            first_detected_timestamp=first_time,
            final_timestamp=timestamp,
            frames_observed=track.total_observed_frames,
            zone_cross_detected=zone_crossed,
            persistence_seconds=round(
                track.consecutive_inside_frames * 0.04, 2
            ),  # approx at 25 fps
            human_review_required=human_review,
            rationale=rationale,
        )

        return new_state

    def record_track_state(
        self,
        track: TrackedCandidate,
        frame_idx: int,
        timestamp: float,
        prev_state: Optional[EventOutcome] = None,
    ) -> None:
        """Records or updates an incident directly from the track's current agent outcome."""
        first_time = track.observations[0].timestamp if track.observations else timestamp
        zone_crossed = any(obs.inside_zone for obs in track.observations)
        human_review = track.state in (EventOutcome.CONFIRMED, EventOutcome.INCONCLUSIVE)

        if prev_state != track.state or track.state == EventOutcome.CONFIRMED:
            self.timeline.append(
                TimelineEntry(
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    event_type=f"STATE_{track.state.value}",
                    candidate_id=track.track_id,
                    state=track.state,
                    evidence_score=track.evidence_score,
                    message=track.rationale,
                )
            )

        self.incidents[track.track_id] = EventIncident(
            event_id=f"EVT-{track.track_id:04d}",
            candidate_id=track.track_id,
            category=track.category,
            final_state=track.state,
            evidence_score=track.evidence_score,
            first_detected_timestamp=first_time,
            final_timestamp=timestamp,
            frames_observed=track.total_observed_frames,
            zone_cross_detected=zone_crossed,
            persistence_seconds=round(track.consecutive_inside_frames * 0.04, 2),
            human_review_required=human_review,
            rationale=track.rationale,
        )

    def get_incidents_list(self) -> List[EventIncident]:
        return list(self.incidents.values())

    def get_timeline(self) -> List[TimelineEntry]:
        return self.timeline
