"""Multi-object candidate tracking and temporal association using OpenCV 5."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from app.api.schemas import CandidateCategory, EventOutcome
from app.vision.motion import MotionCandidate
from app.vision.zones import RestrictedZone


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return float(inter_area) / float(union_area) if union_area > 0 else 0.0


@dataclass
class TrackObservation:
    frame_index: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    signed_dist: float
    inside_zone: bool
    bbox_overlap: float


@dataclass
class TrackedCandidate:
    track_id: int
    category: str = CandidateCategory.MOVING_OBJECT_CANDIDATE.value
    observations: List[TrackObservation] = field(default_factory=list)
    state: EventOutcome = EventOutcome.OUTSIDE
    evidence_score: float = 0.0
    consecutive_inside_frames: int = 0
    missing_frames: int = 0
    is_active: bool = True
    rationale: str = "Initial candidate detection"

    @property
    def current_bbox(self) -> Tuple[int, int, int, int]:
        return self.observations[-1].bbox if self.observations else (0, 0, 0, 0)

    @property
    def current_centroid(self) -> Tuple[int, int]:
        return self.observations[-1].centroid if self.observations else (0, 0)

    @property
    def total_observed_frames(self) -> int:
        return len(self.observations)

    @property
    def duration_seconds(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        return self.observations[-1].timestamp - self.observations[0].timestamp


class CandidateTracker:
    """Associates detections across frames into continuous object tracks."""

    def __init__(
        self,
        zone: RestrictedZone,
        max_distance_threshold: float = 75.0,
        max_missing_frames: int = 15,
    ):
        self.zone = zone
        self.max_distance_threshold = max_distance_threshold
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedCandidate] = {}
        self.completed_tracks: List[TrackedCandidate] = []

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        detections: List[MotionCandidate],
    ) -> List[TrackedCandidate]:
        """Associates incoming detections to active tracks and manages lifecycle."""
        unmatched_track_ids = set(self.active_tracks.keys())
        unmatched_detections = set(range(len(detections)))

        # Build distance / association cost matrix
        if self.active_tracks and detections:
            active_ids = list(self.active_tracks.keys())
            cost_matrix = []

            for tid in active_ids:
                track = self.active_tracks[tid]
                row = []
                for det in detections:
                    dist = np.hypot(
                        track.current_centroid[0] - det.centroid[0],
                        track.current_centroid[1] - det.centroid[1],
                    )
                    iou = compute_iou(track.current_bbox, det.bbox)
                    # Combined metric: distance penalized by IoU bonus
                    effective_cost = dist * (1.0 - 0.5 * iou)
                    row.append(effective_cost)
                cost_matrix.append(row)

            # Greedy Hungarian-style matching
            matched_pairs = []
            while True:
                min_cost = float("inf")
                min_i, min_j = -1, -1
                for i in range(len(active_ids)):
                    if active_ids[i] not in unmatched_track_ids:
                        continue
                    for j in range(len(detections)):
                        if j not in unmatched_detections:
                            continue
                        if cost_matrix[i][j] < min_cost:
                            min_cost = cost_matrix[i][j]
                            min_i, min_j = i, j

                if min_i != -1 and min_cost <= self.max_distance_threshold:
                    matched_tid = active_ids[min_i]
                    matched_pairs.append((matched_tid, min_j))
                    unmatched_track_ids.discard(matched_tid)
                    unmatched_detections.discard(min_j)
                else:
                    break

            # Update matched tracks
            for tid, det_idx in matched_pairs:
                det = detections[det_idx]
                self._update_track(self.active_tracks[tid], frame_idx, timestamp, det)

        # Handle unmatched active tracks (missing observation)
        for tid in list(unmatched_track_ids):
            track = self.active_tracks[tid]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                track.is_active = False
                self.completed_tracks.append(track)
                del self.active_tracks[tid]

        # Handle new unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = TrackedCandidate(track_id=self.next_track_id)
            self.next_track_id += 1
            self._update_track(new_track, frame_idx, timestamp, det)
            self.active_tracks[new_track.track_id] = new_track

        return list(self.active_tracks.values())

    def _update_track(
        self,
        track: TrackedCandidate,
        frame_idx: int,
        timestamp: float,
        detection: MotionCandidate,
    ) -> None:
        """Appends an observation to a track and computes zone metrics."""
        is_inside, signed_dist = self.zone.point_test(detection.centroid)
        bbox_overlap = self.zone.bbox_overlap_ratio(detection.bbox)

        obs = TrackObservation(
            frame_index=frame_idx,
            timestamp=timestamp,
            bbox=detection.bbox,
            centroid=detection.centroid,
            signed_dist=signed_dist,
            inside_zone=is_inside,
            bbox_overlap=bbox_overlap,
        )
        track.observations.append(obs)
        track.missing_frames = 0

        if is_inside or bbox_overlap > 0.4:
            track.consecutive_inside_frames += 1
        else:
            track.consecutive_inside_frames = 0

    def finalize(self) -> List[TrackedCandidate]:
        """Flushes all remaining active tracks into completed list."""
        for track in self.active_tracks.values():
            track.is_active = False
            self.completed_tracks.append(track)
        self.active_tracks.clear()
        return self.completed_tracks
