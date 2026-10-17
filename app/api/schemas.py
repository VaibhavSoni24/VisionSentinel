"""Pydantic schemas and data contracts for Vision Sentinel API."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class EventOutcome(str, Enum):
    OUTSIDE = "OUTSIDE"
    APPROACHING = "APPROACHING"
    CANDIDATE = "CANDIDATE"
    NEEDS_RECHECK = "NEEDS_RECHECK"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CandidateCategory(str, Enum):
    PERSON = "person"
    MOVING_OBJECT_CANDIDATE = "moving_object_candidate"
    AMBIGUOUS_NOISE = "ambiguous_noise"


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0, description="Normalized x coordinate [0.0, 1.0]")
    y: float = Field(ge=0.0, le=1.0, description="Normalized y coordinate [0.0, 1.0]")


class ZonePolygon(BaseModel):
    """Restricted zone polygon defined in normalized coordinates [0.0, 1.0]."""

    name: str = Field(default="Restricted Zone 1")
    points: List[Tuple[float, float]] = Field(
        default=[(0.45, 0.20), (0.85, 0.20), (0.85, 0.80), (0.45, 0.80)],
        min_length=3,
        description="List of [x, y] vertices normalized between 0.0 and 1.0",
    )

    @field_validator("points")
    @classmethod
    def validate_points(cls, v: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        for pt in v:
            if not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                raise ValueError(f"Polygon point {pt} is outside normalized range [0.0, 1.0]")
        return v


class AnalysisJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TimelineEntry(BaseModel):
    timestamp: float
    frame_index: int
    event_type: str
    candidate_id: int
    state: EventOutcome
    evidence_score: float
    message: str


class TraceStep(BaseModel):
    step_index: int
    candidate_id: int
    frame_index: int
    timestamp: float
    event_state: str
    policy_condition: str
    chosen_tool: str
    tool_arguments: Dict[str, Any]
    tool_result: Dict[str, Any]
    updated_evidence_score: float
    new_state: str
    human_review_required: bool
    rationale: str


class EventIncident(BaseModel):
    event_id: str
    candidate_id: int
    category: str
    final_state: EventOutcome
    evidence_score: float
    first_detected_timestamp: float
    final_timestamp: float
    frames_observed: int
    zone_cross_detected: bool
    persistence_seconds: float
    human_review_required: bool
    rationale: str


class JobResult(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    operating_mode: str
    opencv_version: str
    video_metadata: Dict[str, Any]
    zone_polygon: ZonePolygon
    incidents: List[EventIncident]
    timeline: List[TimelineEntry]
    trace_steps: List[TraceStep]
    artifacts: Dict[str, str]
    error: Optional[str] = None
