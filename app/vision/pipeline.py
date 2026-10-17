"""End-to-end restricted-zone entry verification pipeline powered by OpenCV 5."""

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2

from app.agent.orchestrator import AgentOrchestrator
from app.api.schemas import (
    AnalysisJobStatus,
    EventOutcome,
    JobResult,
    ZonePolygon,
)
from app.config import settings
from app.storage.local import storage
from app.vision.annotation import render_annotated_frame
from app.vision.detector import local_detector
from app.vision.events import EventIncidentManager
from app.vision.motion import MotionExtractor
from app.vision.tracking import CandidateTracker
from app.vision.video_io import VideoReader, VideoWriter
from app.vision.zones import RestrictedZone

logger = logging.getLogger(__name__)


def run_verification_pipeline(
    job_id: str,
    video_path: Path,
    zone_config: ZonePolygon,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    send_user_media: bool = False,
) -> JobResult:
    """Executes full OpenCV 5 fixed-camera inspection, candidate tracking, and agent verification."""
    output_dir = storage.get_job_output_dir(job_id)
    annotated_video_path = output_dir / "annotated_result.mp4"
    incidents_json_path = output_dir / "incidents.json"
    timeline_csv_path = output_dir / "timeline.csv"
    trace_jsonl_path = output_dir / "agent_trace.jsonl"
    report_json_path = output_dir / "job_report.json"

    logger.info("Starting analysis job %s on %s", job_id, video_path)
    if progress_callback:
        progress_callback(0.05, "Initializing video reader and probing streams...")

    # 1. Probing video
    reader = VideoReader(video_path)
    meta = reader.metadata
    w, h, fps = meta.width, meta.height, meta.fps

    # 2. Geometric zone
    zone = RestrictedZone(zone_config, frame_width=w, frame_height=h)

    # 3. Computer vision modules
    motion_extractor = MotionExtractor(history=300, var_threshold=16.0, min_contour_area=180)
    tracker = CandidateTracker(zone, max_distance_threshold=65.0, max_missing_frames=12)
    incident_mgr = EventIncidentManager()
    agent = AgentOrchestrator(max_tool_budget=4)
    writer = VideoWriter(annotated_video_path, fps=fps, width=w, height=h)

    operating_mode = settings.get_operating_mode()
    total_frames = max(1, meta.total_frames)

    # 4. Sequential frame processing loop
    try:
        for frame_idx, timestamp, frame in reader.frames():
            # Extract motion candidates
            fg_mask, detections = motion_extractor.extract(frame)

            # Local semantic detector inference if Mode A is active
            if local_detector.is_available:
                for det in detections:
                    # Run detector on candidate crop
                    x, y, bw, bh = det.bbox
                    crop = frame[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
                    if crop.size > 0:
                        sem_dets = local_detector.detect_person(crop)
                        if sem_dets:
                            # Update label to person
                            pass

            # Track candidates
            active_tracks = tracker.update(frame_idx, timestamp, detections)

            # Dynamic agent evaluation for active candidates
            for track in active_tracks:
                prev_state = track.state
                last_obs = track.observations[-1]
                # If candidate is near boundary, inside, or actively being tracked
                if (
                    abs(last_obs.signed_dist) <= 35.0
                    or last_obs.inside_zone
                    or track.state != EventOutcome.OUTSIDE
                ):
                    agent.evaluate_candidate(
                        track=track,
                        frame=frame,
                        fg_mask=fg_mask,
                        zone=zone,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    )
                    incident_mgr.record_track_state(track, frame_idx, timestamp, prev_state)
                else:
                    incident_mgr.process_track(track, frame_idx, timestamp)

            # Render visual HUD overlay
            annotated = render_annotated_frame(
                frame=frame,
                zone=zone,
                tracks=active_tracks,
                frame_idx=frame_idx,
                timestamp=timestamp,
                operating_mode=operating_mode,
            )
            writer.write_frame(annotated)

            if progress_callback and frame_idx % 25 == 0:
                prog = min(0.95, float(frame_idx) / total_frames)
                progress_callback(prog, f"Processing frame {frame_idx}/{total_frames}...")

    finally:
        writer.release()
        completed_tracks = tracker.finalize()
        for tr in completed_tracks:
            last_obs = tr.observations[-1] if tr.observations else None
            last_idx = last_obs.frame_index if last_obs else 0
            last_ts = last_obs.timestamp if last_obs else 0.0
            incident_mgr.record_track_state(tr, last_idx, last_ts)

    if progress_callback:
        progress_callback(0.96, "Generating structured reports and event timelines...")

    # 5. Extract final incidents and timeline
    incidents = incident_mgr.get_incidents_list()
    timeline = incident_mgr.get_timeline()
    trace_steps = agent.trace.to_list()

    # 6. Save persistent artifacts
    storage.save_json(incidents_json_path, [inc.model_dump() for inc in incidents])
    agent.trace.to_jsonl(trace_jsonl_path)

    # Save CSV timeline
    csv_headers = [
        "timestamp",
        "frame_index",
        "event_type",
        "candidate_id",
        "state",
        "evidence_score",
        "message",
    ]
    csv_rows = [
        [
            t.timestamp,
            t.frame_index,
            t.event_type,
            t.candidate_id,
            t.state.value,
            t.evidence_score,
            t.message,
        ]
        for t in timeline
    ]
    storage.save_csv(timeline_csv_path, csv_headers, csv_rows)

    artifacts = {
        "annotated_video": str(annotated_video_path.name),
        "incidents_json": str(incidents_json_path.name),
        "timeline_csv": str(timeline_csv_path.name),
        "agent_trace_jsonl": str(trace_jsonl_path.name),
    }

    result = JobResult(
        job_id=job_id,
        status=AnalysisJobStatus.COMPLETED,
        operating_mode=operating_mode,
        opencv_version=cv2.__version__,
        video_metadata={
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "duration_seconds": meta.duration_seconds,
            "processed_frames": frame_idx + 1 if "frame_idx" in locals() else 0,
        },
        zone_polygon=zone_config,
        incidents=incidents,
        timeline=timeline,
        trace_steps=trace_steps,
        artifacts=artifacts,
    )

    storage.save_json(report_json_path, result.model_dump())
    if progress_callback:
        progress_callback(1.0, "Analysis complete.")

    logger.info(
        "Job %s completed successfully with %d incidents and %d trace steps.",
        job_id,
        len(incidents),
        len(trace_steps),
    )
    return result
