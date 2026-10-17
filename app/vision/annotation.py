"""Visual annotation and video overlay rendering powered by OpenCV 5."""

from typing import List

import cv2
import numpy as np

from app.api.schemas import EventOutcome
from app.vision.tracking import TrackedCandidate
from app.vision.zones import RestrictedZone

# Harmonious, high-contrast BGR color mapping
STATE_COLORS = {
    EventOutcome.OUTSIDE: (220, 180, 60),  # Soft Cyan/Blue
    EventOutcome.APPROACHING: (0, 215, 255),  # Vibrant Amber/Yellow
    EventOutcome.CANDIDATE: (0, 140, 255),  # Deep Orange
    EventOutcome.NEEDS_RECHECK: (180, 105, 255),  # Violet
    EventOutcome.CONFIRMED: (45, 45, 235),  # High-Alert Red
    EventOutcome.DISMISSED: (90, 210, 90),  # Calm Emerald Green
    EventOutcome.INCONCLUSIVE: (180, 180, 180),  # Neutral Grey
}


def render_annotated_frame(
    frame: np.ndarray,
    zone: RestrictedZone,
    tracks: List[TrackedCandidate],
    frame_idx: int,
    timestamp: float,
    operating_mode: str,
) -> np.ndarray:
    """Renders visual verification HUD, zone polygon, tracks, and evidence scores."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Zone Polygon Rendering
    overlay = annotated.copy()
    has_confirmed = any(t.state == EventOutcome.CONFIRMED for t in tracks)
    zone_color = (30, 30, 220) if has_confirmed else (240, 160, 40)

    # Semi-transparent polygon fill
    cv2.fillPoly(overlay, [zone.polygon_pts], zone_color)
    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

    # Crisp polygon boundary
    cv2.polylines(annotated, [zone.polygon_pts], isClosed=True, color=zone_color, thickness=2)

    # Zone label badge
    if len(zone.polygon_pts) > 0:
        top_left_pt = zone.polygon_pts[0]
        cv2.putText(
            annotated,
            f"[{zone.name.upper()}]",
            (top_left_pt[0] + 8, top_left_pt[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            zone_color,
            2,
            cv2.LINE_AA,
        )

    # 2. Tracks & Candidate Bounding Boxes
    for track in tracks:
        if not track.observations:
            continue

        color = STATE_COLORS.get(track.state, (200, 200, 200))
        x, y, bw, bh = track.current_bbox

        # Draw trajectory history line
        if len(track.observations) >= 2:
            pts = [obs.centroid for obs in track.observations[-20:]]
            for i in range(1, len(pts)):
                alpha = float(i) / len(pts)
                thickness = 1 if alpha < 0.5 else 2
                cv2.line(annotated, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        # Candidate Info Tag
        tag = f"ID {track.track_id} | {track.state.value} ({track.evidence_score:.2f})"
        sub_tag = f"{track.category}"

        # Background badge for text legibility
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(
            annotated,
            (x, max(0, y - 32)),
            (x + max(tw, 130) + 8, max(0, y - 2)),
            (25, 25, 25),
            -1,
        )
        cv2.putText(
            annotated,
            tag,
            (x + 4, max(12, y - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            sub_tag,
            (x + 4, max(24, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    # 3. Top HUD Status Header
    cv2.rectangle(annotated, (0, 0), (w, 36), (18, 18, 22), -1)
    cv2.line(annotated, (0, 36), (w, 36), (60, 60, 70), 1)

    hud_title = f"VISION SENTINEL | OpenCV {cv2.__version__} | {operating_mode}"
    cv2.putText(
        annotated,
        hud_title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    hud_time = f"Frame {frame_idx:04d} | {timestamp:.2f}s | Tracks: {len(tracks)}"
    (hw, _), _ = cv2.getTextSize(hud_time, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        annotated,
        hud_time,
        (w - hw - 14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # High-visibility Alert banner if intrusion confirmed
    if has_confirmed:
        alert_box_w = 210
        cv2.rectangle(
            annotated,
            (w // 2 - alert_box_w // 2, 42),
            (w // 2 + alert_box_w // 2, 70),
            (30, 30, 210),
            -1,
        )
        cv2.putText(
            annotated,
            "RESTRICTED ZONE ENTRY",
            (w // 2 - alert_box_w // 2 + 10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated
