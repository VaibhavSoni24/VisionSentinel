"""OpenCV 5-backed computer vision and decision tools registered to the Agent."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.agent.registry import registry
from app.api.schemas import EventOutcome
from app.vision.preprocessing import extract_crop


# ---------------------------------------------------------
# Tool 1: inspect_roi
# ---------------------------------------------------------
class InspectRoiArgs(BaseModel):
    bbox: Tuple[int, int, int, int] = Field(description="(x, y, w, h) bounding box")
    padding: float = Field(default=0.15, ge=0.0, le=0.5)


@registry.register(
    name="inspect_roi",
    description="Extracts a cropped image patch around candidate bounding box and analyzes edge density and contrast using OpenCV 5.",
    args_schema=InspectRoiArgs,
)
def inspect_roi(args: InspectRoiArgs, context: Dict[str, Any]) -> Dict[str, Any]:
    frame: np.ndarray = context.get("current_frame")
    if frame is None:
        return {"error": "No frame present in execution context"}

    crop, padded_bbox = extract_crop(frame, args.bbox, padding=args.padding)
    if crop.size == 0:
        return {"is_valid": False, "reason": "Empty crop extracted"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_val, std_val = cv2.meanStdDev(gray)

    # Edge analysis with Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(cv2.countNonZero(edges)) / float(crop.shape[0] * crop.shape[1])

    # Local Laplacian variance for sharpness/structure
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_structured = edge_density > 0.04 and laplacian_var > 60.0

    return {
        "is_valid": True,
        "crop_dimensions": [crop.shape[1], crop.shape[0]],
        "padded_bbox": list(padded_bbox),
        "mean_intensity": round(float(mean_val[0][0]), 2),
        "contrast_std": round(float(std_val[0][0]), 2),
        "edge_density": round(edge_density, 4),
        "sharpness_laplacian": round(laplacian_var, 2),
        "is_structured_object": is_structured,
    }


# ---------------------------------------------------------
# Tool 2: extract_foreground
# ---------------------------------------------------------
class ExtractForegroundArgs(BaseModel):
    bbox: Tuple[int, int, int, int] = Field(description="(x, y, w, h) bounding box")


@registry.register(
    name="extract_foreground",
    description="Analyzes foreground mask solidity and contour compactness for the candidate region using OpenCV 5.",
    args_schema=ExtractForegroundArgs,
)
def extract_foreground(args: ExtractForegroundArgs, context: Dict[str, Any]) -> Dict[str, Any]:
    fg_mask: np.ndarray = context.get("fg_mask")
    if fg_mask is None:
        return {"error": "No foreground mask present in execution context"}

    x, y, w, h = args.bbox
    h_img, w_img = fg_mask.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x1 >= x2 or y1 >= y2:
        return {"error": "Bounding box coordinates out of image bounds"}

    patch = fg_mask[y1:y2, x1:x2]
    fg_pixels = cv2.countNonZero(patch)
    total_pixels = float((x2 - x1) * (y2 - y1))
    solidity = fg_pixels / total_pixels if total_pixels > 0 else 0.0

    # Contour compactness
    contours, _ = cv2.findContours(patch.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness = 0.0
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        perimeter = cv2.arcLength(largest_cnt, True)
        if perimeter > 0:
            compactness = 4.0 * np.pi * area / (perimeter * perimeter)

    is_solid = solidity > 0.25 and compactness > 0.15
    return {
        "foreground_pixel_count": int(fg_pixels),
        "foreground_solidity": round(float(solidity), 3),
        "compactness": round(float(compactness), 3),
        "is_solid_foreground": is_solid,
    }


# ---------------------------------------------------------
# Tool 3: evaluate_zone_geometry
# ---------------------------------------------------------
class EvaluateZoneGeometryArgs(BaseModel):
    centroid: Tuple[int, int] = Field(description="(cx, cy) candidate centroid")
    bbox: Tuple[int, int, int, int] = Field(description="(x, y, w, h) bounding box")


@registry.register(
    name="evaluate_zone_geometry",
    description="Computes precise point-to-polygon distance, containment, and bounding box overlap ratio with restricted zone.",
    args_schema=EvaluateZoneGeometryArgs,
)
def evaluate_zone_geometry(
    args: EvaluateZoneGeometryArgs, context: Dict[str, Any]
) -> Dict[str, Any]:
    zone = context.get("zone")
    if zone is None:
        return {"error": "No restricted zone present in execution context"}

    is_inside, signed_dist = zone.point_test(args.centroid)
    overlap_ratio = zone.bbox_overlap_ratio(args.bbox)

    return {
        "centroid": list(args.centroid),
        "is_centroid_inside": is_inside,
        "signed_distance_pixels": round(float(signed_dist), 2),
        "bbox_overlap_ratio": round(float(overlap_ratio), 3),
        "crosses_perimeter": is_inside or overlap_ratio > 0.35,
    }


# ---------------------------------------------------------
# Tool 4: verify_track_continuity
# ---------------------------------------------------------
class VerifyTrackContinuityArgs(BaseModel):
    window_frames: int = Field(default=8, ge=2, le=30)


@registry.register(
    name="verify_track_continuity",
    description="Evaluates trajectory smoothness, displacement variance, and missing frame ratio over historical observations.",
    args_schema=VerifyTrackContinuityArgs,
)
def verify_track_continuity(
    args: VerifyTrackContinuityArgs, context: Dict[str, Any]
) -> Dict[str, Any]:
    track = context.get("track")
    if track is None or not track.observations:
        return {"error": "No track history available in execution context"}

    observations = track.observations[-args.window_frames :]
    if len(observations) < 2:
        return {
            "observations_count": len(observations),
            "is_continuous": False,
            "reason": "Insufficient observations in window",
        }

    displacements = []
    for i in range(1, len(observations)):
        p1 = observations[i - 1].centroid
        p2 = observations[i].centroid
        d = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        displacements.append(float(d))

    mean_disp = float(np.mean(displacements))
    std_disp = float(np.std(displacements))
    is_smooth = std_disp < 25.0 and mean_disp < 80.0

    return {
        "observations_count": len(observations),
        "mean_displacement_px": round(mean_disp, 2),
        "std_displacement_px": round(std_disp, 2),
        "is_smooth_trajectory": is_smooth,
        "missing_frames_count": track.missing_frames,
    }


# ---------------------------------------------------------
# Tool 5: recheck_temporal_window
# ---------------------------------------------------------
class RecheckTemporalWindowArgs(BaseModel):
    min_consecutive_frames: int = Field(default=4, ge=1, le=20)


@registry.register(
    name="recheck_temporal_window",
    description="Evaluates consecutive inside-zone observations against persistence requirements to rule out momentary noise.",
    args_schema=RecheckTemporalWindowArgs,
)
def recheck_temporal_window(
    args: RecheckTemporalWindowArgs, context: Dict[str, Any]
) -> Dict[str, Any]:
    track = context.get("track")
    if track is None:
        return {"error": "No track present in execution context"}

    consecutive_inside = track.consecutive_inside_frames
    has_sustained = consecutive_inside >= args.min_consecutive_frames

    return {
        "consecutive_inside_frames": consecutive_inside,
        "threshold_required": args.min_consecutive_frames,
        "is_sustained_entry": has_sustained,
        "total_observed_frames": track.total_observed_frames,
    }


# ---------------------------------------------------------
# Tool 6: confirm_candidate
# ---------------------------------------------------------
class ConfirmCandidateArgs(BaseModel):
    evidence_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=5)


@registry.register(
    name="confirm_candidate",
    description="Finalizes agent decision by confirming restricted-zone entry and flagging for human review.",
    args_schema=ConfirmCandidateArgs,
)
def confirm_candidate(args: ConfirmCandidateArgs, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": EventOutcome.CONFIRMED.value,
        "evidence_score": args.evidence_score,
        "rationale": args.rationale,
        "human_review_required": True,
    }


# ---------------------------------------------------------
# Tool 7: dismiss_candidate
# ---------------------------------------------------------
class DismissCandidateArgs(BaseModel):
    reason: str = Field(min_length=5)


@registry.register(
    name="dismiss_candidate",
    description="Finalizes agent decision by dismissing candidate (e.g. turned away, noisy patch, or moving outside).",
    args_schema=DismissCandidateArgs,
)
def dismiss_candidate(args: DismissCandidateArgs, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": EventOutcome.DISMISSED.value,
        "rationale": args.reason,
        "human_review_required": False,
    }


# ---------------------------------------------------------
# Tool 8: mark_inconclusive
# ---------------------------------------------------------
class MarkInconclusiveArgs(BaseModel):
    reason: str = Field(min_length=5)


@registry.register(
    name="mark_inconclusive",
    description="Records an inconclusive event outcome when visual evidence is contradictory or lost near threshold.",
    args_schema=MarkInconclusiveArgs,
)
def mark_inconclusive(args: MarkInconclusiveArgs, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": EventOutcome.INCONCLUSIVE.value,
        "rationale": args.reason,
        "human_review_required": True,
    }
