"""Deterministic, evidence-driven policy engine for dynamic tool selection."""

from typing import Any, Dict, List, Tuple

from app.vision.tracking import TrackedCandidate
from app.vision.zones import RestrictedZone


class AgentPolicy:
    """Selects the next OpenCV tool call dynamically based on accumulating visual evidence."""

    def __init__(
        self,
        persistence_threshold: int = 4,
        approach_threshold: float = 35.0,
    ):
        self.persistence_threshold = persistence_threshold
        self.approach_threshold = approach_threshold

    def select_next_tool(
        self,
        track: TrackedCandidate,
        zone: RestrictedZone,
        history_tool_results: List[Dict[str, Any]],
        step_index: int,
        max_steps: int = 4,
    ) -> Tuple[str, Dict[str, Any], str]:
        """Chooses the next tool to execute.

        Returns:
            (tool_name, tool_arguments, policy_condition_description)
        """
        if not track.observations:
            return (
                "dismiss_candidate",
                {"reason": "No observations available for track"},
                "EMPTY_OBSERVATIONS",
            )

        last_obs = track.observations[-1]
        bbox = last_obs.bbox
        centroid = last_obs.centroid
        signed_dist = last_obs.signed_dist
        is_inside = last_obs.inside_zone
        consecutive_inside = track.consecutive_inside_frames
        total_observed = track.total_observed_frames

        # Retrieve previous tools executed in this decision cycle
        executed_tools = [tr.get("tool_name") for tr in history_tool_results]

        # Case 1: Candidate is confirmed inside with sustained presence
        if is_inside and consecutive_inside >= self.persistence_threshold:
            evidence_score = min(1.0, 0.70 + 0.05 * consecutive_inside)
            return (
                "confirm_candidate",
                {
                    "evidence_score": round(evidence_score, 2),
                    "rationale": f"Candidate sustained inside restricted zone for {consecutive_inside} consecutive frames",
                },
                "SUSTAINED_ZONE_ENTRY",
            )

        # Case 2: Candidate turned away from the boundary
        if len(track.observations) >= 2:
            dist_delta = track.observations[-1].signed_dist - track.observations[-2].signed_dist
            if signed_dist < 0 and dist_delta < -1.8:
                return (
                    "dismiss_candidate",
                    {
                        "reason": f"Candidate turned away from boundary (retreat delta: {dist_delta:.1f}px)",
                    },
                    "CANDIDATE_TURNED_AWAY",
                )

        # Case 3: Far outside perimeter
        if signed_dist < -self.approach_threshold:
            return (
                "dismiss_candidate",
                {"reason": f"Candidate outside zone boundary ({abs(signed_dist):.1f}px)"},
                "PERIMETER_OUTSIDE",
            )

        # Step budget safeguard: If maximum steps reached without resolution, conclude as inconclusive
        if step_index >= max_steps:
            return (
                "mark_inconclusive",
                {
                    "reason": f"Max inspection steps ({max_steps}) reached without conclusive crossing evidence"
                },
                "MAX_BUDGET_INCONCLUSIVE",
            )

        # Case 4: Candidate is inside zone, but persistence condition needs temporal recheck
        if is_inside and "recheck_temporal_window" not in executed_tools:
            return (
                "recheck_temporal_window",
                {"min_consecutive_frames": self.persistence_threshold},
                "EVALUATE_TEMPORAL_PERSISTENCE",
            )

        # Check result of temporal recheck if already executed
        for tr in history_tool_results:
            if tr.get("tool_name") == "recheck_temporal_window":
                res = tr.get("result", {})
                if res.get("is_sustained_entry"):
                    return (
                        "confirm_candidate",
                        {
                            "evidence_score": 0.88,
                            "rationale": "Temporal recheck confirmed sustained presence inside zone",
                        },
                        "TEMPORAL_VERIFIED_ENTRY",
                    )

        # Case 5: Candidate is small or ambiguous (e.g. shadow or noise)
        bbox_area = bbox[2] * bbox[3]
        if bbox_area < 500 and "inspect_roi" not in executed_tools:
            return (
                "inspect_roi",
                {"bbox": bbox, "padding": 0.15},
                "AMBIGUOUS_CANDIDATE_INSPECT_ROI",
            )

        # Check result of ROI inspection
        for tr in history_tool_results:
            if tr.get("tool_name") == "inspect_roi":
                res = tr.get("result", {})
                if not res.get("is_structured_object", True):
                    # Lack of edge/structure indicates shadow or noise patch
                    return (
                        "dismiss_candidate",
                        {
                            "reason": f"Crop inspection revealed low edge density ({res.get('edge_density', 0):.3f}) consistent with shadow/noise",
                        },
                        "UNSTRUCTURED_NOISE_DISMISSED",
                    )

        # Case 6: Candidate near boundary requiring fine geometric verification
        if (
            abs(signed_dist) <= self.approach_threshold
            and "evaluate_zone_geometry" not in executed_tools
        ):
            return (
                "evaluate_zone_geometry",
                {"centroid": centroid, "bbox": bbox},
                "PROXIMITY_GEOMETRIC_CHECK",
            )

        # Case 7: Track continuity check if erratic
        if total_observed >= 4 and "verify_track_continuity" not in executed_tools:
            return (
                "verify_track_continuity",
                {"window_frames": min(8, total_observed)},
                "TRACK_CONTINUITY_EVALUATION",
            )

        # Fallback to secondary foreground solidity check
        if "extract_foreground" not in executed_tools:
            return (
                "extract_foreground",
                {"bbox": bbox},
                "FOREGROUND_SOLIDITY_RECHECK",
            )

        return (
            "mark_inconclusive",
            {"reason": "Contradictory visual observations prevent definitive categorization"},
            "DEFAULT_INCONCLUSIVE",
        )
