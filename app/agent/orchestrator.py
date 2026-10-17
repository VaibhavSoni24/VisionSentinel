"""Evidence-driven agent orchestrator executing dynamic OpenCV inspection loops."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

import app.agent.tools  # noqa: F401
from app.agent.policy import AgentPolicy
from app.agent.registry import registry
from app.agent.trace import AgentExecutionTrace
from app.api.schemas import EventOutcome
from app.vision.tracking import TrackedCandidate
from app.vision.zones import RestrictedZone

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates dynamic, multi-step inspection for ambiguous or critical candidates."""

    def __init__(
        self,
        policy: Optional[AgentPolicy] = None,
        max_tool_budget: int = 5,
    ):
        self.policy = policy or AgentPolicy()
        self.max_tool_budget = max_tool_budget
        self.trace = AgentExecutionTrace()

    def evaluate_candidate(
        self,
        track: TrackedCandidate,
        frame: np.ndarray,
        fg_mask: np.ndarray,
        zone: RestrictedZone,
        frame_idx: int,
        timestamp: float,
    ) -> EventOutcome:
        """Executes dynamic inspection loop on candidate until terminal state or budget reached."""
        tool_results_history: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {
            "current_frame": frame,
            "fg_mask": fg_mask,
            "zone": zone,
            "track": track,
        }

        current_state = track.state
        evidence_score = track.evidence_score
        step_idx = 1

        while step_idx <= self.max_tool_budget:
            # 1. Dynamic policy selection
            tool_name, tool_args, condition = self.policy.select_next_tool(
                track=track,
                zone=zone,
                history_tool_results=tool_results_history,
                step_index=step_idx,
                max_steps=self.max_tool_budget,
            )

            # 2. Execute OpenCV tool via registry
            try:
                tool_output = registry.execute(tool_name, tool_args, context=context)
            except Exception as e:
                logger.error("Agent failed to execute tool '%s': %s", tool_name, e)
                tool_output = {"error": str(e)}

            tool_results_history.append(
                {"tool_name": tool_name, "args": tool_args, "result": tool_output}
            )

            # 3. Check for terminal decision tools
            if tool_name == "confirm_candidate":
                current_state = EventOutcome.CONFIRMED
                evidence_score = float(tool_output.get("evidence_score", 0.90))
                rationale = tool_output.get("rationale", "Entry confirmed by agent inspection")
                self.trace.record_step(
                    step_index=step_idx,
                    candidate_id=track.track_id,
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    event_state=track.state.value,
                    policy_condition=condition,
                    chosen_tool=tool_name,
                    tool_arguments=tool_args,
                    tool_result=tool_output,
                    updated_evidence_score=evidence_score,
                    new_state=current_state.value,
                    human_review_required=True,
                    rationale=rationale,
                )
                break

            elif tool_name == "dismiss_candidate":
                current_state = EventOutcome.DISMISSED
                evidence_score = 0.10
                rationale = tool_output.get("rationale") or tool_args.get(
                    "reason", "Candidate dismissed by agent inspection"
                )
                self.trace.record_step(
                    step_index=step_idx,
                    candidate_id=track.track_id,
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    event_state=track.state.value,
                    policy_condition=condition,
                    chosen_tool=tool_name,
                    tool_arguments=tool_args,
                    tool_result=tool_output,
                    updated_evidence_score=evidence_score,
                    new_state=current_state.value,
                    human_review_required=False,
                    rationale=rationale,
                )
                break

            elif tool_name == "mark_inconclusive":
                current_state = EventOutcome.INCONCLUSIVE
                evidence_score = 0.50
                rationale = tool_output.get("rationale", "Inconclusive outcome")
                self.trace.record_step(
                    step_index=step_idx,
                    candidate_id=track.track_id,
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    event_state=track.state.value,
                    policy_condition=condition,
                    chosen_tool=tool_name,
                    tool_arguments=tool_args,
                    tool_result=tool_output,
                    updated_evidence_score=evidence_score,
                    new_state=current_state.value,
                    human_review_required=True,
                    rationale=rationale,
                )
                break

            # 4. Intermediate tool (e.g. inspect_roi, evaluate_zone_geometry, recheck_temporal_window)
            # Update evidence based on tool output
            if tool_name == "evaluate_zone_geometry":
                if tool_output.get("crosses_perimeter"):
                    evidence_score = min(1.0, evidence_score + 0.25)
                    current_state = EventOutcome.CANDIDATE
                else:
                    evidence_score = max(0.0, evidence_score - 0.15)

            elif tool_name == "inspect_roi":
                if tool_output.get("is_structured_object"):
                    evidence_score = min(1.0, evidence_score + 0.15)
                else:
                    evidence_score = max(0.0, evidence_score - 0.30)

            elif tool_name == "recheck_temporal_window":
                if tool_output.get("is_sustained_entry"):
                    evidence_score = min(1.0, evidence_score + 0.30)
                    current_state = EventOutcome.CONFIRMED

            self.trace.record_step(
                step_index=step_idx,
                candidate_id=track.track_id,
                frame_index=frame_idx,
                timestamp=timestamp,
                event_state=track.state.value,
                policy_condition=condition,
                chosen_tool=tool_name,
                tool_arguments=tool_args,
                tool_result=tool_output,
                updated_evidence_score=evidence_score,
                new_state=current_state.value,
                human_review_required=(current_state == EventOutcome.CONFIRMED),
                rationale=f"Executed {tool_name}; updated evidence score to {evidence_score:.2f}",
            )

            step_idx += 1

        track.state = current_state
        track.evidence_score = evidence_score
        return current_state
