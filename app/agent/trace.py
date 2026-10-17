"""Agent execution trace recording and JSONL audit logging."""

from pathlib import Path
from typing import Any, Dict, List

from app.api.schemas import TraceStep


class AgentExecutionTrace:
    """Collects and serializes multi-step agent decisions into structured audit traces."""

    def __init__(self):
        self.steps: List[TraceStep] = []

    def record_step(
        self,
        step_index: int,
        candidate_id: int,
        frame_index: int,
        timestamp: float,
        event_state: str,
        policy_condition: str,
        chosen_tool: str,
        tool_arguments: Dict[str, Any],
        tool_result: Dict[str, Any],
        updated_evidence_score: float,
        new_state: str,
        human_review_required: bool,
        rationale: str,
    ) -> TraceStep:
        """Appends an immutable audit step to the trace."""
        step = TraceStep(
            step_index=step_index,
            candidate_id=candidate_id,
            frame_index=frame_index,
            timestamp=round(timestamp, 2),
            event_state=event_state,
            policy_condition=policy_condition,
            chosen_tool=chosen_tool,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            updated_evidence_score=round(updated_evidence_score, 3),
            new_state=new_state,
            human_review_required=human_review_required,
            rationale=rationale,
        )
        self.steps.append(step)
        return step

    def to_jsonl(self, file_path: Path) -> None:
        """Writes trace steps to a JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for step in self.steps:
                f.write(step.model_dump_json() + "\n")

    def to_list(self) -> List[TraceStep]:
        return self.steps
