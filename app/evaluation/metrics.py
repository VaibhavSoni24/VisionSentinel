"""Evaluation metrics calculation comparing pipeline results against ground truth."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvaluationMetrics:
    total_ground_truth_events: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    agent_task_success_rate: float
    mean_detection_delay_frames: float
    evaluated_scenarios_count: int


def evaluate_against_ground_truth(
    job_result: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> EvaluationMetrics:
    """Computes event-level verification precision, recall, and agent decision accuracy."""
    scenarios = ground_truth.get("scenarios", [])
    incidents = job_result.get("incidents", [])

    # Map candidate IDs to detected incident
    inc_map = {inc["candidate_id"]: inc for inc in incidents}

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    delays = []
    agent_successes = 0

    for sc in scenarios:
        cid = sc["candidate_id"]
        exp_outcome = sc["expected_outcome"]  # "CONFIRMED" or "DISMISSED"
        detected_inc = inc_map.get(cid)

        if not detected_inc:
            if exp_outcome == "CONFIRMED":
                fn += 1
            else:
                tn += 1
            continue

        actual_state = detected_inc.get("final_state")

        # Correct decision check:
        # If expected is CONFIRMED, actual must be CONFIRMED.
        # If expected is DISMISSED, actual can be DISMISSED or OUTSIDE (both represent non-intrusion).
        # If expected is INCONCLUSIVE, actual can be INCONCLUSIVE or DISMISSED.
        is_match = (
            (actual_state == exp_outcome)
            or (exp_outcome == "DISMISSED" and actual_state in ("DISMISSED", "OUTSIDE"))
            or (exp_outcome == "INCONCLUSIVE" and actual_state in ("INCONCLUSIVE", "DISMISSED"))
        )

        if is_match:
            agent_successes += 1
            if exp_outcome == "CONFIRMED":
                tp += 1
                # Calculate detection delay if crossing frame is specified
                crossing_f = sc.get("crossing_frame")
                first_det_time = detected_inc.get("first_detected_timestamp", 0.0)
                fps = job_result.get("video_metadata", {}).get("fps", 25.0)
                det_frame = int(first_det_time * fps)
                if crossing_f and det_frame >= crossing_f:
                    delays.append(float(det_frame - crossing_f))
                else:
                    delays.append(2.0)  # Nominal delay
            else:
                tn += 1
        else:
            if actual_state == "CONFIRMED" and exp_outcome != "CONFIRMED":
                fp += 1
            elif actual_state != "CONFIRMED" and exp_outcome == "CONFIRMED":
                fn += 1
            else:
                # Inconclusive or mismatch
                fp += 1

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
    success_rate = float(agent_successes) / float(len(scenarios)) if scenarios else 1.0
    mean_delay = float(sum(delays) / len(delays)) if delays else 0.0

    return EvaluationMetrics(
        total_ground_truth_events=len(scenarios),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1_score=round(f1, 3),
        agent_task_success_rate=round(success_rate, 3),
        mean_detection_delay_frames=round(mean_delay, 1),
        evaluated_scenarios_count=len(scenarios),
    )
