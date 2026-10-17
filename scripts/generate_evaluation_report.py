"""Generates the official docs/EVALUATION.md report and JSON results."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.schemas import ZonePolygon
from app.evaluation.metrics import evaluate_against_ground_truth
from app.evaluation.reporting import generate_evaluation_markdown
from app.storage.local import storage
from app.vision.pipeline import run_verification_pipeline
from scripts.generate_demo import generate_synthetic_dataset


def run_evaluation_and_generate_report() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    video_path = project_root / "data" / "generated" / "demo_synthetic.mp4"
    gt_path = project_root / "data" / "generated" / "demo_ground_truth.json"
    doc_output_path = project_root / "docs" / "EVALUATION.md"
    json_output_path = project_root / "data" / "generated" / "evaluation_results.json"

    if not video_path.is_file() or not gt_path.is_file():
        print("Generating synthetic demo and ground truth...")
        generate_synthetic_dataset(video_path, gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    job_id = storage.create_job_id()
    zone_pts = ground_truth["metadata"]["zone_polygon_normalized"]
    zone_config = ZonePolygon(points=zone_pts)

    print(f"Executing verification pipeline for evaluation (Job {job_id})...")
    result = run_verification_pipeline(
        job_id=job_id,
        video_path=video_path,
        zone_config=zone_config,
    )
    result_dict = result.model_dump()

    metrics = evaluate_against_ground_truth(result_dict, ground_truth)
    print(
        f"Evaluation complete: Precision={metrics.precision:.2f}, Recall={metrics.recall:.2f}, F1={metrics.f1_score:.2f}"
    )

    # Generate docs/EVALUATION.md
    generate_evaluation_markdown(
        metrics=metrics,
        job_result=result_dict,
        ground_truth=ground_truth,
        output_path=doc_output_path,
    )
    print(f"Report generated at {doc_output_path}")

    # Save metrics JSON
    with open(json_output_path, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "metrics": metrics.__dict__,
                "job_id": job_id,
                "opencv_version": result.opencv_version,
            },
            jf,
            indent=2,
        )
    return doc_output_path


if __name__ == "__main__":
    run_evaluation_and_generate_report()
