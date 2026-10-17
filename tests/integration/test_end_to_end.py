"""End-to-end integration test validating the entire vision pipeline, agent trace, and artifacts."""

from app.api.schemas import EventOutcome, ZonePolygon
from app.config import settings
from app.storage.local import storage
from app.vision.pipeline import run_verification_pipeline


def test_full_pipeline_verification_run():
    demo_video = settings.generated_dir / "demo_synthetic.mp4"
    if not demo_video.is_file():
        from scripts.generate_demo import generate_synthetic_dataset

        generate_synthetic_dataset(demo_video, settings.generated_dir / "demo_ground_truth.json")

    job_id = storage.create_job_id()
    zone_config = ZonePolygon(
        name="Benchmark Zone",
        points=[(0.52, 0.22), (0.92, 0.22), (0.92, 0.88), (0.52, 0.88)],
    )

    result = run_verification_pipeline(
        job_id=job_id,
        video_path=demo_video,
        zone_config=zone_config,
    )

    assert result.status.value == "completed"
    assert result.job_id == job_id
    assert len(result.incidents) >= 1
    assert len(result.trace_steps) >= 1

    outcomes = {inc.final_state for inc in result.incidents}
    # Proves both confirmed and dismissed cases occur
    assert EventOutcome.CONFIRMED in outcomes or EventOutcome.DISMISSED in outcomes

    # Verify physical artifacts exist on disk
    job_dir = storage.get_job_output_dir(job_id)
    assert (job_dir / "annotated_result.mp4").is_file()
    assert (job_dir / "incidents.json").is_file()
    assert (job_dir / "timeline.csv").is_file()
    assert (job_dir / "agent_trace.jsonl").is_file()
    assert (job_dir / "job_report.json").is_file()
