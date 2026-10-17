"""REST API endpoints for video analysis, demo execution, and artifact downloads."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.api.schemas import (
    AnalysisJobStatus,
    JobResult,
    ZonePolygon,
)
from app.config import settings
from app.storage.local import StorageSecurityError, storage
from app.vision.pipeline import run_verification_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Vision Sentinel API"])

# In-memory job registry for instant status lookups
JOBS_STORE: Dict[str, Dict[str, Any]] = {}


@router.post("/api/demo", response_model=JobResult, summary="Run built-in synthetic demo")
async def run_demo(
    zone_points_json: Optional[str] = Form(None),
) -> JobResult:
    """Executes verification pipeline on the built-in deterministic synthetic scenario."""
    job_id = storage.create_job_id()
    demo_video_path = settings.generated_dir / "demo_synthetic.mp4"
    demo_gt_path = settings.generated_dir / "demo_ground_truth.json"

    # Ensure synthetic dataset is generated
    if not demo_video_path.is_file():
        from scripts.generate_demo import generate_synthetic_dataset

        generate_synthetic_dataset(demo_video_path, demo_gt_path)

    # Zone configuration
    if zone_points_json:
        try:
            pts = json.loads(zone_points_json)
            zone_config = ZonePolygon(points=pts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid zone polygon points JSON: {e}")
    else:
        # Default zone matching demo scenario (right 40% of room)
        zone_config = ZonePolygon(
            name="Restricted Zone 1",
            points=[(0.52, 0.22), (0.92, 0.22), (0.92, 0.88), (0.52, 0.88)],
        )

    JOBS_STORE[job_id] = {"status": AnalysisJobStatus.RUNNING.value, "progress": 0.1}

    try:
        result = run_verification_pipeline(
            job_id=job_id,
            video_path=demo_video_path,
            zone_config=zone_config,
        )
        JOBS_STORE[job_id] = {
            "status": AnalysisJobStatus.COMPLETED.value,
            "progress": 1.0,
            "result": result.model_dump(),
        }
        return result
    except Exception as e:
        logger.error("Demo execution failed: %s", e, exc_info=True)
        JOBS_STORE[job_id] = {"status": AnalysisJobStatus.FAILED.value, "error": str(e)}
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {e}")


@router.post(
    "/api/analyze", response_model=JobResult, summary="Analyze uploaded fixed-camera video"
)
async def analyze_video(
    video: UploadFile = File(...),
    zone_points_json: Optional[str] = Form(None),
    send_user_media: bool = Form(False),
) -> JobResult:
    """Uploads a video, validates size/format, and executes restricted-zone verification."""
    job_id = storage.create_job_id()

    # Read and validate upload
    content = await video.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Upload size ({size_mb:.1f} MB) exceeds maximum limit of {settings.max_upload_mb} MB",
        )

    saved_upload_path = storage.save_upload(job_id, video.filename or "upload.mp4", content)

    # Parse polygon
    if zone_points_json:
        try:
            pts = json.loads(zone_points_json)
            zone_config = ZonePolygon(points=pts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid zone polygon JSON: {e}")
    else:
        zone_config = ZonePolygon()

    JOBS_STORE[job_id] = {"status": AnalysisJobStatus.RUNNING.value, "progress": 0.1}

    try:
        result = run_verification_pipeline(
            job_id=job_id,
            video_path=saved_upload_path,
            zone_config=zone_config,
            send_user_media=send_user_media,
        )
        JOBS_STORE[job_id] = {
            "status": AnalysisJobStatus.COMPLETED.value,
            "progress": 1.0,
            "result": result.model_dump(),
        }
        return result
    except Exception as e:
        logger.error("Analysis job %s failed: %s", job_id, e, exc_info=True)
        JOBS_STORE[job_id] = {"status": AnalysisJobStatus.FAILED.value, "error": str(e)}
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@router.get("/api/results/{job_id}", summary="Get job status and results")
async def get_results(job_id: str) -> Dict[str, Any]:
    """Retrieves analysis results for a completed job."""
    if job_id in JOBS_STORE and "result" in JOBS_STORE[job_id]:
        return JOBS_STORE[job_id]["result"]

    output_dir = storage.outputs_dir / job_id
    report_file = output_dir / "job_report.json"
    data = storage.load_json(report_file)
    if data:
        return data

    if job_id in JOBS_STORE:
        return JOBS_STORE[job_id]

    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@router.get("/api/download/{job_id}/{artifact_type}", summary="Download analysis artifacts")
async def download_artifact(job_id: str, artifact_type: str):
    """Safely streams generated artifacts (video, JSON, CSV, JSONL)."""
    try:
        job_dir = storage.get_job_output_dir(job_id)
    except StorageSecurityError:
        raise HTTPException(status_code=400, detail="Invalid job ID path")

    artifact_map = {
        "video": (job_dir / "annotated_result.mp4", "video/mp4", f"vision_sentinel_{job_id}.mp4"),
        "incidents": (job_dir / "incidents.json", "application/json", f"incidents_{job_id}.json"),
        "timeline": (job_dir / "timeline.csv", "text/csv", f"timeline_{job_id}.csv"),
        "trace": (
            job_dir / "agent_trace.jsonl",
            "application/x-jsonlines",
            f"agent_trace_{job_id}.jsonl",
        ),
        "report": (job_dir / "job_report.json", "application/json", f"report_{job_id}.json"),
    }

    if artifact_type not in artifact_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown artifact type '{artifact_type}'. Valid: {list(artifact_map.keys())}",
        )

    file_path, media_type, download_filename = artifact_map[artifact_type]
    if not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Artifact '{artifact_type}' not found for job '{job_id}'"
        )

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=download_filename,
    )


@router.get("/", response_class=HTMLResponse, summary="Serve UI")
async def index_page():
    """Serves the main Vision Sentinel web interface."""
    index_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Vision Sentinel API Running</h1><p>UI template loading...</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
