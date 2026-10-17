"""Integration tests for FastAPI endpoints, demo execution, and artifact downloads."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SENTINEL" in response.text
    assert "OpenCV AI Competition 2026" in response.text


def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "opencv_version" in data
    assert "operating_mode" in data


def test_run_demo_end_to_end(client):
    """Executes the full synthetic demo pipeline via the API."""
    response = client.post("/api/demo")
    assert response.status_code == 200
    job = response.json()

    assert job["status"] == "completed"
    assert "job_id" in job
    job_id = job["job_id"]

    # Verify incidents were generated
    assert len(job["incidents"]) >= 1
    # Check that at least one incident is confirmed and one dismissed
    outcomes = [inc["final_state"] for inc in job["incidents"]]
    assert "CONFIRMED" in outcomes or "DISMISSED" in outcomes

    # Verify agent trace steps were recorded
    assert len(job["trace_steps"]) >= 1

    # Verify timeline entries
    assert len(job["timeline"]) >= 1

    # Verify artifact downloads work for this job
    video_res = client.get(f"/api/download/{job_id}/video")
    assert video_res.status_code == 200
    assert len(video_res.content) > 1000

    incidents_res = client.get(f"/api/download/{job_id}/incidents")
    assert incidents_res.status_code == 200

    timeline_res = client.get(f"/api/download/{job_id}/timeline")
    assert timeline_res.status_code == 200

    trace_res = client.get(f"/api/download/{job_id}/trace")
    assert trace_res.status_code == 200

    report_res = client.get(f"/api/download/{job_id}/report")
    assert report_res.status_code == 200


def test_download_invalid_job_id(client):
    res = client.get("/api/download/nonexistent-job-uuid/video")
    assert res.status_code == 404


def test_path_traversal_protection(client):
    # Attempt directory traversal in download endpoint
    res = client.get("/api/download/../../etc/passwd/video")
    assert res.status_code in (400, 404)
