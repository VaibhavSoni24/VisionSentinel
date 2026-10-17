"""Unit tests for foundational application health, environment checks, and settings."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "opencv_version" in data
    assert data["opencv_version"].startswith("5.")
    assert data["is_opencv_5"] is True


def test_api_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "operating_mode" in data
    assert "max_upload_mb" in data
    assert "gemini_active" in data
    assert data["gemini_active"] is False  # Safe default


def test_settings_operating_mode():
    mode = settings.get_operating_mode()
    assert "Mode" in mode
    # When no model file exists, should be Mode B
    if not settings.model_path.is_file():
        assert "Mode B" in mode
