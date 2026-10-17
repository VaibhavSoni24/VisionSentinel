"""Application configuration and environment settings for Vision Sentinel."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv()


class Settings(BaseModel):
    """Centralized, strongly-typed application configuration."""

    # Core Application Settings
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    app_port: int = Field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    max_upload_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "100")))
    max_video_seconds: int = Field(
        default_factory=lambda: int(os.getenv("MAX_VIDEO_SECONDS", "180"))
    )

    # Local Pretrained Detector (Mode A vs Mode B)
    model_path: Path = Field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", "./models/person_detector.onnx"))
    )
    local_model_enabled: bool = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_ENABLED", "true").lower() == "true"
    )

    # Optional Google Gemini Reasoning Provider
    gemini_enabled: bool = Field(
        default_factory=lambda: os.getenv("GEMINI_ENABLED", "false").lower() == "true"
    )
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_max_calls_per_job: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_CALLS_PER_JOB", "3"))
    )
    gemini_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))
    )
    gemini_send_user_media: bool = Field(
        default_factory=lambda: os.getenv("GEMINI_SEND_USER_MEDIA", "false").lower() == "true"
    )
    gemini_free_tier_only: bool = Field(
        default_factory=lambda: os.getenv("GEMINI_FREE_TIER_ONLY", "true").lower() == "true"
    )

    # Cloud & Optimization Stubs
    aws_storage_enabled: bool = Field(
        default_factory=lambda: os.getenv("AWS_STORAGE_ENABLED", "false").lower() == "true"
    )
    aws_region: str = Field(default_factory=lambda: os.getenv("AWS_REGION", "ap-south-1"))
    aws_s3_bucket: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_S3_BUCKET") or None)
    cool_enabled: bool = Field(
        default_factory=lambda: os.getenv("COOL_ENABLED", "false").lower() == "true"
    )

    @property
    def uploads_dir(self) -> Path:
        p = self.data_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def outputs_dir(self) -> Path:
        p = self.data_dir / "outputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def generated_dir(self) -> Path:
        p = self.data_dir / "generated"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def is_model_available(self) -> bool:
        """Checks if local ONNX weights exist and local detector is enabled."""
        return self.local_model_enabled and self.model_path.is_file()

    def get_operating_mode(self) -> str:
        """Returns the honest perception mode description."""
        if self.is_model_available():
            return "Mode A (Local ONNX Detector)"
        return "Mode B (Classical OpenCV Motion Candidate)"

    def is_gemini_active(self) -> bool:
        """Gemini is only active when enabled AND a non-empty key is provided."""
        return self.gemini_enabled and bool(self.gemini_api_key and self.gemini_api_key.strip())


# Global settings instance singleton
settings = Settings()
