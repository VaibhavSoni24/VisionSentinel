"""Local file and artifact storage manager with path-traversal protection."""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


class StorageSecurityError(Exception):
    """Raised when an illegal path traversal or unauthorized file access is detected."""

    pass


class LocalStorage:
    """Manages job-scoped data persistence, uploads, annotations, and audit traces."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = (data_dir or settings.data_dir).resolve()
        self.uploads_dir = self.data_dir / "uploads"
        self.outputs_dir = self.data_dir / "outputs"
        self.generated_dir = self.data_dir / "generated"

        for p in (self.uploads_dir, self.outputs_dir, self.generated_dir):
            p.mkdir(parents=True, exist_ok=True)

    def _sanitize_path(self, target_path: Path) -> Path:
        """Ensures that target_path is strictly within self.data_dir."""
        resolved = target_path.resolve()
        try:
            resolved.relative_to(self.data_dir)
        except ValueError:
            raise StorageSecurityError(
                f"Access denied: Path traversal detected outside data directory: {target_path}"
            )
        return resolved

    def create_job_id(self) -> str:
        """Generates a secure, unique job ID."""
        return str(uuid.uuid4())

    def get_job_output_dir(self, job_id: str) -> Path:
        """Returns the isolated output directory for a specific job."""
        job_dir = self.outputs_dir / job_id
        self._sanitize_path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_upload(self, job_id: str, filename: str, content: bytes) -> Path:
        """Saves an uploaded file safely within the uploads directory."""
        clean_name = Path(filename).name  # Strips any directory traversal components
        if not clean_name:
            clean_name = f"upload_{job_id}.mp4"
        job_upload_dir = self.uploads_dir / job_id
        self._sanitize_path(job_upload_dir)
        job_upload_dir.mkdir(parents=True, exist_ok=True)

        target_file = job_upload_dir / clean_name
        self._sanitize_path(target_file)
        with open(target_file, "wb") as f:
            f.write(content)
        return target_file

    def save_json(self, file_path: Path, data: Any) -> None:
        """Atomically saves serialized JSON data."""
        self._sanitize_path(file_path)
        temp_file = file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(file_path)

    def load_json(self, file_path: Path) -> Optional[Any]:
        """Safely loads and deserializes a JSON artifact."""
        self._sanitize_path(file_path)
        if not file_path.is_file():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_trace_line(self, trace_file: Path, entry: Dict[str, Any]) -> None:
        """Appends a single JSONL event trace entry."""
        self._sanitize_path(trace_file)
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_csv(self, file_path: Path, headers: List[str], rows: List[List[Any]]) -> None:
        """Saves an event timeline as CSV format."""
        self._sanitize_path(file_path)
        import csv

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def cleanup_job(self, job_id: str) -> None:
        """Removes all files associated with a job ID."""
        for base in (self.uploads_dir, self.outputs_dir):
            job_path = base / job_id
            if job_path.exists() and job_path.is_dir():
                shutil.rmtree(job_path, ignore_errors=True)


# Global storage instance
storage = LocalStorage()
