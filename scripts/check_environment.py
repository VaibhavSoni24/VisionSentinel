#!/usr/bin/env python3
"""Environment and dependency diagnostic utility for Vision Sentinel.

Checks:
- Python runtime and OS architecture
- OpenCV 5 installation and cv2.__version__
- Required runtime packages
- Local model availability (Mode A vs Mode B)
- Optional Gemini configuration (Free Tier safety checks)
- Data storage directories and permissions
"""

import os
import platform
import sys
from pathlib import Path

# Add project root to sys.path so app packages can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_python_environment() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "is_64bit": sys.maxsize > 2**32,
    }


def check_opencv() -> dict:
    status = {"installed": False, "version": None, "is_opencv_5": False}
    try:
        import cv2

        status["installed"] = True
        status["version"] = cv2.__version__
        status["is_opencv_5"] = cv2.__version__.startswith("5.")
    except ImportError as e:
        status["error"] = str(e)
    return status


def check_core_dependencies() -> dict:
    dependencies = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "dotenv",
        "jinja2",
        "httpx",
        "pytest",
    ]
    results = {}
    for dep in dependencies:
        try:
            if dep == "dotenv":
                import dotenv  # noqa: F401
            else:
                __import__(dep)
            results[dep] = True
        except ImportError:
            results[dep] = False
    return results


def check_model_and_mode(project_root: Path) -> dict:
    model_path = project_root / "models" / "person_detector.onnx"
    exists = model_path.is_file()
    return {
        "model_path": str(model_path),
        "model_file_exists": exists,
        "operating_mode": "Mode A (Local ONNX Detector)"
        if exists
        else "Mode B (Classical OpenCV Motion Candidate)",
    }


def check_gemini_config() -> dict:
    from dotenv import load_dotenv

    load_dotenv()
    enabled = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
    api_key_present = bool(os.getenv("GEMINI_API_KEY", "").strip())
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    send_user_media = os.getenv("GEMINI_SEND_USER_MEDIA", "false").lower() == "true"

    return {
        "enabled": enabled,
        "api_key_configured": api_key_present,
        "model": model,
        "send_user_media_consent": send_user_media,
        "free_tier_enforced": os.getenv("GEMINI_FREE_TIER_ONLY", "true").lower() == "true",
    }


def check_storage_directories(project_root: Path) -> dict:
    dirs = [
        project_root / "data" / "uploads",
        project_root / "data" / "outputs",
        project_root / "data" / "generated",
    ]
    status = {}
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        writable = os.access(d, os.W_OK)
        status[d.name] = {"path": str(d), "exists": d.exists(), "writable": writable}
    return status


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    print("=" * 60)
    print("  VISION SENTINEL — ENVIRONMENT DIAGNOSTIC CHECK")
    print("=" * 60)

    py_env = check_python_environment()
    print(
        f"[Python] Runtime: {py_env['python_version']} on {py_env['platform']} ({py_env['architecture']})"
    )

    cv_env = check_opencv()
    if cv_env["installed"]:
        cv_status = "PASS" if cv_env["is_opencv_5"] else "WARNING (Not OpenCV 5)"
        print(f"[OpenCV] Version: {cv_env['version']} [{cv_status}]")
    else:
        print(f"[OpenCV] NOT INSTALLED: {cv_env.get('error')}")

    deps = check_core_dependencies()
    missing_deps = [dep for dep, ok in deps.items() if not ok]
    if not missing_deps:
        print("[Dependencies] All core web & testing dependencies verified.")
    else:
        print(f"[Dependencies] Missing: {', '.join(missing_deps)}")

    mode_info = check_model_and_mode(project_root)
    print(f"[Vision Engine] Active Mode: {mode_info['operating_mode']}")

    gemini_info = check_gemini_config()
    print(
        f"[Gemini Reasoning] Enabled: {gemini_info['enabled']}, Key Present: {gemini_info['api_key_configured']}"
    )

    storage = check_storage_directories(project_root)
    all_storage_ok = all(s["writable"] for s in storage.values())
    print(f"[Storage] Directories status: {'OK' if all_storage_ok else 'Permission Error'}")

    print("=" * 60)
    ready = cv_env["is_opencv_5"] and not missing_deps and all_storage_ok
    if ready:
        print("  STATUS: ENVIRONMENT VERIFIED & READY")
    else:
        print("  STATUS: ATTENTION REQUIRED (See details above)")
    print("=" * 60)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
