#!/usr/bin/env python3
"""Model setup script for Vision Sentinel.

Downloads and verifies a small, permissively licensed ONNX detector
(MediaPipe Person Detection from OpenCV Zoo, Apache 2.0 license)
for Mode A operation. If omitted or offline, Vision Sentinel defaults
to Mode B (classical OpenCV 5 motion candidate detection), which
requires no downloads or external model weights.
"""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_model")

# MediaPipe Person Detection model from OpenCV Zoo (Apache-2.0 License)
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx"
DEFAULT_TARGET_PATH = Path("models/person_detector.onnx")


def verify_onnx_with_opencv(path: Path) -> bool:
    """Tests whether OpenCV 5 DNN can successfully load the ONNX model."""
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("OpenCV 5 DNN successfully loaded model from: %s", path)
        return True
    except Exception as exc:
        logger.error("OpenCV 5 DNN failed to load model from %s: %s", path, exc)
        return False


def download_model(url: str, destination: Path) -> bool:
    """Downloads model weights with progress reporting and error handling."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(".tmp")

    logger.info("Downloading ONNX model from: %s", url)
    logger.info("Destination path: %s", destination)

    try:
        headers = {"User-Agent": "VisionSentinel-Setup/1.0"}
        req = urllib.request.Request(url, headers=headers)

        with (
            urllib.request.urlopen(req, timeout=30) as response,
            open(temp_destination, "wb") as out_file,
        ):
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None
            downloaded = 0
            block_size = 64 * 1024

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(
                        f"\rDownloading: {downloaded / 1024:.1f} KB / {total_size / 1024:.1f} KB ({percent:.1f}%)"
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloading: {downloaded / 1024:.1f} KB")
                    sys.stdout.flush()

        sys.stdout.write("\n")
        temp_destination.replace(destination)
        logger.info("Download completed successfully.")
        return True
    except Exception as exc:
        logger.warning("Download failed: %s", exc)
        if temp_destination.exists():
            temp_destination.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup local ONNX object detector for Vision Sentinel"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TARGET_PATH,
        help="Path where ONNX model should be placed (default: models/person_detector.onnx)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=MODEL_URL,
        help="URL to fetch ONNX model from",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if the model is currently available and valid without downloading",
    )
    args = parser.parse_args()

    target_path: Path = args.output

    if args.check_only:
        if target_path.exists():
            print(f"[STATUS] Model file found at: {target_path}")
            valid = verify_onnx_with_opencv(target_path)
            if valid:
                print("[STATUS] Mode A (Local ONNX Detector) is READY.")
                sys.exit(0)
            else:
                print(
                    "[STATUS] Model file invalid. Mode B (Classical Motion Detection) will be used."
                )
                sys.exit(1)
        else:
            print(f"[STATUS] Model not present at: {target_path}")
            print(
                "[STATUS] Vision Sentinel will run in Mode B (Classical OpenCV 5 Motion Detection)."
            )
            print("         Mode B is completely self-contained and requires zero downloads.")
            sys.exit(0)

    if target_path.exists():
        logger.info("Model already exists at: %s", target_path)
        if verify_onnx_with_opencv(target_path):
            logger.info("Existing model is valid and ready for Mode A.")
            sys.exit(0)
        else:
            logger.warning("Existing model file failed verification. Re-downloading...")

    success = download_model(args.url, target_path)
    if success:
        valid = verify_onnx_with_opencv(target_path)
        if valid:
            logger.info("Setup complete. Vision Sentinel is configured for Mode A.")
            sys.exit(0)
        else:
            logger.error("Downloaded file is not a valid ONNX model. Removing invalid file.")
            target_path.unlink(missing_ok=True)
            sys.exit(1)
    else:
        logger.info(
            "Vision Sentinel will continue using Mode B (Classical OpenCV 5 Motion Candidate)."
        )
        logger.info(
            "No external weights required; core verification workflow remains 100% operational."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
