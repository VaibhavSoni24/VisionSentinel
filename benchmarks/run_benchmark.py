"""Reproducible computer-vision pipeline benchmark for Vision Sentinel.

Measures:
- Preprocessing and Gaussian smoothing latency
- OpenCV 5 MOG2 background subtraction latency
- Contour extraction and centroid moment calculations
- Spatial point-in-polygon and boundary distance testing
- Median, mean, p95 frame processing latency, throughput (FPS)
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.api.schemas import ZonePolygon
from app.vision.motion import MotionExtractor
from app.vision.preprocessing import apply_gaussian_blur
from app.vision.tracking import CandidateTracker
from app.vision.video_io import VideoReader
from app.vision.zones import RestrictedZone


def run_benchmark(
    video_path: Path,
    num_runs: int = 3,
    warmup_frames: int = 25,
) -> Dict[str, Any]:
    print(f"Starting benchmark on {video_path}...")
    reader = VideoReader(video_path)
    meta = reader.metadata

    zone_config = ZonePolygon(
        points=[(0.52, 0.22), (0.92, 0.22), (0.92, 0.88), (0.52, 0.88)],
    )
    zone = RestrictedZone(zone_config, meta.width, meta.height)

    # 1. Warm-up run
    print(f"Performing warm-up ({warmup_frames} frames)...")
    warmup_extractor = MotionExtractor(history=300)
    warmup_count = 0
    for _, _, frame in reader.frames():
        _ = apply_gaussian_blur(frame, 3)
        _, _ = warmup_extractor.extract(frame)
        warmup_count += 1
        if warmup_count >= warmup_frames:
            break

    # 2. Benchmark execution runs
    run_fps_list: List[float] = []
    frame_latencies_ms: List[float] = []
    component_times = {"preprocess_ms": [], "motion_ms": [], "tracking_ms": []}

    print(f"Executing {num_runs} benchmark passes...")
    for r in range(num_runs):
        extractor = MotionExtractor(history=300)
        tracker = CandidateTracker(zone)
        frames_processed = 0

        t_start = time.perf_counter()
        for frame_idx, ts, frame in reader.frames():
            t0 = time.perf_counter()

            # Preprocess
            blurred = apply_gaussian_blur(frame, 3)
            t1 = time.perf_counter()

            # Motion & Contours
            fg_mask, detections = extractor.extract(blurred)
            t2 = time.perf_counter()

            # Track & Zone Tests
            active_tracks = tracker.update(frame_idx, ts, detections)
            for track in active_tracks:
                _ = zone.point_test(track.current_centroid)
            t3 = time.perf_counter()

            dt_frame = (t3 - t0) * 1000.0
            frame_latencies_ms.append(dt_frame)
            component_times["preprocess_ms"].append((t1 - t0) * 1000.0)
            component_times["motion_ms"].append((t2 - t1) * 1000.0)
            component_times["tracking_ms"].append((t3 - t2) * 1000.0)
            frames_processed += 1

        t_total = time.perf_counter() - t_start
        fps = float(frames_processed) / t_total if t_total > 0 else 0.0
        run_fps_list.append(fps)
        print(
            f"  Pass {r + 1}/{num_runs}: {frames_processed} frames in {t_total:.2f}s ({fps:.1f} FPS)"
        )

    latencies = np.array(frame_latencies_ms)
    median_lat = float(np.median(latencies))
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    mean_fps = float(np.mean(run_fps_list))

    results = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "opencv_version": cv2.__version__,
            "python_version": platform.python_version(),
            "os_platform": platform.platform(),
            "cpu_architecture": platform.machine(),
            "frame_dimensions": f"{meta.width}x{meta.height}",
            "total_frames_measured": len(frame_latencies_ms),
            "num_runs": num_runs,
        },
        "throughput": {
            "mean_fps": round(mean_fps, 2),
            "run_fps": [round(f, 2) for f in run_fps_list],
            "realtime_factor": round(mean_fps / meta.fps, 2),
        },
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "median": round(median_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
        },
        "component_breakdown_ms": {
            "preprocess_mean": round(float(np.mean(component_times["preprocess_ms"])), 2),
            "motion_extraction_mean": round(float(np.mean(component_times["motion_ms"])), 2),
            "tracking_and_zones_mean": round(float(np.mean(component_times["tracking_ms"])), 2),
        },
    }

    print("=" * 60)
    print("  VISION SENTINEL — BENCHMARK RESULTS")
    print("=" * 60)
    print(f"OpenCV Version: {cv2.__version__} on {platform.machine()} ({platform.system()})")
    print(
        f"Throughput:     {results['throughput']['mean_fps']} FPS ({results['throughput']['realtime_factor']}x realtime)"
    )
    print(
        f"Frame Latency:  Median: {results['latency_ms']['median']} ms | p95: {results['latency_ms']['p95']} ms"
    )
    print(
        f"Components:     Preprocess: {results['component_breakdown_ms']['preprocess_mean']}ms | MOG2: {results['component_breakdown_ms']['motion_extraction_mean']}ms | Tracking: {results['component_breakdown_ms']['tracking_and_zones_mean']}ms"
    )
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenCV 5 vision pipeline performance benchmark"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("./data/generated/demo_synthetic.mp4"),
        help="Path to benchmark video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmarks/benchmark_results.json"),
        help="Path to write JSON benchmark metrics",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs")
    args = parser.parse_args()

    if not args.video.is_file():
        from scripts.generate_demo import generate_synthetic_dataset

        generate_synthetic_dataset(args.video, args.video.parent / "demo_ground_truth.json")

    results = run_benchmark(args.video, num_runs=args.runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results written to {args.output}")


if __name__ == "__main__":
    main()
