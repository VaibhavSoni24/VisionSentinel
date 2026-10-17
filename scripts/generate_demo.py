"""Deterministic synthetic video and ground-truth dataset generator using OpenCV 5.

Generates reproducible fixed-camera scenarios covering:
1. Object moving strictly outside the restricted zone (Safe Outside / Dismissed).
2. Object approaching and crossing into the zone (Confirmed Intrusion).
3. Object approaching the boundary and turning away (Dismissed Turning-Away).
4. Ambiguous low-contrast candidate requiring secondary inspection (Dismissed Noise).
5. Object entering and remaining stationary inside the zone (Confirmed Sustained Entry).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Fixed random seed for complete reproducibility
np.random.seed(42)


def render_scene_background(width: int, height: int) -> np.ndarray:
    """Renders a structured indoor surveillance scene with walls, perspective floor, and grid."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Top wall (cool grey)
    wall_h = int(height * 0.30)
    bg[0:wall_h, :] = (65, 60, 55)

    # Baseboard line
    cv2.line(bg, (0, wall_h), (width, wall_h), (95, 90, 85), 2)

    # Floor (gradient warm grey)
    for y in range(wall_h, height):
        ratio = (y - wall_h) / (height - wall_h)
        val = int(120 + 35 * ratio)
        bg[y, :] = (val - 10, val, val + 5)

    # Perspective floor tile grid
    for x in range(0, width, 60):
        cv2.line(bg, (x, wall_h), (int(x * 1.3) - 80, height), (105, 105, 105), 1)
    for y in range(wall_h + 30, height, 40):
        cv2.line(bg, (0, y), (width, y), (110, 110, 110), 1)

    # Fixed environmental object (table/console on far left)
    cv2.rectangle(bg, (20, wall_h + 10), (100, wall_h + 90), (45, 45, 50), -1)
    cv2.rectangle(bg, (20, wall_h + 10), (100, wall_h + 90), (70, 70, 75), 2)

    return bg


def draw_synthetic_object(
    frame: np.ndarray,
    center: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    is_ambiguous: bool = False,
) -> None:
    """Renders a moving figure-like visual object onto the frame."""
    cx, cy = center
    w, h = size

    if is_ambiguous:
        # Smooth low-texture moving blob (e.g. shadow / ambient reflection)
        cv2.ellipse(frame, (cx, cy), (w // 2, h // 2), 0, 0, 360, (55, 55, 60), -1)
        return

    # Soft contact shadow
    shadow_w = int(w * 1.1)
    shadow_h = int(h * 0.25)
    cv2.ellipse(
        frame,
        (cx, cy + h // 2),
        (shadow_w // 2, shadow_h // 2),
        0,
        0,
        360,
        (40, 40, 40),
        -1,
    )

    # Body torso
    torso_top = cy - h // 3
    torso_bot = cy + h // 2 - 4
    cv2.rectangle(
        frame,
        (cx - w // 2, torso_top),
        (cx + w // 2, torso_bot),
        color,
        -1,
    )
    cv2.rectangle(
        frame,
        (cx - w // 2, torso_top),
        (cx + w // 2, torso_bot),
        (30, 30, 30),
        1,
    )

    # Head
    head_radius = max(6, int(w * 0.35))
    head_cy = torso_top - head_radius + 2
    cv2.circle(frame, (cx, head_cy), head_radius, (180, 200, 220), -1)
    cv2.circle(frame, (cx, head_cy), head_radius, (30, 30, 30), 1)


def generate_synthetic_dataset(
    output_video_path: Path,
    output_gt_path: Path,
    width: int = 640,
    height: int = 360,
    fps: float = 25.0,
    total_frames: int = 380,
) -> Dict:
    """Generates the MP4 video and ground-truth metadata."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_gt_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    bg = render_scene_background(width, height)

    # Restricted zone in normalized coordinates (right ~40% of the room)
    zone_norm = [(0.52, 0.22), (0.92, 0.22), (0.92, 0.88), (0.52, 0.88)]

    ground_truth = {
        "metadata": {
            "dataset_name": "VisionSentinel-Synthetic-Benchmark-v1",
            "frame_width": width,
            "frame_height": height,
            "fps": fps,
            "total_frames": total_frames,
            "zone_polygon_normalized": zone_norm,
        },
        "scenarios": [
            {
                "candidate_id": 1,
                "label": "moving_outside",
                "expected_outcome": "DISMISSED",
                "active_frame_range": [15, 65],
                "description": "Object traverses solely on the left side of the room, outside zone perimeter.",
            },
            {
                "candidate_id": 2,
                "label": "boundary_crossing",
                "expected_outcome": "CONFIRMED",
                "active_frame_range": [85, 155],
                "crossing_frame": 115,
                "description": "Object approaches boundary from left, crosses into restricted zone, continues inside.",
            },
            {
                "candidate_id": 3,
                "label": "approaching_and_turning_away",
                "expected_outcome": "DISMISSED",
                "active_frame_range": [175, 240],
                "turn_frame": 205,
                "description": "Object approaches within 15px of boundary, halts, turns away towards bottom-left.",
            },
            {
                "candidate_id": 4,
                "label": "ambiguous_noise_patch",
                "expected_outcome": "INCONCLUSIVE",
                "active_frame_range": [255, 290],
                "description": "Faint, low-contrast shadow/reflection near boundary triggering secondary inspection and safe inconclusive verdict.",
            },
            {
                "candidate_id": 5,
                "label": "sustained_zone_entry",
                "expected_outcome": "CONFIRMED",
                "active_frame_range": [305, 375],
                "crossing_frame": 325,
                "description": "Object enters zone, halts in center, remains stationary inside zone (>40 frames).",
            },
        ],
    }

    # Frame generation loop
    for f in range(total_frames):
        frame = bg.copy()

        # Slight noise fluctuation
        noise = np.random.normal(0, 0.8, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 1. Candidate 1 (f: 15..65): Left-side movement (Safe Outside)
        if 15 <= f <= 65:
            t = (f - 15) / 50.0
            cx = int(70 + t * 90)
            cy = int(140 + np.sin(t * np.pi) * 35)
            draw_synthetic_object(frame, (cx, cy), (28, 56), (160, 80, 50))

        # 2. Candidate 2 (f: 85..155): Crosses into zone (Confirmed)
        if 85 <= f <= 155:
            t = (f - 85) / 70.0
            cx = int(180 + t * 250)  # Crossing boundary x=332 around f=115
            cy = int(180 + t * 30)
            draw_synthetic_object(frame, (cx, cy), (30, 60), (40, 110, 180))

        # 3. Candidate 3 (f: 175..240): Approaches boundary and turns away (Dismissed)
        if 175 <= f <= 240:
            if f < 205:  # Approaching boundary
                t = (f - 175) / 30.0
                cx = int(180 + t * 135)  # Reaches x=315
                cy = int(220 + t * 15)
            else:  # Turning away and retreating
                t = (f - 205) / 35.0
                cx = int(315 - t * 130)  # Retreats back to x=185
                cy = int(235 + t * 65)
            draw_synthetic_object(frame, (cx, cy), (32, 62), (60, 150, 70))

        # 4. Candidate 4 (f: 255..290): Ambiguous noisy patch near boundary
        if 255 <= f <= 290:
            t = (f - 255) / 35.0
            cx = int(320 + np.sin(t * 4 * np.pi) * 8)
            cy = int(240 + np.cos(t * 3 * np.pi) * 6)
            draw_synthetic_object(frame, (cx, cy), (24, 38), (110, 110, 115), is_ambiguous=True)

        # 5. Candidate 5 (f: 305..375): Enters zone and remains stationary inside (Sustained Confirmed)
        if 305 <= f <= 375:
            if f < 335:  # Walking in
                t = (f - 305) / 30.0
                cx = int(240 + t * 180)  # Reaches x=420 (inside zone)
                cy = int(170 + t * 30)
            else:  # Stationary inside zone
                cx = 420
                cy = 200
            draw_synthetic_object(frame, (cx, cy), (30, 62), (180, 50, 140))

        writer.write(frame)

    writer.release()

    # Save ground-truth JSON
    with open(output_gt_path, "w", encoding="utf-8") as gf:
        json.dump(ground_truth, gf, indent=2)

    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic surveillance demo video and ground truth")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("./data/generated/demo_synthetic.mp4"),
        help="Path for generated MP4 video",
    )
    parser.add_argument(
        "--output-gt",
        type=Path,
        default=Path("./data/generated/demo_ground_truth.json"),
        help="Path for generated ground-truth JSON",
    )
    args = parser.parse_args()

    print(f"Generating synthetic video: {args.output_video}")
    gt = generate_synthetic_dataset(args.output_video, args.output_gt)
    print(f"Generated {gt['metadata']['total_frames']} frames at {gt['metadata']['fps']} FPS.")
    print(f"Ground-truth metadata written to {args.output_gt}")


if __name__ == "__main__":
    main()
