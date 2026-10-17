"""Restricted-zone spatial geometry and boundary crossing analysis using OpenCV 5."""

from typing import Tuple

import cv2
import numpy as np

from app.api.schemas import ZonePolygon


class RestrictedZone:
    """Represents a polygonal restricted area with spatial intersection tests."""

    def __init__(self, zone_config: ZonePolygon, frame_width: int, frame_height: int):
        self.name = zone_config.name
        self.normalized_points = zone_config.points
        self.width = frame_width
        self.height = frame_height

        # Denormalize to pixel coordinates
        pixel_pts = [
            [int(round(pt[0] * frame_width)), int(round(pt[1] * frame_height))]
            for pt in self.normalized_points
        ]
        self.polygon_pts = np.array(pixel_pts, dtype=np.int32)
        # Reshape to (N, 1, 2) for OpenCV polygon functions
        self.contour = self.polygon_pts.reshape((-1, 1, 2))

        # Precompute polygon mask for fast overlap analysis
        self._mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(self._mask, [self.polygon_pts], 255)
        self.zone_area = float(cv2.countNonZero(self._mask))

    def point_test(self, point: Tuple[int, int]) -> Tuple[bool, float]:
        """Performs point-in-polygon test with signed distance.

        Returns:
            (is_inside_or_on_edge, signed_distance)
            signed_distance > 0: strictly inside
            signed_distance == 0: exactly on edge
            signed_distance < 0: strictly outside (absolute value is distance to edge in pixels)
        """
        dist = cv2.pointPolygonTest(
            self.contour, (float(point[0]), float(point[1])), measureDist=True
        )
        is_inside = dist >= 0
        return is_inside, float(dist)

    def bbox_overlap_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculates fraction of the bounding box that intersects the restricted zone."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return 0.0

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        bbox_crop_mask = self._mask[y1:y2, x1:x2]
        overlap_pixels = cv2.countNonZero(bbox_crop_mask)
        total_bbox_pixels = float(w * h)
        return float(overlap_pixels) / total_bbox_pixels if total_bbox_pixels > 0 else 0.0

    def evaluate_trajectory_motion(
        self,
        prev_pt: Tuple[int, int],
        curr_pt: Tuple[int, int],
    ) -> Tuple[float, float, str]:
        """Evaluates whether motion is approaching, turning away from, or entering the zone.

        Returns:
            (distance_delta, total_displacement, motion_state)
        """
        _, prev_dist = self.point_test(prev_pt)
        _, curr_dist = self.point_test(curr_pt)

        # signed distance: positive = inside, negative = outside
        dist_delta = curr_dist - prev_dist
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        displacement = float(np.hypot(dx, dy))

        if curr_dist >= 0 and prev_dist < 0:
            state = "CROSSED_INSIDE"
        elif curr_dist >= 0:
            state = "WITHIN_ZONE"
        elif dist_delta > 1.5:  # Distance becoming less negative -> moving closer
            state = "APPROACHING"
        elif dist_delta < -1.5:  # Distance becoming more negative -> moving away
            state = "RECEDING"
        else:
            state = "PARALLEL_OR_STATIONARY"

        return dist_delta, displacement, state
