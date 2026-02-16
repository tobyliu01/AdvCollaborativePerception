from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import CAVFramePrediction


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class RangeVisibilityModel:
    """
    Simple fallback FOV model when LiDAR ray-traced FOV polygons are unavailable.
    """

    max_range_m: float = 120.0
    horizontal_fov_deg: float = 360.0

    def is_visible(self, cav: CAVFramePrediction, target_bbox: np.ndarray, frame_idx: int) -> bool:
        _ = frame_idx
        if cav.pose is None or cav.pose.shape[0] < 2:
            return True

        cx = float(cav.pose[0])
        cy = float(cav.pose[1])
        yaw = float(cav.pose[3]) if cav.pose.shape[0] >= 4 else 0.0

        tx = float(target_bbox[0])
        ty = float(target_bbox[1])
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        if dist > self.max_range_m:
            return False

        if self.horizontal_fov_deg >= 359.9:
            return True

        bearing = math.atan2(dy, dx)
        rel = abs(_wrap_to_pi(bearing - yaw))
        return rel <= math.radians(self.horizontal_fov_deg) * 0.5
