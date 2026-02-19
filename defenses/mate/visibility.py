from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mvp.data.util import bbox_map_to_sensor
from opencood.utils.box_utils import mask_boxes_outside_range_numpy

from .fov import point_in_polygon_strict
from .types import CAVFramePrediction


@dataclass
class RangeVisibilityModel:
    """
    Simple range-based visibility model.
    """

    max_range_m: float = 120.0
    cav_lidar_range: Optional[np.ndarray] = None

    def is_visible(self, cav: CAVFramePrediction, target_bbox: np.ndarray, frame_idx: int) -> bool:
        return bool(self.visibility_status(cav, target_bbox, frame_idx)["visible"])

    def visibility_status(
        self,
        cav: CAVFramePrediction,
        target_bbox: np.ndarray,
        frame_idx: int,
    ) -> dict:
        _ = frame_idx
        if cav.pose is None or cav.pose.shape[0] < 2:
            return {
                "visible": True,
                "out_of_range": False,
                "out_of_range_120m": False,
                "out_of_model_range": False,
                "out_of_polygon": False,
                "distance_m": None,
            }

        cx = float(cav.pose[0])
        cy = float(cav.pose[1])

        tx = float(target_bbox[0])
        ty = float(target_bbox[1])
        dx = tx - cx
        dy = ty - cy
        dist = math.hypot(dx, dy)
        out_of_range_120m = dist >= float(self.max_range_m)

        out_of_model_range = False
        out_of_polygon = False
        target_local = None
        cav_lidar_range = self.cav_lidar_range
        if cav_lidar_range is not None and cav.pose.shape[0] >= 6:
            try:
                range = np.asarray(cav_lidar_range, dtype=np.float32).reshape(-1)
                if range.shape[0] >= 6:
                    target_local = bbox_map_to_sensor(
                        np.asarray(target_bbox, dtype=np.float32),
                        np.asarray(cav.pose, dtype=np.float32),
                    )
                    local_box = np.asarray(target_local, dtype=np.float32).reshape(1, -1)
                    _, in_range_mask = mask_boxes_outside_range_numpy(
                        boxes=local_box,
                        limit_range=range[:6],
                        order="lwh",
                        return_mask=True,
                    )
                    out_of_model_range = not bool(in_range_mask[0])
            except Exception:
                out_of_model_range = False

        if out_of_range_120m or out_of_model_range:
            return {
                "visible": False,
                "out_of_range": True,
                "out_of_range_120m": out_of_range_120m,
                "out_of_model_range": out_of_model_range,
                "out_of_polygon": False,
                "distance_m": dist,
            }

        if target_local is None and cav.pose is not None and cav.pose.shape[0] >= 6:
            try:
                target_local = bbox_map_to_sensor(
                    np.asarray(target_bbox, dtype=np.float32),
                    np.asarray(cav.pose, dtype=np.float32),
                )
            except Exception:
                target_local = None

        point_xy = None
        if target_local is not None and np.asarray(target_local).shape[0] >= 2:
            point_xy = np.asarray(target_local[:2], dtype=np.float32)

        fast_poly = cav.fov_polygon_fast
        slow_poly = cav.fov_polygon_slow
        fast_valid = isinstance(fast_poly, np.ndarray) and fast_poly.ndim == 2 and fast_poly.shape[0] >= 3
        slow_valid = isinstance(slow_poly, np.ndarray) and slow_poly.ndim == 2 and slow_poly.shape[0] >= 3

        if point_xy is not None and (fast_valid or slow_valid):
            mode = str(getattr(cav, "fov_polygon_mode", "fast")).lower()
            fast_visible = point_in_polygon_strict(point_xy, fast_poly) if fast_valid else True
            slow_visible = point_in_polygon_strict(point_xy, slow_poly) if slow_valid else True

            if mode == "slow":
                poly_visible = slow_visible if slow_valid else fast_visible
            elif mode == "both":
                if fast_valid and slow_valid:
                    poly_visible = fast_visible and slow_visible
                elif fast_valid:
                    poly_visible = fast_visible
                else:
                    poly_visible = slow_visible
            else:
                poly_visible = fast_visible if fast_valid else slow_visible

            out_of_polygon = not bool(poly_visible)

        if out_of_polygon:
            return {
                "visible": False,
                "out_of_range": True,
                "out_of_range_120m": False,
                "out_of_model_range": False,
                "out_of_polygon": True,
                "distance_m": dist,
            }

        return {
            "visible": True,
            "out_of_range": False,
            "out_of_range_120m": False,
            "out_of_model_range": False,
            "out_of_polygon": False,
            "distance_m": dist,
        }
