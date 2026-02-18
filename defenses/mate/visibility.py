from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mvp.data.util import bbox_map_to_sensor
from opencood.utils.box_utils import mask_boxes_outside_range_numpy

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
                "distance_m": dist,
            }

        return {
            "visible": True,
            "out_of_range": False,
            "out_of_range_120m": False,
            "out_of_model_range": False,
            "distance_m": dist,
        }
