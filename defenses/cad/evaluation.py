from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from mvp.config import model_3d_examples
from mvp.data.util import bbox_sensor_to_map
from mvp.tools.polygon_space import bbox_to_polygon

from .common import dict_get
from .config import ATTACK_MATCH_DISTANCE_THRESHOLD


def build_attack_gt_bbox_by_frame(
    *,
    case: Any,
    attack_by_ego: "OrderedDict[Any, Any]",
    frame_ids: Sequence[int],
    default_shift_model: str,
    victim_vehicle_id: Any,
) -> Dict[int, Polygon]:
    if len(attack_by_ego) == 0:
        return {}

    if victim_vehicle_id in attack_by_ego:
        reference_ego_id = victim_vehicle_id
    else:
        reference_ego_id = next(iter(attack_by_ego.keys()))
    reference_attack = attack_by_ego[reference_ego_id]

    attack_meta = dict_get(reference_attack, "attack_meta", {})
    attack_bboxes = attack_meta.get("bboxes", None)
    assert(attack_bboxes is not None)

    attack_bbox_by_frame: Dict[int, Polygon] = {}
    for frame_id in frame_ids:
        if reference_ego_id not in case[frame_id]:
            continue
        attack_bbox_sensor = np.asarray(
            attack_bboxes[frame_id],
            dtype=np.float32,
        ).reshape(7)
        attack_bbox_sensor = attack_bbox_sensor.copy()
        attack_bbox_sensor[3:6] = np.asarray(
            model_3d_examples[default_shift_model][3:6],
            dtype=np.float32,
        )
        lidar_pose = np.asarray(
            case[frame_id][reference_ego_id]["lidar_pose"],
            dtype=np.float32,
        )
        attack_bbox_map = bbox_sensor_to_map(
            attack_bbox_sensor.reshape(1, 7),
            lidar_pose,
        )[0]
        attack_bbox_by_frame[int(frame_id)] = bbox_to_polygon(attack_bbox_map)
    return attack_bbox_by_frame


def defense_evaluation(
    *,
    case_pair_metrics: "OrderedDict[Tuple[int, int], Any]",
    case_pair_attack_bbox_by_frame: "OrderedDict[Tuple[int, int], Dict[int, Polygon]]",
    frame_ids: Sequence[int],
    logger: Any,
) -> Dict[str, Any]:
    total_frames = 0
    anomaly_frames = 0
    matched_frames = 0
    failed_frames_total = 0
    failure_by_case_pair: "OrderedDict[Tuple[int, int], list[int]]" = OrderedDict()

    for (case_id, pair_id), metrics in case_pair_metrics.items():
        failure_by_case_pair[(int(case_id), int(pair_id))] = []
        attack_bbox_by_frame = case_pair_attack_bbox_by_frame.get((case_id, pair_id), {})
        for frame_id in frame_ids:
            frame_metric = metrics[frame_id]
            total_frames += 1
            if bool(frame_metric.get("conflicted", False)):
                anomaly_frames += 1
            attack_bbox = attack_bbox_by_frame.get(int(frame_id))
            conflict_polygons = frame_metric.get("_conflicted_geoms", [])
            matched = False
            if attack_bbox is not None and len(conflict_polygons) > 0:
                for conflict_polygon in conflict_polygons:
                    if float(conflict_polygon.distance(attack_bbox)) <= ATTACK_MATCH_DISTANCE_THRESHOLD:
                        matched = True
                        break
            if matched:
                matched_frames += 1
            else:
                failed_frames_total += 1
                failure_by_case_pair[(int(case_id), int(pair_id))].append(int(frame_id))

    success_frames = matched_frames
    success_rate = float(success_frames) / float(total_frames) if total_frames > 0 else 0.0
    logger.info(
        "[CAD] Occupancy anomaly frames: %d/%d",
        int(anomaly_frames),
        int(total_frames),
    )
    logger.info(
        "[CAD] Occupancy anomaly success frames: %d/%d",
        int(success_frames),
        int(total_frames),
    )
    logger.info(
        "[CAD] Occupancy anomaly success rate: %.6f",
        float(success_rate),
    )
    logger.info("[CAD] Occupancy anomaly failure:")
    for (case_id, pair_id), failed_frames in failure_by_case_pair.items():
        logger.info("(%d, %d):%s", int(case_id), int(pair_id), str(failed_frames))
    return {
        "anomaly_frames": int(anomaly_frames),
        "total_frames": int(total_frames),
        "success_frames": int(success_frames),
        "success_rate": float(success_rate),
        "matched_frames": int(matched_frames),
        "failed_frames": int(failed_frames_total),
        "failure_by_case_pair": dict(failure_by_case_pair),
    }
