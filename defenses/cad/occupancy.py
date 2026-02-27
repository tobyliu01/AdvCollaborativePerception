from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from mvp.config import data_root, model_3d_examples
from mvp.data.util import bbox_sensor_to_map, pcd_sensor_to_map
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.polygon_space import bbox_to_polygon, get_free_space, get_occupied_space

from .common import apply_attack_to_lidar, as_boxes, dict_get, frame_payload, occupancy_map_case_pair_dir
from .config import MAX_RANGE, TOTAL_FRAMES


def save_fused_occupancy_map_frame(
    *,
    default_shift_model: str,
    case_id: int,
    pair_id: int,
    frame_id: int,
    frame_occupancy: Dict[Any, Dict[str, Any]],
    frame_metric: Dict[str, Any],
) -> None:
    save_dir = occupancy_map_case_pair_dir(
        default_shift_model=default_shift_model,
        case_id=case_id,
        pair_id=pair_id,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "frame{}.pkl".format(int(frame_id)))

    per_cav_regions: Dict[Any, Dict[str, Any]] = OrderedDict()
    for cav_id, vehicle_data in sorted(frame_occupancy.items(), key=lambda x: str(x[0])):
        per_cav_regions[cav_id] = {
            "occupied_areas": vehicle_data.get("occupied_areas", []),
            "free_areas": vehicle_data.get("free_areas", []),
            "ego_area": vehicle_data.get("ego_area"),
        }

    payload = {
        "case_id": int(case_id),
        "pair_id": int(pair_id),
        "frame_id": int(frame_id),
        "regions": {
            "per_cav": per_cav_regions,
            "fused_occupied": frame_metric.get("_fused_occupied_geom"),
            "fused_free": frame_metric.get("_fused_free_geom"),
            "conflicted": frame_metric.get("_conflicted_geoms", []),
        },
        "summary": {
            "conflicted": bool(frame_metric.get("conflicted", False)),
            "conflicted_count": int(frame_metric.get("conflicted_count", 0)),
            "conflicted_area_total": float(frame_metric.get("conflicted_area_total", 0.0)),
            "conflicted_regions": frame_metric.get("conflicted_regions", []),
            "conflict_pair_details": frame_metric.get("conflict_pair_details", []),
        },
    }
    with open(save_file, "wb") as f:
        pickle.dump(payload, f)


def load_cached_occupancy_feature(
    *,
    default_shift_model: str,
    case_id: int,
    pair_id: int,
    frame_ids: Sequence[int],
) -> List[Dict[Any, Dict[str, Any]]]:
    occupancy_feature: List[Dict[Any, Dict[str, Any]]] = [{} for _ in range(TOTAL_FRAMES)]
    load_dir = occupancy_map_case_pair_dir(
        default_shift_model=default_shift_model,
        case_id=case_id,
        pair_id=pair_id,
    )
    for frame_id in frame_ids:
        load_file = os.path.join(load_dir, "frame{}.pkl".format(int(frame_id)))
        if not os.path.isfile(load_file):
            raise FileNotFoundError("Cached occupancy map not found: {}".format(load_file))
        with open(load_file, "rb") as f:
            payload = pickle.load(f)
        regions = payload.get("regions", {})
        per_cav = regions.get("per_cav", {})
        frame_data: Dict[Any, Dict[str, Any]] = {}
        for cav_id, cav_regions in per_cav.items():
            frame_data[cav_id] = {
                "occupied_areas": cav_regions.get("occupied_areas", []),
                "free_areas": cav_regions.get("free_areas", []),
                "ego_area": cav_regions.get("ego_area"),
            }
        occupancy_feature[int(frame_id)] = frame_data
    return occupancy_feature


def occupancy_map(
    *,
    case_id: int,
    pair_id: int,
    case: Any,
    attack_by_ego: "OrderedDict[Any, Any]",
    cav_ids: Sequence[Any],
    frame_ids: Sequence[int],
    result_dir: str,
    perception_model_name: str,
    default_shift_model: str,
    pickle_cache_load: Callable[[str], Any],
    lidar_seg_api: Any,
    logger: Any,
) -> List[Dict[Any, Dict[str, Any]]]:
    occupancy_feature: List[Dict[Any, Dict[str, Any]]] = [
        {} for _ in range(TOTAL_FRAMES)
    ]

    attack_info_by_cav: Dict[Any, Any] = {}
    for cav_id in cav_ids:
        attack_info_file = os.path.join(
            result_dir,
            "attack/{}/{}/case{:06d}/pair{:02d}/{}/attack_info.pkl".format(
                perception_model_name,
                default_shift_model,
                int(case_id),
                int(pair_id),
                cav_id,
            ),
        )
        if os.path.isfile(attack_info_file):
            attack_info_by_cav[cav_id] = pickle_cache_load(attack_info_file)
        else:
            attack_info_by_cav[cav_id] = [{} for _ in range(TOTAL_FRAMES)]
            logger.warning(
                "[CAD] Missing attack_info for case %06d pair %02d CAV %s: %s",
                int(case_id),
                int(pair_id),
                cav_id,
                attack_info_file,
            )

    target_object_id = None
    if len(attack_by_ego) > 0:
        sample_attack = next(iter(attack_by_ego.values()))
        target_object_id = dict_get(sample_attack, "attack_meta", {}).get("object_id")

    for frame_id in frame_ids:
        frame_case = case[frame_id]

        for cav_id in cav_ids:
            if cav_id not in frame_case:
                continue
            vehicle_data = frame_case[cav_id]
            lidar_pose = np.asarray(
                vehicle_data.get("lidar_pose", np.zeros((6,), dtype=np.float32)),
                dtype=np.float32,
            )
            base_lidar = np.asarray(
                vehicle_data.get("lidar", np.empty((0, 4), dtype=np.float32)),
                dtype=np.float32,
            )

            frame_attack_data = frame_payload(attack_info_by_cav.get(cav_id, {}), frame_id)
            attack_entry = dict_get(frame_attack_data, cav_id, {})
            attacked_lidar = apply_attack_to_lidar(base_lidar, attack_entry)

            gt_bboxes_sensor = as_boxes(vehicle_data.get("gt_bboxes"))
            object_ids = list(vehicle_data.get("object_ids", []))
            valid_count = min(gt_bboxes_sensor.shape[0], len(object_ids))
            if valid_count > 0:
                gt_bboxes_sensor = gt_bboxes_sensor[:valid_count]
                object_ids = object_ids[:valid_count]
            else:
                gt_bboxes_sensor = np.empty((0, 7), dtype=np.float32)
                object_ids = []

            if target_object_id is not None and cav_id in attack_by_ego:
                attack_meta = dict_get(attack_by_ego[cav_id], "attack_meta", {})
                attack_bboxes = attack_meta.get("bboxes", None)
                assert(attack_bboxes is not None)
                attacked_target_bbox = np.asarray(
                    attack_bboxes[frame_id],
                    dtype=np.float32,
                ).reshape(7)
                attacked_target_bbox[3:6] = np.asarray(
                    model_3d_examples[default_shift_model][3:6],
                    dtype=np.float32,
                )
                for local_idx, local_object_id in enumerate(object_ids):
                    if local_object_id == target_object_id:
                        if local_idx < gt_bboxes_sensor.shape[0]:
                            gt_bboxes_sensor[local_idx] = attacked_target_bbox
                        break

            gt_bboxes_map = (
                bbox_sensor_to_map(gt_bboxes_sensor, lidar_pose)
                if gt_bboxes_sensor.shape[0] > 0
                else np.empty((0, 7), dtype=np.float32)
            )

            map_name = vehicle_data.get("map")
            if map_name is None:
                logger.warning(
                    "[CAD] Missing map name at case %06d pair %02d frame %02d CAV %s",
                    int(case_id),
                    int(pair_id),
                    int(frame_id),
                    cav_id,
                )
                continue

            lane_info = pickle_cache_load(
                os.path.join(data_root, "carla/{}_lane_info.pkl".format(map_name))
            )
            lane_areas = pickle_cache_load(
                os.path.join(data_root, "carla/{}_lane_areas.pkl".format(map_name))
            )
            lane_planes = pickle_cache_load(
                os.path.join(data_root, "carla/{}_ground_planes.pkl".format(map_name))
            )

            if attacked_lidar.shape[0] == 0:
                pcd_map = np.empty((0, 3), dtype=np.float32)
                ground_indices = np.empty((0,), dtype=np.int64)
                in_lane_mask = np.zeros((0,), dtype=bool)
                point_height = np.empty((0,), dtype=np.float32)
                object_segments = []
                occupied_areas = []
                occupied_areas_height = []
                free_areas = []
            else:
                pcd_map = pcd_sensor_to_map(attacked_lidar, lidar_pose)
                ground_indices, in_lane_mask, point_height = get_ground_plane(
                    pcd_map,
                    lane_info=lane_info,
                    lane_areas=lane_areas,
                    lane_planes=lane_planes,
                    method="map",
                )
                lidar_seg = lidar_segmentation(
                    attacked_lidar,
                    method="squeezeseq",
                    interface=lidar_seg_api,
                )
                object_segments = filter_segmentation(
                    attacked_lidar,
                    lidar_seg,
                    lidar_pose,
                    in_lane_mask=in_lane_mask,
                    point_height=point_height,
                    max_range=MAX_RANGE,
                )
                object_mask = np.zeros((pcd_map.shape[0],), dtype=bool)
                if len(object_segments) > 0:
                    object_indices = np.hstack(object_segments).reshape(-1)
                    object_mask[object_indices] = True

                occupied_areas, occupied_areas_height = get_occupied_space(
                    pcd_map,
                    object_segments,
                    point_height=point_height,
                    height_thres=0,
                )
                free_areas = get_free_space(
                    attacked_lidar,
                    lidar_pose,
                    object_mask,
                    in_lane_mask=in_lane_mask,
                    point_height=point_height,
                    max_range=MAX_RANGE,
                    height_thres=0,
                    height_tolerance=0.2,
                )

            ego_bbox = np.asarray(
                vehicle_data.get("ego_bbox", np.zeros((7,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            assert(ego_bbox.shape[0] == 7)
            ego_bbox = ego_bbox[:7]
            ego_area = bbox_to_polygon(ego_bbox)
            ego_area_height = float(ego_bbox[5])

            occupancy_feature[frame_id][cav_id] = {
                "lidar": attacked_lidar,
                "lidar_pose": lidar_pose,
                "map": map_name,
                "ego_bbox": ego_bbox,
                "ego_area": ego_area,
                "ego_area_height": ego_area_height,
                "ground_indices": ground_indices,
                "point_height": point_height,
                "object_segments": object_segments,
                "occupied_areas": occupied_areas,
                "occupied_areas_height": occupied_areas_height,
                "free_areas": free_areas,
                "gt_bboxes": gt_bboxes_sensor,
                "gt_bboxes_map": gt_bboxes_map,
                "object_ids": np.asarray(object_ids, dtype=np.int64),
            }
    return occupancy_feature
