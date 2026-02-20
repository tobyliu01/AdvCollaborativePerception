from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from mvp.config import model_3d_examples
from mvp.data.util import bbox_sensor_to_map

from .association import jvc_distance_assignment
from .estimator import MATEConfig, MATEEstimator
from .fov import (
    apply_attack_to_lidar,
    estimate_fov_polygon_fast,
    estimate_fov_polygon_slow,
    resolve_point_count_visibility_overrides,
)
from .fov_visualization import save_fov_visualization
from .types import CAVFramePrediction, FrameData, ScenarioData
from .visibility import RangeVisibilityModel

# Debug-only filter for rapid iteration.
DEBUG_CASE_IDS = {0, 1}
DEBUG_PAIR_IDS = {0, 1, 2}
DEBUG_MODE = True

# Path and settings.
OPENCOOD_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/models/OpenCOOD"
FOV_POLYGON_MODE = "fast"  # fast / slow / both
FOV_VISUALIZATION_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/mate_visulization"
UNMATCHED_GROUND_TRUTH_VISIBILITY_METHOD = "point_count"  # polygon / point_count
UNMATCHED_GROUND_TRUTH_POINT_THRESHOLD = 60


# Get a dict value using raw, string, or integer key forms with a fallback default.
def _dict_get(dct: Any, key: Any, default: Any = None) -> Any:
    if not isinstance(dct, dict):
        return default
    if key in dct:
        return dct[key]
    str_key = str(key)
    if str_key in dct:
        return dct[str_key]
    try:
        int_key = int(key)
        if int_key in dct:
            return dct[int_key]
    except Exception:
        pass
    return default

# Return the frame-level dictionary from either list- or dict-backed payload containers.
def _frame_payload(container: Any, frame_id: int) -> dict:
    if isinstance(container, list):
        if 0 <= frame_id < len(container):
            payload = container[frame_id]
            return payload if isinstance(payload, dict) else {}
        return {}
    if isinstance(container, dict):
        payload = _dict_get(container, frame_id, {})
        return payload if isinstance(payload, dict) else {}
    return {}


# Convert input box data to a float32 `(N, 7)` array.
def _as_boxes(data: Any) -> np.ndarray:
    if data is None:
        return np.empty((0, 7), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float32)
    return arr[:, :7]


# Load one agent's frame prediction file.
def _load_agent_prediction(
    prediction_file: str,
    frame_id: int,
    agent_id: Any,
    pickle_cache_load: Callable[[str], Any],
    logger: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(prediction_file):
        logger.warning(
            "Prediction file missing for CAV %s frame %02d: %s",
            agent_id,
            frame_id,
            prediction_file,
        )
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float32)
    payload = pickle_cache_load(prediction_file)
    frame_data = _frame_payload(payload, frame_id)
    agent_data = _dict_get(frame_data, agent_id, {})
    if not isinstance(agent_data, dict):
        logger.warning(
            "Prediction payload invalid for CAV %s frame %02d in file: %s",
            agent_id,
            frame_id,
            prediction_file,
        )
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float32)
    pred_boxes = _as_boxes(agent_data.get("pred_bboxes"))
    raw_scores = agent_data.get("pred_scores")
    if pred_boxes.shape[0] == 0:
        pred_scores = np.empty((0,), dtype=np.float32)
        raw_scores = np.empty((0,), dtype=np.float32)
    else:
        pred_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        assert(pred_scores.shape[0] == pred_boxes.shape[0])
    if pred_boxes.shape[0] == 0:
        logger.warning(
            "Prediction empty for CAV %s frame %02d in file: %s",
            agent_id,
            frame_id,
            prediction_file,
        )
    return pred_boxes, pred_scores


# Check whether the frame has non-empty attack metadata for the specified agent.
def _frame_has_valid_attack(
    attack_info: Any,
    frame_id: int,
    agent_id: Any,
) -> bool:
    frame_data = _frame_payload(attack_info, frame_id)
    agent_info = _dict_get(frame_data, agent_id, {})
    return isinstance(agent_info, dict) and len(agent_info) > 0


# Read OpenCOOD config and extract `cav_lidar_range` for the perception model.
def _load_perception_lidar_range(
    perception_name: str,
) -> np.ndarray:
    config_path = os.path.join(
        OPENCOOD_ROOT,
        "{}_fusion".format(perception_name),
        "config.yaml",
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "OpenCOOD config not found: {}".format(config_path)
        )

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if not isinstance(config, dict):
        raise ValueError("Invalid YAML config format: {}".format(config_path))

    cav_lidar_range = _dict_get(
        _dict_get(_dict_get(config, "postprocess", {}), "anchor_args", {}),
        "cav_lidar_range",
        None,
    )
    if not isinstance(cav_lidar_range, list):
        raise ValueError(
            "postprocess.anchor_args.cav_lidar_range must be a list in {}".format(config_path)
        )
    cav_lidar_range = np.asarray(cav_lidar_range, dtype=np.float32).reshape(-1)
    if cav_lidar_range.shape[0] < 6:
        raise ValueError("cav_lidar_range is incomplete in {}".format(config_path))
    cav_lidar_range = cav_lidar_range[:6]
    return cav_lidar_range


# Convert numpy scalars/arrays to JSON-serializable Python types.
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_mate_attack_evaluation(
    attacker: Any,
    dataset: Any,
    result_dir: str,
    perception_model_name: str,
    default_shift_model: str,
    perception_name: str,
    attack_frame_ids: Sequence[int],
    pickle_cache_load: Callable[[str], Any],
    logger: Any,
) -> None:
    logger.info(
        "Evaluating defense mate for attack %s at perception %s on mesh model %s",
        attacker.name,
        perception_name,
        default_shift_model,
    )

    # Iterate all attack cases.
    combination_groups: OrderedDict[tuple[int, int], list[Any]] = OrderedDict()
    for ego_attack in attacker.attack_list:
        ego_meta = ego_attack["attack_meta"]
        case_id = int(ego_meta["case_id"])
        pair_id = int(ego_meta["pair_id"])
        if DEBUG_MODE:
            if case_id not in DEBUG_CASE_IDS:
                continue
            if pair_id not in DEBUG_PAIR_IDS:
                continue
        key = (case_id, pair_id)
        combination_groups.setdefault(key, []).append(ego_attack)
    if len(combination_groups) == 0:
        logger.warning("No attack combinations found for attacker %s.", attacker.name)
        return
    case_cache: dict[int, Any] = {}

    # Load perception model range and create a visibility model.
    perception_lidar_range = _load_perception_lidar_range(
        perception_name=perception_name,
    )
    # perception_lidar_range[0] = -120.0
    # perception_lidar_range[1] += 5.0
    # perception_lidar_range[3] = 120.0
    # perception_lidar_range[4] -= 5.0
    logger.info("[MATE] Loaded perception cav_lidar_range: %s", perception_lidar_range.tolist())
    mate_config = MATEConfig(
        penalize_unmatched_predictions=False,
    )
    visibility_model = RangeVisibilityModel(
        max_range_m=120.0,
        cav_lidar_range=(
            np.asarray(perception_lidar_range, dtype=np.float32)
            if perception_lidar_range.shape[0] >= 6
            else None
        ),
    )
    unmatched_gt_visibility_method = str(UNMATCHED_GROUND_TRUTH_VISIBILITY_METHOD).lower()
    if unmatched_gt_visibility_method not in {"polygon", "point_count"}:
        raise ValueError(
            "UNMATCHED_GT_VISIBILITY_METHOD must be one of {polygon, point_count}, got {}".format(
                UNMATCHED_GROUND_TRUTH_VISIBILITY_METHOD
            )
        )
    logger.info(
        "[MATE] unmatched_gt_visibility_method: %s point_threshold: %d",
        unmatched_gt_visibility_method,
        int(UNMATCHED_GROUND_TRUTH_POINT_THRESHOLD),
    )

    # Build the core MATE estimator.
    mate_estimator = MATEEstimator(config=mate_config, visibility_model=visibility_model)

    # Iterate all scenarios.
    for (case_id, pair_id), attack_group in combination_groups.items():
        # Fetch necessary information of each pair.
        pair_meta = attack_group[0]["attack_meta"]
        victim_vehicle_id = pair_meta.get("victim_vehicle_id")
        object_id = pair_meta.get("object_id")
        target_object_id = object_id

        try:
            if case_id not in case_cache:
                case_cache[case_id] = dataset.get_case(
                    case_id,
                    tag="multi_frame",
                    use_lidar=True,
                    use_camera=False,
                )
            case = case_cache[case_id]
        except Exception as e:
            logger.warning(
                "Skipping case %06d pair %02d: failed to load case (%s).",
                case_id,
                pair_id,
                str(e),
            )
            continue

        # Fetch necessary information of each CAV in one pair.
        attack_by_ego: OrderedDict[Any, Any] = OrderedDict()
        for ego_attack in attack_group:
            ego_meta = ego_attack["attack_meta"]
            ego_id = ego_meta["ego_vehicle_id"]
            attack_by_ego.setdefault(ego_id, ego_attack)
        cav_ids = list(pair_meta.get("vehicle_ids", []))

        # Build FrameData for each frame in one scenario and aggregate them into one list.
        scenario_frames: list[FrameData] = []
        # Iterate all frames in this scenario.
        for frame_id in attack_frame_ids:
            if frame_id >= len(case):
                continue
            frame_case = case[frame_id]

            if len(frame_case) == 0:
                scenario_frames.append(
                    FrameData(
                        frame_id=frame_id,
                        gt_bboxes=np.empty((0, 7), dtype=np.float32),
                        cavs={},
                        gt_ids=None,
                    )
                )
                continue

            # Build an aggregator ground truth set in this frame as the union of all CAV ground truth IDs.
            aggregator_gt: OrderedDict[int, np.ndarray] = OrderedDict()
            cav_gt_ids_by_vehicle: dict[Any, list[int]] = {}
            frame_cav_ids = sorted(frame_case.keys(), key=lambda x: str(x))
            # Iterate all CAVs in this frame.
            for frame_cav_id in frame_cav_ids:
                frame_vehicle_data = frame_case[frame_cav_id]
                local_gt_bboxes_sensor = _as_boxes(frame_vehicle_data.get("gt_bboxes"))
                local_object_ids = list(frame_vehicle_data.get("object_ids", []))
                m = min(local_gt_bboxes_sensor.shape[0], len(local_object_ids))
                if m == 0:
                    cav_gt_ids_by_vehicle[frame_cav_id] = []
                    continue
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[:m]
                local_object_ids = local_object_ids[:m]

                # Remove ego vehicle ID from all track IDs.
                ego_id_removal_mask = np.asarray(
                    [obj_id != frame_cav_id for obj_id in local_object_ids],
                    dtype=bool,
                )
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[ego_id_removal_mask]
                local_object_ids = [
                    obj_id for obj_id, keep in zip(local_object_ids, ego_id_removal_mask) if keep
                ]

                # Transform bbox in local vehicle frame to map frame.
                frame_vehicle_pose = np.asarray(
                    frame_vehicle_data["lidar_pose"], dtype=np.float32
                )
                local_gt_bboxes_map = (
                    bbox_sensor_to_map(local_gt_bboxes_sensor, frame_vehicle_pose)
                    if local_gt_bboxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                # Fetch the attack ground truth bbox and update the ground truth bboxes.
                if target_object_id is not None:
                    attack_bbox = np.copy(
                        attack_by_ego[frame_cav_id]["attack_meta"]["bboxes"][frame_id]
                    )
                    attack_bbox[3:6] = model_3d_examples[default_shift_model][3:6]
                    target_bbox_map = bbox_sensor_to_map(
                        np.asarray(attack_bbox, dtype=np.float32).reshape(1, 7),
                        frame_vehicle_pose,
                    )[0]
                    for local_idx, local_obj_id in enumerate(local_object_ids):
                        if local_obj_id == target_object_id:
                            if local_idx < local_gt_bboxes_map.shape[0]:
                                local_gt_bboxes_map[local_idx] = target_bbox_map
                            break

                # Build a list of all track IDs of the current CAV. Also add the track ID and bbox in aggregator list.
                cav_gt_ids: list[int] = []
                for local_obj_id, local_gt_bbox_map in zip(local_object_ids, local_gt_bboxes_map):
                    local_object_id = local_obj_id
                    cav_gt_ids.append(local_object_id)
                    if local_object_id not in aggregator_gt:
                        aggregator_gt[local_object_id] = local_gt_bbox_map
                cav_gt_ids_by_vehicle[frame_cav_id] = list(
                    OrderedDict((x, None) for x in cav_gt_ids).keys()
                )

            # Remove duplicated track IDs in the aggregator.
            if len(aggregator_gt) > 0:
                gt_ids = np.asarray(list(aggregator_gt.keys()), dtype=np.int64)
                gt_bboxes_map = np.stack(list(aggregator_gt.values()), axis=0).astype(np.float32)
            else:
                gt_ids = np.empty((0,), dtype=np.int64)
                gt_bboxes_map = np.empty((0, 7), dtype=np.float32)

            cavs: dict[Any, CAVFramePrediction] = {}
            # Iterate all CAVs in this frame.
            for cav_id in cav_ids:
                gt_track_ids_list = cav_gt_ids_by_vehicle.get(cav_id, [])
                gt_track_ids_set = set(gt_track_ids_list)
                aggregator_gt_ids_list = gt_ids.tolist()
                debug_payload = {
                    "gt_track_ids": gt_track_ids_list,
                    "aggregator_gt_ids": aggregator_gt_ids_list,
                    "matched_track_ids": [],
                    "unmatched_aggregator_gt_ids": [],
                    "filtered_unmatched_gt_ids": [],
                    "out_of_range_unmatched_gt_ids": [],
                    "out_of_range_filtered_unmatched_gt_ids": [],
                    "out_of_polygon_filtered_unmatched_gt_ids": [],
                    "out_of_point_count_filtered_unmatched_gt_ids": [],
                    "point_count_by_track_id": {},
                    "penalized_unmatched_gt_ids": [],
                    "status": "ok",
                }

                # Handle dataset exceptions.
                if cav_id not in frame_case:
                    logger.warning(
                        "CAV %s not found in case %06d pair %02d frame %02d.",
                        cav_id,
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue
                if attack_by_ego.get(cav_id) is None:
                    logger.warning(
                        "CAV %s has no attack entry in case %06d pair %02d frame %02d.",
                        cav_id,
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue
                frame_case_value = frame_case[cav_id]

                # Fetch the lidar point attack info and test if it's a valid attack.
                vehicle_dir = os.path.join(
                    result_dir,
                    "attack/{}/{}/case{:06d}/pair{:02d}/{}".format(
                        perception_model_name,
                        default_shift_model,
                        case_id,
                        pair_id,
                        cav_id,
                    ),
                )
                attack_info_file = os.path.join(vehicle_dir, "attack_info.pkl")
                if not os.path.isfile(attack_info_file):
                    logger.warning(
                        "Missing attack_info file for CAV %s at case %06d pair %02d frame %02d",
                        cav_id,
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue
                attack_info = pickle_cache_load(attack_info_file)
                if not _frame_has_valid_attack(attack_info, frame_id, cav_id):
                    logger.warning(
                        "Invalid attack for CAV %s at case %06d pair %02d frame %02d",
                        cav_id,
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue

                # Fetch the pred bboxes and pred scores of the current CAV.
                prediction_file = os.path.join(
                    vehicle_dir,
                    "frame{}".format(frame_id),
                    "{}.pkl".format(perception_name),
                )
                pred_boxes_sensor, pred_scores = _load_agent_prediction(
                    prediction_file=prediction_file,
                    frame_id=frame_id,
                    agent_id=cav_id,
                    pickle_cache_load=pickle_cache_load,
                    logger=logger,
                )

                # Transform the pred bboxes in vehicle frame into lidar frame.
                vehicle_pose = np.asarray(frame_case_value["lidar_pose"], dtype=np.float32)
                pred_boxes_map = (
                    bbox_sensor_to_map(pred_boxes_sensor, vehicle_pose)
                    if pred_boxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                # Fetch the perturbed lidar data and replace the original lidar points.
                frame_attack_data = _frame_payload(attack_info, frame_id)
                attack_entry = _dict_get(frame_attack_data, cav_id, {})
                base_lidar = np.asarray(
                    frame_case_value.get("lidar", np.empty((0, 4), dtype=np.float32)),
                    dtype=np.float32,
                )
                attacked_lidar = apply_attack_to_lidar(base_lidar, attack_entry)

                # Generate the FOV polygon.
                fov_polygon_fast = np.empty((0, 2), dtype=np.float32)
                fov_polygon_slow = np.empty((0, 2), dtype=np.float32)
                if unmatched_gt_visibility_method == "polygon":
                    range_max = mate_config.lidar_visibility_range_m
                    polygon_mode = str(FOV_POLYGON_MODE).lower()
                    if polygon_mode in {"fast", "both"}:
                        fov_polygon_fast = estimate_fov_polygon_fast(
                            lidar=attacked_lidar,
                            z_min=float(perception_lidar_range[2]),
                            z_max=float(perception_lidar_range[5]),
                            range_max=float(range_max),
                        )
                    if polygon_mode in {"slow", "both"}:
                        fov_polygon_slow = estimate_fov_polygon_slow(
                            lidar=attacked_lidar,
                            z_min=float(perception_lidar_range[2]),
                            z_max=float(perception_lidar_range[5]),
                            range_max=float(range_max),
                        )

                # Draw FOV polygon.
                if unmatched_gt_visibility_method == "polygon":
                    save_fov_visualization(
                        output_root=FOV_VISUALIZATION_ROOT,
                        case_id=case_id,
                        pair_id=pair_id,
                        frame_id=frame_id,
                        cav_id=cav_id,
                        lidar_local=attacked_lidar,
                        pred_boxes_local=pred_boxes_sensor,
                        fov_polygon_fast=fov_polygon_fast,
                        fov_polygon_slow=fov_polygon_slow,
                    )

                # Get the matched AGG track IDs.
                assignment = jvc_distance_assignment(
                    left_boxes=pred_boxes_map,
                    right_boxes=gt_bboxes_map,
                    max_distance_m=mate_config.assignment_distance_m,
                )
                matched_track_ids = []
                if (
                    isinstance(gt_ids, np.ndarray)
                    and gt_ids.shape[0] == gt_bboxes_map.shape[0]
                ):
                    for _, gt_idx in assignment.matched_pairs:
                        matched_track_ids.append(int(gt_ids[gt_idx]))
                debug_payload["matched_track_ids"] = matched_track_ids

                # Get all unmatched AGG track IDs.
                unmatched_gt_ids = [int(gt_ids[idx]) for idx in assignment.unmatched_right]
                debug_payload["unmatched_aggregator_gt_ids"] = unmatched_gt_ids
                filtered_unmatched_gt_indices = [
                    idx for idx in assignment.unmatched_right
                    if gt_ids[idx] in gt_track_ids_set
                ]
                debug_payload["filtered_unmatched_gt_ids"] = [
                    gt_ids[idx] for idx in filtered_unmatched_gt_indices
                ]

                # Create a per-frame data object.
                cav_prediction = CAVFramePrediction(
                    pred_bboxes=pred_boxes_map,
                    pred_scores=pred_scores,
                    pose=vehicle_pose,
                    visible_gt_ids=np.asarray(gt_track_ids_list, dtype=np.int64),
                    fov_polygon_fast=fov_polygon_fast,
                    fov_polygon_slow=fov_polygon_slow,
                    fov_polygon_mode=FOV_POLYGON_MODE,
                    visibility_override_by_track_id={},
                    assignment_matched_pairs=list(assignment.matched_pairs),
                    assignment_unmatched_left=list(assignment.unmatched_left),
                    assignment_unmatched_right=list(assignment.unmatched_right),
                    bboxes_in_global=True,
                )

                status_by_track_id: dict[int, dict] = {}
                visibility_override_by_track_id: dict[int, bool] = {}
                point_count_by_track_id: dict[int, int] = {}
                in_range_filtered_unmatched_gt_indices = []
                forced_track_id = None
                if (
                    victim_vehicle_id is not None
                    and cav_id == victim_vehicle_id
                    and target_object_id is not None
                ):
                    forced_track_id = target_object_id
                # Iterate all unmatched AGG track IDs.
                for gt_idx in filtered_unmatched_gt_indices:
                    track_id = int(gt_ids[int(gt_idx)])

                    # Get the distance-based and polygon-based visibility.
                    status = mate_estimator.visibility_model.visibility_status(
                        cav_prediction,
                        gt_bboxes_map[int(gt_idx)],
                        frame_id,
                    )
                    status_by_track_id[track_id] = status

                    # Filter tracks out of FOV by distance.
                    out_of_distance_range = bool(status.get("out_of_range", False))
                    if out_of_distance_range:
                        visibility_override_by_track_id[track_id] = False
                    else:
                        in_range_filtered_unmatched_gt_indices.append(gt_idx)

                # Filter tracks out of FOV by polygon.
                if unmatched_gt_visibility_method == "polygon":
                    for gt_idx in in_range_filtered_unmatched_gt_indices:
                        track_id = int(gt_ids[int(gt_idx)])
                        if forced_track_id is not None and track_id == forced_track_id:
                            visibility_override_by_track_id[track_id] = True
                            continue
                        status = status_by_track_id[track_id]
                        visibility_override_by_track_id[track_id] = not bool(
                            status.get("out_of_polygon", False)
                        )

                # Filter tracks out of FOV by counting points.
                if unmatched_gt_visibility_method == "point_count":
                    (
                        visibility_override_by_track_id,
                        point_count_by_track_id,
                    ) = resolve_point_count_visibility_overrides(
                        gt_ids=gt_ids,
                        gt_bboxes_map=gt_bboxes_map,
                        in_range_unmatched_gt_indices=in_range_filtered_unmatched_gt_indices,
                        cav_pose=vehicle_pose,
                        points_sensor=attacked_lidar,
                        point_threshold=int(UNMATCHED_GROUND_TRUTH_POINT_THRESHOLD),
                        forced_track_id=forced_track_id,
                    )
                    for track_id, point_count in point_count_by_track_id.items():
                        debug_payload["point_count_by_track_id"][track_id] = int(point_count)
                cav_prediction.visibility_override_by_track_id = visibility_override_by_track_id

                # For debugging.
                out_of_range_unmatched_gt_ids = []
                out_of_range_filtered_unmatched_gt_ids = []
                out_of_polygon_filtered_unmatched_gt_ids = []
                out_of_point_count_filtered_unmatched_gt_ids = []
                penalized_unmatched_gt_ids = []
                for idx in assignment.unmatched_right:
                    track_id = int(gt_ids[int(idx)])
                    status = status_by_track_id.get(track_id)
                    if status is None:
                        status = mate_estimator.visibility_model.visibility_status(
                            cav_prediction,
                            gt_bboxes_map[int(idx)],
                            frame_id,
                        )
                    if bool(status.get("out_of_range", False)):
                        out_of_range_unmatched_gt_ids.append(int(gt_ids[int(idx)]))
                for idx in filtered_unmatched_gt_indices:
                    track_id = int(gt_ids[int(idx)])
                    status = status_by_track_id.get(track_id)
                    if status is None:
                        status = mate_estimator.visibility_model.visibility_status(
                            cav_prediction,
                            gt_bboxes_map[int(idx)],
                            frame_id,
                        )
                    out_of_distance_range = bool(status.get("out_of_range", False))
                    if out_of_distance_range:
                        out_of_range_filtered_unmatched_gt_ids.append(track_id)
                        continue
                    if bool(visibility_override_by_track_id.get(track_id, True)):
                        penalized_unmatched_gt_ids.append(track_id)
                        continue
                    if unmatched_gt_visibility_method == "polygon":
                        out_of_polygon_filtered_unmatched_gt_ids.append(track_id)
                    else:
                        out_of_point_count_filtered_unmatched_gt_ids.append(track_id)
                debug_payload["out_of_range_unmatched_gt_ids"] = out_of_range_unmatched_gt_ids
                debug_payload["out_of_range_filtered_unmatched_gt_ids"] = out_of_range_filtered_unmatched_gt_ids
                debug_payload["out_of_polygon_filtered_unmatched_gt_ids"] = (
                    out_of_polygon_filtered_unmatched_gt_ids
                )
                debug_payload["out_of_point_count_filtered_unmatched_gt_ids"] = (
                    out_of_point_count_filtered_unmatched_gt_ids
                )
                debug_payload["penalized_unmatched_gt_ids"] = penalized_unmatched_gt_ids
                if DEBUG_MODE:
                    logger.info(
                        "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                        case_id,
                        pair_id,
                        frame_id,
                        cav_id,
                        json.dumps(_to_jsonable(debug_payload), sort_keys=True),
                    )

                # Store the predictions of the current CAV.
                cavs[cav_id] = cav_prediction

            # Store the predictions of all CAVs in the current frame.
            scenario_frames.append(
                FrameData(
                    frame_id=frame_id,
                    gt_bboxes=gt_bboxes_map,
                    cavs=cavs,
                    gt_ids=gt_ids,
                )
            )

        # Store data of all frames in one scenario.
        if len(scenario_frames) == 0:
            logger.warning(
                "Skipping MATE on case %06d pair %02d: no frames available.",
                case_id,
                pair_id,
            )
            continue
        scenario_id = str(
            attack_group[0]["attack_meta"].get(
                "scenario_id",
                "case{:06d}_pair{:02d}".format(case_id, pair_id),
            )
        )
        scenario = ScenarioData(
            scenario_id="{}_case{:06d}_pair{:02d}".format(scenario_id, case_id, pair_id),
            frames=scenario_frames,
        )

        # Run MATE.
        try:
            result = mate_estimator.run_scenario(scenario)
        except Exception as e:
            logger.warning(
                "MATE failed on case %06d pair %02d: %s",
                case_id,
                pair_id,
                str(e),
            )
            continue

        # Log results of each (case, pair).
        logger.info(
            "[MATE] case %06d pair %02d victim_vehicle_id: %s object_id: %s",
            case_id,
            pair_id,
            victim_vehicle_id,
            object_id,
        )
        final_agent_trust = {
            agent_id: float(score)
            for agent_id, score in result.final_agent_trust.items()
        }
        final_track_trust = {
            track_id: float(score)
            for track_id, score in result.final_track_trust.items()
        }
        agent_trust_history = {
            agent_id: [float(x) for x in history]
            for agent_id, history in result.agent_trust_history.items()
        }
        logger.info(
            "[MATE] case %06d pair %02d final_agent_trust: %s",
            case_id,
            pair_id,
            json.dumps(final_agent_trust, sort_keys=True),
        )
        logger.info(
            "[MATE] case %06d pair %02d final_track_trust: %s",
            case_id,
            pair_id,
            json.dumps(final_track_trust, sort_keys=True),
        )
        logger.info(
            "[MATE] case %06d pair %02d agent_trust_history: %s",
            case_id,
            pair_id,
            json.dumps(agent_trust_history, sort_keys=True),
        )
        logger.info("")
