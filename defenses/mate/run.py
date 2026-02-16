from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Sequence

import numpy as np

from mvp.data.util import bbox_sensor_to_map

from .association import jvc_distance_assignment
from .estimator import MATEConfig, MATEEstimator
from .types import CAVFramePrediction, FrameData, ScenarioData

# Debug-only filter for rapid iteration.
DEBUG_CASE_IDS = {0, 1}
DEBUG_PAIR_IDS = {0, 1, 2}


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


def _align_scores(scores: Any, n_boxes: int) -> np.ndarray:
    if n_boxes <= 0:
        return np.empty((0,), dtype=np.float32)
    if scores is None:
        return np.ones((n_boxes,), dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.shape[0] == n_boxes:
        return arr
    if arr.shape[0] == 0:
        return np.ones((n_boxes,), dtype=np.float32)
    aligned = np.ones((n_boxes,), dtype=np.float32)
    m = min(n_boxes, arr.shape[0])
    aligned[:m] = arr[:m]
    return aligned


def _load_agent_prediction(
    prediction_file: str,
    frame_id: int,
    agent_id: Any,
    pickle_cache_load: Callable[[str], Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(prediction_file):
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float32)
    payload = pickle_cache_load(prediction_file)
    frame_data = _frame_payload(payload, frame_id)
    agent_data = _dict_get(frame_data, agent_id, {})
    if not isinstance(agent_data, dict):
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float32)
    pred_boxes = _as_boxes(agent_data.get("pred_bboxes"))
    pred_scores = _align_scores(agent_data.get("pred_scores"), pred_boxes.shape[0])
    return pred_boxes, pred_scores


def _frame_has_valid_attack(
    attack_info: Any,
    frame_id: int,
    agent_id: Any,
) -> bool:
    frame_data = _frame_payload(attack_info, frame_id)
    agent_info = _dict_get(frame_data, agent_id, {})
    return isinstance(agent_info, dict) and len(agent_info) > 0


def _to_int_track_id(track_id: Any) -> Any:
    try:
        return int(track_id)
    except Exception:
        return None


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

    combination_groups: OrderedDict[tuple[int, int], list[Any]] = OrderedDict()
    for attack in attacker.attack_list:
        meta = attack["attack_meta"]
        case_id = int(meta["case_id"])
        pair_id = int(meta["pair_id"])
        if case_id not in DEBUG_CASE_IDS or pair_id not in DEBUG_PAIR_IDS:
            continue
        key = (case_id, pair_id)
        combination_groups.setdefault(key, []).append(attack)

    if len(combination_groups) == 0:
        logger.warning("No attack combinations found for attacker %s.", attacker.name)
        return

    case_cache: dict[int, Any] = {}
    mate_config = MATEConfig()
    mate_estimator = MATEEstimator(config=mate_config)

    for (case_id, pair_id), attack_group in combination_groups.items():
        try:
            if case_id not in case_cache:
                case_cache[case_id] = dataset.get_case(
                    case_id,
                    tag="multi_frame",
                    use_lidar=False,
                    use_camera=False,
                )
            case = case_cache[case_id]
        except Exception as exc:
            logger.warning(
                "Skipping case %06d pair %02d: failed to load case (%s).",
                case_id,
                pair_id,
                str(exc),
            )
            continue

        attack_by_ego: OrderedDict[Any, Any] = OrderedDict()
        vehicle_id_order: OrderedDict[Any, None] = OrderedDict()
        for attack in attack_group:
            meta = attack["attack_meta"]
            ego_id = meta["ego_vehicle_id"]
            attack_by_ego.setdefault(ego_id, attack)
            for vehicle_id in meta.get("vehicle_ids", []):
                vehicle_id_order.setdefault(vehicle_id, None)
            vehicle_id_order.setdefault(ego_id, None)
        vehicle_ids = list(vehicle_id_order.keys())

        scenario_frames: list[FrameData] = []
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
            frame_vehicle_ids = sorted(frame_case.keys(), key=lambda x: str(x))
            for frame_vehicle_id in frame_vehicle_ids:
                frame_vehicle_data = frame_case[frame_vehicle_id]
                local_gt_bboxes_sensor = _as_boxes(frame_vehicle_data.get("gt_bboxes"))
                local_object_ids = list(frame_vehicle_data.get("object_ids", []))
                m = min(local_gt_bboxes_sensor.shape[0], len(local_object_ids))
                if m <= 0:
                    cav_gt_ids_by_vehicle[frame_vehicle_id] = []
                    continue
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[:m]
                local_object_ids = local_object_ids[:m]
                keep_mask = np.asarray(
                    [str(obj_id) != str(frame_vehicle_id) for obj_id in local_object_ids],
                    dtype=bool,
                )
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[keep_mask]
                local_object_ids = [
                    obj_id for obj_id, keep in zip(local_object_ids, keep_mask) if keep
                ]

                frame_vehicle_pose = np.asarray(
                    frame_vehicle_data["lidar_pose"], dtype=np.float32
                )
                local_gt_bboxes_map = (
                    bbox_sensor_to_map(local_gt_bboxes_sensor, frame_vehicle_pose)
                    if local_gt_bboxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                cav_gt_ids: list[int] = []
                for local_obj_id, local_gt_bbox_map in zip(local_object_ids, local_gt_bboxes_map):
                    int_track_id = _to_int_track_id(local_obj_id)
                    if int_track_id is None:
                        continue
                    cav_gt_ids.append(int_track_id)
                    if int_track_id not in aggregator_gt:
                        aggregator_gt[int_track_id] = local_gt_bbox_map
                cav_gt_ids_by_vehicle[frame_vehicle_id] = list(
                    OrderedDict((x, None) for x in cav_gt_ids).keys()
                )

            if len(aggregator_gt) > 0:
                gt_ids = np.asarray(list(aggregator_gt.keys()), dtype=np.int64)
                gt_bboxes_map = np.stack(list(aggregator_gt.values()), axis=0).astype(np.float32)
            else:
                gt_ids = np.empty((0,), dtype=np.int64)
                gt_bboxes_map = np.empty((0, 7), dtype=np.float32)

            cavs: dict[Any, CAVFramePrediction] = {}
            for vehicle_id in vehicle_ids:
                gt_track_ids_list = cav_gt_ids_by_vehicle.get(vehicle_id, [])
                cav_gt_ids_set = set(gt_track_ids_list)
                aggregator_gt_ids_list = [int(x) for x in gt_ids.tolist()]
                debug_payload = {
                    "gt_track_ids": gt_track_ids_list,
                    "aggregator_gt_ids": aggregator_gt_ids_list,
                    "matched_track_ids": [],
                    "matched_pairs": [],
                    "unmatched_pred_indices": [],
                    "unmatched_gt_indices": [],
                    "unmatched_aggregator_gt_ids": [],
                    "filtered_unmatched_gt_indices": [],
                    "filtered_unmatched_gt_ids": [],
                    "status": "ok",
                }

                if vehicle_id not in frame_case:
                    debug_payload["status"] = "vehicle_missing_in_frame"
                    logger.info(
                        "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                        case_id,
                        pair_id,
                        frame_id,
                        str(vehicle_id),
                        json.dumps(debug_payload, sort_keys=True),
                    )
                    continue
                if attack_by_ego.get(vehicle_id) is None:
                    debug_payload["status"] = "vehicle_not_in_attack_group"
                    logger.info(
                        "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                        case_id,
                        pair_id,
                        frame_id,
                        str(vehicle_id),
                        json.dumps(debug_payload, sort_keys=True),
                    )
                    continue

                vehicle_dir = os.path.join(
                    result_dir,
                    "attack/{}/{}/case{:06d}/pair{:02d}/{}".format(
                        perception_model_name,
                        default_shift_model,
                        case_id,
                        pair_id,
                        str(vehicle_id),
                    ),
                )
                attack_info_file = os.path.join(vehicle_dir, "attack_info.pkl")
                if not os.path.isfile(attack_info_file):
                    debug_payload["status"] = "missing_attack_info_file"
                    logger.info(
                        "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                        case_id,
                        pair_id,
                        frame_id,
                        str(vehicle_id),
                        json.dumps(debug_payload, sort_keys=True),
                    )
                    continue

                attack_info = pickle_cache_load(attack_info_file)
                if not _frame_has_valid_attack(attack_info, frame_id, vehicle_id):
                    debug_payload["status"] = "invalid_attack_frame"
                    logger.info(
                        "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                        case_id,
                        pair_id,
                        frame_id,
                        str(vehicle_id),
                        json.dumps(debug_payload, sort_keys=True),
                    )
                    continue

                prediction_file = os.path.join(
                    vehicle_dir,
                    "frame{}".format(frame_id),
                    "{}.pkl".format(perception_name),
                )
                pred_boxes_sensor, pred_scores = _load_agent_prediction(
                    prediction_file=prediction_file,
                    frame_id=frame_id,
                    agent_id=vehicle_id,
                    pickle_cache_load=pickle_cache_load,
                )

                vehicle_pose = np.asarray(frame_case[vehicle_id]["lidar_pose"], dtype=np.float32)
                pred_boxes_map = (
                    bbox_sensor_to_map(pred_boxes_sensor, vehicle_pose)
                    if pred_boxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                assignment = jvc_distance_assignment(
                    left_boxes=pred_boxes_map,
                    right_boxes=gt_bboxes_map,
                    max_distance_m=mate_config.assignment_distance_m,
                )
                matched_track_ids = []
                if isinstance(gt_ids, np.ndarray) and gt_ids.shape[0] == gt_bboxes_map.shape[0]:
                    for _, gt_idx in assignment.matched_pairs:
                        matched_track_ids.append(int(gt_ids[gt_idx]))
                debug_payload["matched_track_ids"] = matched_track_ids
                debug_payload["matched_pairs"] = [
                    [int(li), int(ri)] for li, ri in assignment.matched_pairs
                ]
                debug_payload["unmatched_pred_indices"] = [int(x) for x in assignment.unmatched_left]
                debug_payload["unmatched_gt_indices"] = [int(x) for x in assignment.unmatched_right]
                unmatched_gt_ids = [int(gt_ids[idx]) for idx in assignment.unmatched_right]
                debug_payload["unmatched_aggregator_gt_ids"] = unmatched_gt_ids
                filtered_unmatched_gt_indices = [
                    int(idx) for idx in assignment.unmatched_right
                    if int(gt_ids[idx]) in cav_gt_ids_set
                ]
                debug_payload["filtered_unmatched_gt_indices"] = filtered_unmatched_gt_indices
                debug_payload["filtered_unmatched_gt_ids"] = [
                    int(gt_ids[idx]) for idx in filtered_unmatched_gt_indices
                ]
                logger.info(
                    "[MATE_DEBUG] case %06d pair %02d frame %02d cav %s matches: %s",
                    case_id,
                    pair_id,
                    frame_id,
                    str(vehicle_id),
                    json.dumps(debug_payload, sort_keys=True),
                )

                cavs[vehicle_id] = CAVFramePrediction(
                    pred_bboxes=pred_boxes_map,
                    pred_scores=pred_scores,
                    pose=vehicle_pose[:2],
                    visible_gt_ids=np.asarray(gt_track_ids_list, dtype=np.int64),
                    boxes_in_global=True,
                )

            scenario_frames.append(
                FrameData(
                    frame_id=frame_id,
                    gt_bboxes=gt_bboxes_map,
                    cavs=cavs,
                    gt_ids=gt_ids,
                )
            )

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
        pair_meta = attack_group[0]["attack_meta"]
        victim_vehicle_id = pair_meta.get("victim_vehicle_id")
        object_id = pair_meta.get("object_id")

        try:
            result = mate_estimator.run_scenario(scenario)
        except Exception as exc:
            logger.warning(
                "MATE failed on case %06d pair %02d: %s",
                case_id,
                pair_id,
                str(exc),
            )
            continue

        logger.info(
            "[MATE] case %06d pair %02d victim_vehicle_id: %s object_id: %s",
            case_id,
            pair_id,
            str(victim_vehicle_id),
            str(object_id),
        )
        final_agent_trust = {
            str(agent_id): float(score)
            for agent_id, score in result.final_agent_trust.items()
        }
        final_track_trust = {
            str(track_id): float(score)
            for track_id, score in result.final_track_trust.items()
        }
        agent_trust_history = {
            str(agent_id): [float(x) for x in history]
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
