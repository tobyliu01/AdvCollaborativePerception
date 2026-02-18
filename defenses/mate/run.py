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
    logger: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(prediction_file):
        logger.warning(
            "Prediction file missing for CAV %s frame %02d: %s",
            str(agent_id),
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
            str(agent_id),
            frame_id,
            prediction_file,
        )
        return np.empty((0, 7), dtype=np.float32), np.empty((0,), dtype=np.float32)
    pred_boxes = _as_boxes(agent_data.get("pred_bboxes"))
    pred_scores = _align_scores(agent_data.get("pred_scores"), pred_boxes.shape[0])
    if pred_boxes.shape[0] == 0:
        logger.warning(
            "Prediction empty for CAV %s frame %02d in file: %s",
            str(agent_id),
            frame_id,
            prediction_file,
        )
    return pred_boxes, pred_scores


def _frame_has_valid_attack(
    attack_info: Any,
    frame_id: int,
    agent_id: Any,
) -> bool:
    frame_data = _frame_payload(attack_info, frame_id)
    agent_info = _dict_get(frame_data, agent_id, {})
    return isinstance(agent_info, dict) and len(agent_info) > 0


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
        if case_id not in DEBUG_CASE_IDS or pair_id not in DEBUG_PAIR_IDS:
            continue
        key = (case_id, pair_id)
        combination_groups.setdefault(key, []).append(ego_attack)

    if len(combination_groups) == 0:
        logger.warning("No attack combinations found for attacker %s.", attacker.name)
        return

    case_cache: dict[int, Any] = {}
    mate_config = MATEConfig(
        penalize_unmatched_predictions=False,
    )
    mate_estimator = MATEEstimator(config=mate_config)

    # Iterate all scenarios.
    for (case_id, pair_id), attack_group in combination_groups.items():
        # Fetch necessary information of each pair.
        pair_meta = attack_group[0]["attack_meta"]
        victim_vehicle_id = pair_meta.get("victim_vehicle_id")
        object_id = pair_meta.get("object_id")

        try:
            if case_id not in case_cache:
                case_cache[case_id] = dataset.get_case(
                    case_id,
                    tag="multi_frame",
                    use_lidar=False,
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
                if m <= 0:
                    cav_gt_ids_by_vehicle[frame_cav_id] = []
                    continue
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[:m]
                local_object_ids = local_object_ids[:m]

                # Remove ego vehicle ID from all track IDs.
                ego_id_removal_mask = np.asarray(
                    [str(obj_id) != str(frame_cav_id) for obj_id in local_object_ids],
                    dtype=bool,
                )
                local_gt_bboxes_sensor = local_gt_bboxes_sensor[ego_id_removal_mask]
                local_object_ids = [
                    obj_id for obj_id, keep in zip(local_object_ids, ego_id_removal_mask) if keep
                ]

                # Transform bbox in local vehicle frame to map frame
                frame_vehicle_pose = np.asarray(
                    frame_vehicle_data["lidar_pose"], dtype=np.float32
                )
                local_gt_bboxes_map = (
                    bbox_sensor_to_map(local_gt_bboxes_sensor, frame_vehicle_pose)
                    if local_gt_bboxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                # Build a list of all track IDs of the current CAV. Also add the track ID and bbox in aggregator list.
                cav_gt_ids: list[int] = []
                for local_obj_id, local_gt_bbox_map in zip(local_object_ids, local_gt_bboxes_map):
                    local_object_id = int(local_obj_id)
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
                aggregator_gt_ids_list = [int(x) for x in gt_ids.tolist()]
                debug_payload = {
                    "gt_track_ids": gt_track_ids_list,
                    "aggregator_gt_ids": aggregator_gt_ids_list,
                    "unmatched_pred_used_for_trust": bool(mate_config.penalize_unmatched_predictions),
                    "matched_track_ids": [],
                    "matched_pairs": [],
                    "unmatched_pred_indices": [],
                    "unmatched_gt_indices": [],
                    "unmatched_aggregator_gt_ids": [],
                    "filtered_unmatched_gt_indices": [],
                    "filtered_unmatched_gt_ids": [],
                    "status": "ok",
                }

                # Handle dataset exceptions.
                if cav_id not in frame_case:
                    logger.warning(
                        "CAV %s not found in case %06d pair %02d frame %02d.",
                        str(cav_id),
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue
                if attack_by_ego.get(cav_id) is None:
                    logger.warning(
                        "CAV %s has no attack entry in case %06d pair %02d frame %02d.",
                        str(cav_id),
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue

                # Fetch the lidar point attack info and test if it's a valid attack.
                vehicle_dir = os.path.join(
                    result_dir,
                    "attack/{}/{}/case{:06d}/pair{:02d}/{}".format(
                        perception_model_name,
                        default_shift_model,
                        case_id,
                        pair_id,
                        str(cav_id),
                    ),
                )
                attack_info_file = os.path.join(vehicle_dir, "attack_info.pkl")
                if not os.path.isfile(attack_info_file):
                    logger.warning(
                        "Missing attack_info file for CAV %s at case %06d pair %02d frame %02d",
                        str(cav_id),
                        case_id,
                        pair_id,
                        frame_id,
                    )
                    continue
                attack_info = pickle_cache_load(attack_info_file)
                if not _frame_has_valid_attack(attack_info, frame_id, cav_id):
                    logger.warning(
                        "Invalid attack for CAV %s at case %06d pair %02d frame %02d",
                        str(cav_id),
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
                vehicle_pose = np.asarray(frame_case[cav_id]["lidar_pose"], dtype=np.float32)
                pred_boxes_map = (
                    bbox_sensor_to_map(pred_boxes_sensor, vehicle_pose)
                    if pred_boxes_sensor.shape[0] > 0
                    else np.empty((0, 7), dtype=np.float32)
                )

                # Get the matched/unmatched prediction bboxes according to the ground truth.
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
                debug_payload["matched_pairs"] = [
                    [int(li), int(ri)] for li, ri in assignment.matched_pairs
                ]
                debug_payload["unmatched_pred_indices"] = [int(x) for x in assignment.unmatched_left]
                debug_payload["unmatched_gt_indices"] = [int(x) for x in assignment.unmatched_right]
                unmatched_gt_ids = [int(gt_ids[idx]) for idx in assignment.unmatched_right]
                debug_payload["unmatched_aggregator_gt_ids"] = unmatched_gt_ids
                filtered_unmatched_gt_indices = [
                    int(idx) for idx in assignment.unmatched_right
                    if int(gt_ids[idx]) in gt_track_ids_set
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
                    str(cav_id),
                    json.dumps(debug_payload, sort_keys=True),
                )

                # Store the predictions of the current CAV.
                cavs[cav_id] = CAVFramePrediction(
                    pred_bboxes=pred_boxes_map,
                    pred_scores=pred_scores,
                    pose=vehicle_pose[:2],
                    visible_gt_ids=np.asarray(gt_track_ids_list, dtype=np.int64),
                    bboxes_in_global=True,
                )

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

        # Log results.
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
