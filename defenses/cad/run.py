from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Tuple

from mvp.tools.squeezeseg.interface import SqueezeSegInterface

from .common import strip_metric_geometry
from .config import (
    CAD_OCCUPANCY_MAP_ROOT,
    CAD_RESULT_ROOT,
    CAD_VISUALIZATION_ROOT,
    CONFLICT_AREA_THRESHOLD,
    DEBUG_CASE_IDS,
    DEBUG_MODE,
    DEBUG_PAIR_IDS,
    LOAD_CACHED_OCCUPANCY_MAP,
    TOTAL_FRAMES,
)
from .evaluation import build_attack_gt_bbox_by_frame, defense_evaluation
from .occupancy import (
    load_cached_occupancy_feature,
    occupancy_map,
    save_fused_occupancy_map_frame,
)
from .perception_defender import CADPerceptionDefender
from .visualization import save_fused_occupancy_visualization


def defense(
    *,
    occupancy_feature: List[Dict[Any, Dict[str, Any]]],
    frame_ids: Sequence[int],
    conflict_area_threshold: float,
    logger: Any,
    case_id: int,
    pair_id: int,
) -> List[Dict[str, Any]]:
    defender = CADPerceptionDefender(conflict_area_threshold=conflict_area_threshold)
    metrics = defender.run(
        occupancy_feature=occupancy_feature,
        frame_ids=list(frame_ids),
    )

    for frame_id in frame_ids:
        frame_metric = metrics[frame_id]
        logger.info(
            "[CAD] case %06d pair %02d frame %02d conflicted=%s conflicted_regions=%d conflicted_area=%.4f",
            int(case_id),
            int(pair_id),
            int(frame_id),
            bool(frame_metric.get("conflicted", False)),
            int(frame_metric.get("conflicted_count", 0)),
            float(frame_metric.get("conflicted_area_total", 0.0)),
        )
        if DEBUG_MODE:
            logger.info(
                "[CAD_DEBUG] case %06d pair %02d frame %02d conflict_pair_details=%s",
                int(case_id),
                int(pair_id),
                int(frame_id),
                str(frame_metric.get("conflict_pair_details", [])),
            )
    return metrics


def run_cad_attack_evaluation(
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
    del perception_name

    model_result_root = os.path.join(CAD_RESULT_ROOT, default_shift_model)
    model_visualization_root = os.path.join(CAD_VISUALIZATION_ROOT, default_shift_model)
    model_occupancy_root = os.path.join(CAD_OCCUPANCY_MAP_ROOT, default_shift_model)
    os.makedirs(model_result_root, exist_ok=True)
    os.makedirs(model_visualization_root, exist_ok=True)
    os.makedirs(model_occupancy_root, exist_ok=True)

    frame_ids = sorted(set(int(frame_id) for frame_id in attack_frame_ids if 0 <= int(frame_id) < TOTAL_FRAMES))
    if len(frame_ids) == 0:
        frame_ids = list(range(TOTAL_FRAMES))

    logger.info(
        "[CAD] Evaluating defense cad for attack %s on mesh model %s",
        attacker.name,
        default_shift_model,
    )
    logger.info(
        "[CAD] DEBUG_MODE=%s DEBUG_CASE_IDS=%s DEBUG_PAIR_IDS=%s",
        bool(DEBUG_MODE),
        sorted(list(DEBUG_CASE_IDS)),
        sorted(list(DEBUG_PAIR_IDS)),
    )

    combination_groups: "OrderedDict[Tuple[int, int], List[Any]]" = OrderedDict()
    for ego_attack in attacker.attack_list:
        attack_meta = ego_attack.get("attack_meta", {})
        case_id = int(attack_meta["case_id"])
        pair_id = int(attack_meta["pair_id"])
        if DEBUG_MODE:
            if case_id not in DEBUG_CASE_IDS or pair_id not in DEBUG_PAIR_IDS:
                continue
        combination_groups.setdefault((case_id, pair_id), []).append(ego_attack)

    if len(combination_groups) == 0:
        logger.warning("[CAD] No attack combinations found for attacker %s.", attacker.name)
        return

    lidar_seg_api = SqueezeSegInterface()
    case_cache: Dict[int, Any] = {}
    case_pair_metrics: "OrderedDict[Tuple[int, int], List[Dict[str, Any]]]" = OrderedDict()
    case_pair_attack_bbox_by_frame: "OrderedDict[Tuple[int, int], Dict[int, Any]]" = OrderedDict()

    for (case_id, pair_id), attack_group in combination_groups.items():
        try:
            if case_id not in case_cache:
                case_cache[case_id] = dataset.get_case(
                    case_id,
                    tag="multi_frame",
                    use_lidar=True,
                    use_camera=False,
                )
            case = case_cache[case_id]
        except Exception as exc:
            logger.warning(
                "[CAD] Skipping case %06d pair %02d: failed to load case (%s).",
                int(case_id),
                int(pair_id),
                str(exc),
            )
            continue

        attack_by_ego: "OrderedDict[Any, Any]" = OrderedDict()
        for ego_attack in attack_group:
            ego_id = ego_attack.get("attack_meta", {}).get("ego_vehicle_id")
            attack_by_ego[ego_id] = ego_attack

        pair_meta = attack_group[0].get("attack_meta", {})
        victim_vehicle_id = pair_meta.get("victim_vehicle_id")
        cav_ids = list(pair_meta.get("vehicle_ids", []))
        if len(cav_ids) == 0 and len(case) > 0:
            cav_ids = sorted(case[0].keys(), key=lambda x: str(x))

        logger.info(
            "[CAD] Processing case %06d pair %02d with CAVs %s",
            int(case_id),
            int(pair_id),
            str(cav_ids),
        )

        if LOAD_CACHED_OCCUPANCY_MAP:
            logger.info(
                "[CAD] Load cached occupancy map: model=%s case=%06d pair=%02d",
                default_shift_model,
                int(case_id),
                int(pair_id),
            )
            occupancy_feature = load_cached_occupancy_feature(
                default_shift_model=default_shift_model,
                case_id=case_id,
                pair_id=pair_id,
                frame_ids=frame_ids,
            )
        else:
            occupancy_feature = occupancy_map(
                case_id=case_id,
                pair_id=pair_id,
                case=case,
                attack_by_ego=attack_by_ego,
                cav_ids=cav_ids,
                frame_ids=frame_ids,
                result_dir=result_dir,
                perception_model_name=perception_model_name,
                default_shift_model=default_shift_model,
                pickle_cache_load=pickle_cache_load,
                lidar_seg_api=lidar_seg_api,
                logger=logger,
            )

        metrics = defense(
            occupancy_feature=occupancy_feature,
            frame_ids=frame_ids,
            conflict_area_threshold=CONFLICT_AREA_THRESHOLD,
            logger=logger,
            case_id=case_id,
            pair_id=pair_id,
        )

        attack_bbox_by_frame = build_attack_gt_bbox_by_frame(
            case=case,
            attack_by_ego=attack_by_ego,
            frame_ids=frame_ids,
            default_shift_model=default_shift_model,
            victim_vehicle_id=victim_vehicle_id,
        )

        for frame_id in frame_ids:
            save_fused_occupancy_map_frame(
                default_shift_model=default_shift_model,
                case_id=case_id,
                pair_id=pair_id,
                frame_id=frame_id,
                frame_occupancy=occupancy_feature[frame_id],
                frame_metric=metrics[frame_id],
            )
            save_fused_occupancy_visualization(
                visualization_root=model_visualization_root,
                case_id=case_id,
                pair_id=pair_id,
                frame_id=frame_id,
                frame_occupancy=occupancy_feature[frame_id],
                frame_metric=metrics[frame_id],
                logger=logger,
            )

        metric_to_save = [strip_metric_geometry(metric) for metric in metrics]
        save_dir = os.path.join(
            model_result_root,
            "case{:06d}".format(int(case_id)),
            "pair{:02d}".format(int(pair_id)),
        )
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "metric.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(metric_to_save, f)
        logger.info("[CAD] Saved metric file: %s", save_file)

        case_pair_metrics[(case_id, pair_id)] = metrics
        case_pair_attack_bbox_by_frame[(case_id, pair_id)] = attack_bbox_by_frame

    summary = defense_evaluation(
        case_pair_metrics=case_pair_metrics,
        case_pair_attack_bbox_by_frame=case_pair_attack_bbox_by_frame,
        frame_ids=frame_ids,
        logger=logger,
    )
    summary_file = os.path.join(
        model_result_root,
        "summary_{}_{}.pkl".format(default_shift_model, attacker.name),
    )
    with open(summary_file, "wb") as f:
        pickle.dump(summary, f)
    logger.info(
        "[CAD] Occupancy anomaly frames: %d/%d, anomaly success frames: %d/%d, success_rate=%.6f",
        summary["anomaly_frames"],
        summary["total_frames"],
        summary["success_frames"],
        summary["total_frames"],
        summary["success_rate"],
    )
