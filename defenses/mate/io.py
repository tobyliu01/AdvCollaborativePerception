from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .types import CAVFramePrediction, FrameData, ScenarioData


def load_scenarios_from_pickle(path: str | Path) -> List[ScenarioData]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return parse_scenarios(raw)


def parse_scenarios(raw: Any) -> List[ScenarioData]:
    if isinstance(raw, list):
        scenarios: List[ScenarioData] = []
        for idx, item in enumerate(raw):
            scenarios.append(_parse_single_scenario(item, fallback_id=f"scenario_{idx}"))
        return scenarios

    if isinstance(raw, dict):
        if "frames" in raw:
            return [_parse_single_scenario(raw, fallback_id=str(raw.get("scenario_id", "scenario_0")))]
        scenarios = []
        for sid, scenario_raw in raw.items():
            scenarios.append(_parse_single_scenario(scenario_raw, fallback_id=str(sid)))
        return scenarios

    raise TypeError(f"Unsupported root type for scenario parsing: {type(raw)!r}")


def _parse_single_scenario(raw: Any, fallback_id: str) -> ScenarioData:
    if isinstance(raw, ScenarioData):
        return raw
    if isinstance(raw, list):
        frames = [
            _parse_frame(frame_payload, fallback_frame_id=idx)
            for idx, frame_payload in enumerate(raw)
        ]
        return ScenarioData(scenario_id=fallback_id, frames=frames)
    if not isinstance(raw, dict):
        raise TypeError(f"Scenario must be dict-like, got {type(raw)!r}")

    scenario_id = str(raw.get("scenario_id", fallback_id))
    if "frames" in raw:
        frames_raw = raw["frames"]
        if isinstance(frames_raw, dict):
            frame_items = sorted(frames_raw.items(), key=lambda kv: _to_frame_sort_key(kv[0]))
        else:
            frame_items = list(enumerate(frames_raw))
    else:
        frame_items = sorted(raw.items(), key=lambda kv: _to_frame_sort_key(kv[0]))

    frames: List[FrameData] = []
    for frame_fallback_id, frame_payload in frame_items:
        frame = _parse_frame(frame_payload, fallback_frame_id=int(_to_frame_sort_key(frame_fallback_id)))
        frames.append(frame)

    return ScenarioData(scenario_id=scenario_id, frames=frames)


def _parse_frame(raw: Any, fallback_frame_id: int) -> FrameData:
    if isinstance(raw, FrameData):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Frame must be dict-like, got {type(raw)!r}")

    frame_id = int(raw.get("frame_id", fallback_frame_id))
    gt_bboxes = _extract_first_array(raw, ("gt_bboxes", "gt_boxes", "gt_bbox", "object_bbx_center", "gt"))
    gt_ids = _extract_first_array(raw, ("gt_ids", "object_ids", "gt_object_ids"), allow_missing=True)
    cavs_raw = _extract_cavs(raw)
    cavs = {agent_id: _parse_cav_prediction(agent_payload) for agent_id, agent_payload in cavs_raw.items()}

    return FrameData(
        frame_id=frame_id,
        gt_bboxes=_to_boxes(gt_bboxes),
        gt_ids=None if gt_ids is None else np.asarray(gt_ids, dtype=np.int64),
        cavs=cavs,
    )


def _extract_cavs(frame_raw: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    if "cavs" in frame_raw and isinstance(frame_raw["cavs"], dict):
        return frame_raw["cavs"]

    cavs: Dict[Any, Dict[str, Any]] = {}
    for key, value in frame_raw.items():
        if isinstance(value, dict) and any(k in value for k in ("pred_bboxes", "pred_boxes", "boxes")):
            cavs[key] = value
    return cavs


def _parse_cav_prediction(raw: Dict[str, Any]) -> CAVFramePrediction:
    pred_bboxes = _extract_first_array(raw, ("pred_bboxes", "pred_boxes", "boxes"))
    pred_scores = _extract_first_array(raw, ("pred_scores", "scores", "conf"), allow_missing=True)
    pose = _extract_first_array(raw, ("pose", "lidar_pose", "ego_lidar_pose"), allow_missing=True)

    boxes_in_global = bool(raw.get("boxes_in_global", raw.get("pred_boxes_in_global", True)))
    return CAVFramePrediction(
        pred_bboxes=_to_boxes(pred_bboxes),
        pred_scores=None if pred_scores is None else np.asarray(pred_scores, dtype=np.float32),
        pose=None if pose is None else np.asarray(pose, dtype=np.float32),
        boxes_in_global=boxes_in_global,
    )


def _extract_first_array(
    raw: Dict[str, Any],
    keys: Iterable[str],
    allow_missing: bool = False,
) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    if allow_missing:
        return None
    raise KeyError(f"Could not find any of keys {tuple(keys)!r} in payload.")


def _to_boxes(arr: Any) -> np.ndarray:
    if arr is None:
        return np.empty((0, 7), dtype=np.float32)
    boxes = np.asarray(arr, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"Expected boxes shape (N,7+), got {boxes.shape}")
    return boxes[:, :7]


def _to_frame_sort_key(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            return int(digits)
    return 0
