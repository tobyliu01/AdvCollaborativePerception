from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .config import CAD_OCCUPANCY_MAP_ROOT


def apply_attack_to_lidar(lidar: np.ndarray, attack_entry: Any) -> np.ndarray:
    base = np.asarray(lidar, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 3:
        return np.empty((0, 4), dtype=np.float32)
    pcd = base.copy()
    if not isinstance(attack_entry, dict) or len(attack_entry) == 0:
        return pcd

    replace_indices = np.asarray(attack_entry.get("replace_indices", []), dtype=np.int64).reshape(-1)
    replace_data = np.asarray(attack_entry.get("replace_data", []), dtype=np.float32)
    if replace_data.ndim == 1 and replace_data.size > 0:
        replace_data = replace_data.reshape(1, -1)
    if replace_data.ndim == 2 and replace_data.shape[0] > 0 and replace_indices.shape[0] > 0:
        valid_count = min(replace_indices.shape[0], replace_data.shape[0])
        idx = replace_indices[:valid_count]
        rep = replace_data[:valid_count, :3]
        valid = (idx >= 0) & (idx < pcd.shape[0])
        if np.any(valid):
            pcd[idx[valid], :3] = rep[valid]

    ignore_indices = np.asarray(attack_entry.get("ignore_indices", []), dtype=np.int64).reshape(-1)
    if ignore_indices.shape[0] > 0:
        valid = (ignore_indices >= 0) & (ignore_indices < pcd.shape[0])
        ignore_indices = ignore_indices[valid]
        if ignore_indices.shape[0] > 0:
            keep_mask = np.ones((pcd.shape[0],), dtype=bool)
            keep_mask[ignore_indices] = False
            pcd = pcd[keep_mask]

    append_data = attack_entry.get("append_data")
    if append_data is not None:
        append_xyz = np.asarray(append_data, dtype=np.float32)
        if append_xyz.ndim == 1 and append_xyz.size > 0:
            append_xyz = append_xyz.reshape(1, -1)
        if append_xyz.ndim == 2 and append_xyz.shape[0] > 0 and append_xyz.shape[1] >= 3:
            n_cols = pcd.shape[1]
            append_full = np.zeros((append_xyz.shape[0], n_cols), dtype=np.float32)
            append_full[:, :3] = append_xyz[:, :3]
            pcd = np.vstack([pcd, append_full])
    return pcd


def dict_get(dct: Any, key: Any, default: Any = None) -> Any:
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


def frame_payload(container: Any, frame_id: int) -> dict:
    if isinstance(container, list):
        if 0 <= frame_id < len(container):
            payload = container[frame_id]
            return payload if isinstance(payload, dict) else {}
        return {}
    if isinstance(container, dict):
        payload = dict_get(container, frame_id, {})
        return payload if isinstance(payload, dict) else {}
    return {}


def as_boxes(data: Any) -> np.ndarray:
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


def strip_metric_geometry(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metric.items()
        if key not in {"_fused_occupied_geom", "_fused_free_geom", "_conflicted_geoms"}
    }


def occupancy_map_case_pair_dir(default_shift_model: str, case_id: int, pair_id: int) -> str:
    return os.path.join(
        CAD_OCCUPANCY_MAP_ROOT,
        str(default_shift_model),
        str(int(case_id)),
        str(int(pair_id)),
    )
