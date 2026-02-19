from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mvp.data.util import bbox_map_to_sensor, get_point_indices_in_bbox


def apply_attack_to_lidar(
    lidar: np.ndarray,
    attack_entry: Optional[dict],
) -> np.ndarray:
    """
    Reconstruct attacked LiDAR from clean LiDAR and saved attack_info entry.

    The attack_info format follows mvp/attack/lidar_shift_early_attacker.py.
    """
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
        m = min(replace_indices.shape[0], replace_data.shape[0])
        idx = replace_indices[:m]
        rep = replace_data[:m, :3]
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


def _prepare_bev_points(
    lidar: np.ndarray,
    z_min: float,
    z_max: float,
    range_max: float,
) -> np.ndarray:
    arr = np.asarray(lidar, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    z_mask = (arr[:, 2] >= float(z_min)) & (arr[:, 2] <= float(z_max))
    xy = arr[finite_mask & z_mask, :2]
    if xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if range_max > 0:
        r = np.linalg.norm(xy, axis=1)
        xy = xy[r <= float(range_max)]
    return xy.astype(np.float32)


def _fill_missing_ranges_cyclic(rng_out: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = rng_out.copy()
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.shape[0] == 0:
        return out
    invalid_idx = np.where(~valid_mask)[0]
    n = out.shape[0]
    for idx in invalid_idx:
        d = np.abs(valid_idx - idx)
        d = np.minimum(d, n - d)
        out[idx] = out[valid_idx[np.argmin(d)]]
    return out


def _polygon_from_polar(azimuth: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    x = ranges * np.cos(azimuth)
    y = ranges * np.sin(azimuth)
    poly = np.stack([x, y], axis=1).astype(np.float32)
    finite = np.isfinite(poly[:, 0]) & np.isfinite(poly[:, 1])
    poly = poly[finite]
    if poly.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)
    return poly


def estimate_fov_polygon_fast(
    lidar: np.ndarray,
    z_min: float,
    z_max: float,
    range_max: float,
    n_azimuth_bins: int = 1000,
    n_range_bins: int = 1000,
) -> np.ndarray:
    bev = _prepare_bev_points(
        lidar=lidar,
        z_min=z_min,
        z_max=z_max,
        range_max=range_max,
    )
    if bev.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)

    az = np.arctan2(bev[:, 1], bev[:, 0])
    rng = np.linalg.norm(bev, axis=1)
    valid = np.isfinite(az) & np.isfinite(rng) & (rng > 1e-6)
    az = az[valid]
    rng = rng[valid]
    if az.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)

    az_edges = np.linspace(-np.pi, np.pi, num=int(n_azimuth_bins) + 1, dtype=np.float32)
    range_upper = min(float(range_max), float(np.max(rng)))
    if range_upper <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)
    rng_edges = np.linspace(0.0, range_upper, num=int(n_range_bins) + 1, dtype=np.float32)

    hist, _, _ = np.histogram2d(az, rng, bins=[az_edges, rng_edges])
    row_has = hist > 0
    idx_last = np.full((int(n_azimuth_bins),), -1, dtype=np.int64)
    for i in range(int(n_azimuth_bins)):
        nz = np.where(row_has[i])[0]
        if nz.shape[0] > 0:
            idx_last[i] = int(nz[-1])
    valid_rows = idx_last >= 0
    if not np.any(valid_rows):
        return np.empty((0, 2), dtype=np.float32)

    rng_out = np.zeros((int(n_azimuth_bins),), dtype=np.float32)
    rng_out[valid_rows] = rng_edges[idx_last[valid_rows] + 1]
    rng_out = _fill_missing_ranges_cyclic(rng_out, valid_rows)
    az_out = 0.5 * (az_edges[:-1] + az_edges[1:])

    return _polygon_from_polar(az_out, rng_out)


def estimate_fov_polygon_slow(
    lidar: np.ndarray,
    z_min: float,
    z_max: float,
    range_max: float,
    n_azimuth_samples: int = 1000,
    az_tolerance: float = 0.0,
) -> np.ndarray:
    bev = _prepare_bev_points(
        lidar=lidar,
        z_min=z_min,
        z_max=z_max,
        range_max=range_max,
    )
    if bev.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)

    az = np.arctan2(bev[:, 1], bev[:, 0])
    rng = np.linalg.norm(bev, axis=1)
    valid = np.isfinite(az) & np.isfinite(rng) & (rng > 1e-6)
    az = az[valid]
    rng = rng[valid]
    if az.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)

    az_queries = np.linspace(-np.pi, np.pi, num=int(n_azimuth_samples), endpoint=False, dtype=np.float32)
    rng_out = np.zeros((int(n_azimuth_samples),), dtype=np.float32)
    valid_mask = np.zeros((int(n_azimuth_samples),), dtype=bool)

    for i, az_q in enumerate(az_queries):
        d = np.abs(((az - az_q + np.pi) % (2.0 * np.pi)) - np.pi)
        m = d <= float(az_tolerance)
        if np.any(m):
            rng_out[i] = float(np.max(rng[m]))
            valid_mask[i] = True

    if not np.any(valid_mask):
        return np.empty((0, 2), dtype=np.float32)

    rng_out = _fill_missing_ranges_cyclic(rng_out, valid_mask)
    return _polygon_from_polar(az_queries, rng_out)


def point_in_polygon_strict(point_xy: np.ndarray, polygon_xy: np.ndarray, eps: float = 1e-7) -> bool:
    """
    Strict point-in-polygon in 2D.
    Returns False when the point lies exactly on an edge (boundary is treated as out-of-range).
    """
    p = np.asarray(point_xy, dtype=np.float64).reshape(-1)
    poly = np.asarray(polygon_xy, dtype=np.float64)
    if p.shape[0] < 2 or poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return False
    x, y = float(p[0]), float(p[1])
    n = poly.shape[0]

    # Boundary check first.
    for i in range(n):
        x1, y1 = float(poly[i, 0]), float(poly[i, 1])
        x2, y2 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        vx = x2 - x1
        vy = y2 - y1
        wx = x - x1
        wy = y - y1
        cross = vx * wy - vy * wx
        if abs(cross) <= eps:
            dot = wx * vx + wy * vy
            if dot >= -eps:
                seg_len2 = vx * vx + vy * vy
                if dot <= seg_len2 + eps:
                    return False

    # Ray casting (strict interior).
    inside = False
    for i in range(n):
        x1, y1 = float(poly[i, 0]), float(poly[i, 1])
        x2, y2 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return bool(inside)


def point_count_in_gt_bbox(
    target_bbox_map: np.ndarray,
    cav_pose: np.ndarray,
    points_sensor: np.ndarray,
) -> int:
    pts = np.asarray(points_sensor, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return 0

    try:
        target_local = bbox_map_to_sensor(
            np.asarray(target_bbox_map, dtype=np.float32),
            np.asarray(cav_pose, dtype=np.float32),
        )
    except Exception:
        return 0

    bbox_local = np.asarray(target_local, dtype=np.float32).reshape(-1)
    if bbox_local.shape[0] < 7:
        return 0

    try:
        point_indices = get_point_indices_in_bbox(bbox_local, pts)
    except Exception:
        return 0
    return int(np.asarray(point_indices).reshape(-1).shape[0])


def resolve_point_count_visibility_overrides(
    gt_ids: np.ndarray,
    gt_bboxes_map: np.ndarray,
    filtered_unmatched_gt_indices: List[int],
    status_by_track_id: Dict[int, dict],
    cav_pose: np.ndarray,
    points_sensor: np.ndarray,
    point_threshold: int,
    forced_track_id: Optional[int] = None,
) -> Tuple[Dict[int, bool], Dict[int, int], List[int]]:
    """
    Resolve visibility overrides for filtered unmatched tracks using point-count checks.

    Returns:
      - visibility_override_by_track_id: track_id -> visible(bool)
      - point_count_by_track_id: track_id -> count (only tracks checked by point-count)
      - forced_penalized_track_ids: track ids that bypass point-count and are forced visible
    """
    visibility_override_by_track_id: Dict[int, bool] = {}
    point_count_by_track_id: Dict[int, int] = {}
    forced_penalized_track_ids: List[int] = []
    threshold = int(point_threshold)

    for gt_idx in filtered_unmatched_gt_indices:
        track_id = int(gt_ids[int(gt_idx)])
        status = status_by_track_id.get(track_id, {})
        out_of_distance_range = bool(
            status.get("out_of_range_120m", False)
            or status.get("out_of_model_range", False)
        )
        if out_of_distance_range:
            visibility_override_by_track_id[track_id] = False
            continue

        if forced_track_id is not None and track_id == int(forced_track_id):
            visibility_override_by_track_id[track_id] = True
            forced_penalized_track_ids.append(track_id)
            continue

        point_count = point_count_in_gt_bbox(
            target_bbox_map=gt_bboxes_map[int(gt_idx)],
            cav_pose=cav_pose,
            points_sensor=points_sensor,
        )
        point_count_by_track_id[track_id] = int(point_count)
        visibility_override_by_track_id[track_id] = (point_count > threshold)

    return (
        visibility_override_by_track_id,
        point_count_by_track_id,
        forced_penalized_track_ids,
    )
