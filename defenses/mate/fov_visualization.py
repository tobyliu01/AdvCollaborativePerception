from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


def _as_boxes(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 7:
        return np.empty((0, 7), dtype=np.float32)
    return arr[:, :7]


def _bbox_corners_2d(box: np.ndarray) -> np.ndarray:
    x, y, _, l, w, _, yaw = [float(v) for v in box[:7]]
    dx = l * 0.5
    dy = w * 0.5
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    local = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]],
        dtype=np.float32,
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = local @ rot.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def save_fov_visualization(
    output_root: str,
    case_id: int,
    pair_id: int,
    frame_id: int,
    cav_id: Any,
    lidar_local: np.ndarray,
    pred_boxes_local: np.ndarray,
    fov_polygon_fast: Optional[np.ndarray],
    fov_polygon_slow: Optional[np.ndarray],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lidar = np.asarray(lidar_local, dtype=np.float32)
    if lidar.ndim != 2 or lidar.shape[1] < 2:
        lidar = np.empty((0, 2), dtype=np.float32)
    boxes = _as_boxes(pred_boxes_local)
    poly_fast = np.asarray(fov_polygon_fast, dtype=np.float32) if fov_polygon_fast is not None else np.empty((0, 2), dtype=np.float32)
    poly_slow = np.asarray(fov_polygon_slow, dtype=np.float32) if fov_polygon_slow is not None else np.empty((0, 2), dtype=np.float32)

    save_dir = os.path.join(
        output_root,
        "case{:06d}".format(case_id),
        "pair{:02d}".format(pair_id),
        str(cav_id),
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "frame{:02d}.png".format(frame_id))

    if lidar.shape[0] > 60000:
        idx = np.random.choice(lidar.shape[0], 60000, replace=False)
        lidar_plot = lidar[idx]
    else:
        lidar_plot = lidar

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=120)
    if lidar_plot.shape[0] > 0:
        ax.scatter(lidar_plot[:, 0], lidar_plot[:, 1], s=0.4, c="black", alpha=0.45, linewidths=0)

    if poly_fast.ndim == 2 and poly_fast.shape[0] >= 3:
        poly = np.vstack([poly_fast, poly_fast[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="tab:blue", linewidth=1.2, label="FOV fast")
    if poly_slow.ndim == 2 and poly_slow.shape[0] >= 3:
        poly = np.vstack([poly_slow, poly_slow[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="tab:orange", linewidth=1.0, linestyle="--", label="FOV slow")

    for box in boxes:
        c2d = _bbox_corners_2d(box)
        ax.plot(c2d[:, 0], c2d[:, 1], color="tab:red", linewidth=1.0)

    ax.scatter([0.0], [0.0], c="tab:green", s=16, label="CAV")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.set_title(
        "case{:06d} pair{:02d} cav{} frame{:02d}".format(
            case_id, pair_id, str(cav_id), frame_id
        )
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_file)
    plt.close(fig)
