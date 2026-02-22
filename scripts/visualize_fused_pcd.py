import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(ROOT)

from mvp.config import data_root
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import pcd_sensor_to_map
from mvp.visualize.lidar_3d_export import save_ascii_ply_xyzi


CASE_ID = 0
PAIR_ID = 0
FRAME_IDX = 0
DEFAULT_SHIFT_MODEL = "adv_real_car_with_plane_victim_03"

RESULT_DIR = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result"
PERCEPTION_NAME = "pixor_early"


def read_ascii_ply_xyzi(ply_path):
    """Read ASCII PLY with at least x,y,z and optional intensity."""
    with open(ply_path, "r") as f:
        lines = f.readlines()

    header_end = None
    vertex_count = None
    prop_names = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            vertex_count = int(s.split()[-1])
        elif s.startswith("property"):
            toks = s.split()
            if len(toks) >= 3:
                prop_names.append(toks[-1])
        elif s == "end_header":
            header_end = i + 1
            break

    if header_end is None or vertex_count is None:
        raise RuntimeError(f"Invalid PLY format: {ply_path}")

    data = np.loadtxt(lines[header_end: header_end + vertex_count], dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    col = {name: idx for idx, name in enumerate(prop_names)}
    if not {"x", "y", "z"}.issubset(col.keys()):
        raise RuntimeError(f"PLY missing xyz fields: {ply_path}")

    xyz = np.stack(
        [data[:, col["x"]], data[:, col["y"]], data[:, col["z"]]],
        axis=1,
    )
    if "intensity" in col:
        intensity = data[:, col["intensity"]][:, None]
    else:
        intensity = np.ones((xyz.shape[0], 1), dtype=np.float32) * 0.1
    return np.hstack([xyz, intensity]).astype(np.float32)


def main():
    pair_dir = os.path.join(
        RESULT_DIR,
        "attack",
        PERCEPTION_NAME.split("_")[0],  # e.g. "pixor"
        DEFAULT_SHIFT_MODEL,
        f"case{CASE_ID:06d}",
        f"pair{PAIR_ID:02d}",
    )
    if not os.path.isdir(pair_dir):
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")
    case = dataset.get_case(CASE_ID, tag="multi_frame", use_lidar=True, use_camera=False)
    frame_case = case[FRAME_IDX]

    fused_map_xyzi = []
    vehicle_dirs = sorted(
        [
            x
            for x in os.listdir(pair_dir)
            if os.path.isdir(os.path.join(pair_dir, x)) and x.isdigit()
        ]
    )

    for vehicle_dir in vehicle_dirs:
        vehicle_id = int(vehicle_dir)
        frame_dir = os.path.join(pair_dir, vehicle_dir, f"frame{FRAME_IDX}")
        pcd_file = os.path.join(frame_dir, f"{PERCEPTION_NAME}.ply")
        if not os.path.isfile(pcd_file):
            continue
        if vehicle_id not in frame_case:
            continue

        sensor_xyzi = read_ascii_ply_xyzi(pcd_file)
        lidar_pose = frame_case[vehicle_id]["lidar_pose"]
        map_xyzi = pcd_sensor_to_map(sensor_xyzi, lidar_pose)
        fused_map_xyzi.append(map_xyzi)

    if len(fused_map_xyzi) == 0:
        raise RuntimeError(
            "No saved PLY files found for this case/pair/frame under "
            f"{pair_dir}/*/frame{FRAME_IDX}/{PERCEPTION_NAME}.ply"
        )

    fused_map_xyzi = np.vstack(fused_map_xyzi)
    out_name = (
        f"fused_case{CASE_ID:02d}_pair{PAIR_ID:02d}_frame{FRAME_IDX:02d}.ply"
    )
    out_path = os.path.join(os.getcwd(), out_name)
    save_ascii_ply_xyzi(out_path, fused_map_xyzi)
    print(f"Saved fused map-frame point cloud: {out_path}")
    print(f"Number of points: {fused_map_xyzi.shape[0]}")


if __name__ == "__main__":
    main()
