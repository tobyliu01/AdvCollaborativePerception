import os
from pathlib import Path
from typing import Any, Optional, List
import numpy as np

# ---------- Pose utilities (ZYX: roll->pitch->yaw, but we rotate by yaw,pitch,roll w.r.t. map z,y,x) ----------

def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def _roty(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float64)

def _rotx(roll: float) -> np.ndarray:
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]], dtype=np.float64)

def _pose_to_T(pose: Any) -> Optional[np.ndarray]:
    """
    Convert lidar_pose to 4x4 homogeneous transform.
    Accepted formats:
      - 4x4 matrix
      - [x, y, z, roll, pitch, yaw]
      - [x, y, yaw]  (assumes z=0, roll=pitch=0)
    Return None -> points assumed already in map frame.
    """
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=float)

    if pose.shape == (4, 4):
        return pose

    if pose.size >= 6:  # x,y,z,roll,pitch,yaw
        x, y, z, roll, pitch, yaw = pose[:6]
        R = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
    elif pose.size == 3:  # x,y,yaw -> ground assumption
        x, y, yaw = pose
        z = 0.0
        R = _rotz(yaw)
    else:
        return None

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

# ---------- Public helpers you will call from draw_attack() ----------

def lidar_to_map_xyz(lidar_xyz: np.ndarray, lidar_pose: Any) -> np.ndarray:
    """
    Input:
      lidar_xyz : (N, >=3) array in sensor frame
      lidar_pose: one of the accepted pose formats (above)
    Output:
      xyz_map   : (N, 3) array transformed into map/world frame
    """
    if lidar_xyz is None:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.asarray(lidar_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    T = _pose_to_T(lidar_pose)
    if T is None:
        # Assume already in map frame
        return pts[:, :3]

    N = pts.shape[0]
    homo = np.ones((N, 4), dtype=np.float64)
    homo[:, :3] = pts[:, :3]
    out = (T @ homo.T).T
    return out[:, :3]

def save_ascii_ply_xyz(path: str, xyz_map: np.ndarray) -> str:
    """
    Write an ASCII .ply (x y z) that MeshLab can open.
    """
    xyz = np.asarray(xyz_map, dtype=float)
    n = xyz.shape[0]
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    return path


def save_ascii_ply_xyzi(path: str, xyzi: np.ndarray) -> str:
    """
    Write an ASCII .ply with x y z intensity.
    """
    pts = np.asarray(xyzi, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        pts = np.zeros((0, 4), dtype=float)
    if pts.shape[1] < 4:
        intensity = np.ones((pts.shape[0], 1), dtype=float) * 0.1
        pts = np.hstack([pts[:, :3], intensity])

    n = pts.shape[0]
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float intensity\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z, intensity = pts[i, :4]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n")
    return path

def concat_xyz(list_of_xyz: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple (N_i, 3) arrays safely.
    """
    arrs = [np.asarray(a, dtype=float).reshape(-1, 3) for a in list_of_xyz if a is not None and a.size > 0]
    if not arrs:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(arrs)
