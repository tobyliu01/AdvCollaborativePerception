import os
import pickle
import math
import numpy as np
from typing import Any, Dict, List

# =========================
# PARAMETERS (edit these)
# =========================
DATADIR = "../data/OPV2V"
DATASET = "test"
SCENARIO_ID = "2021_08_22_21_41_24"
START_FRAME_IDS = [69, 89, 109, 129, 149, 169, 189]
TARGET_VEHICLE_ID = 916
REFERENCE_VEHICLE_ID = 886
OUTPUT_PATH = "positions_output.pkl"
# =========================

def load_meta(datadir: str, dataset: str) -> Dict[str, Any]:
    pkl_path = os.path.join(datadir, f"{dataset}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def get_label(labels_frame: Dict[Any, Dict[str, Any]], vid: int) -> Dict[str, Any]:
    if vid in labels_frame:
        return labels_frame[vid]
    s = str(vid)
    if s in labels_frame:
        return labels_frame[s]
    raise KeyError(f"Vehicle id {vid} not found in this frame.")

def label_to_abs_bbx7(label: Dict[str, Any]) -> np.ndarray:
    """
    Convert OPV2V label to absolute [x,y,z,l,w,h,yaw] in MAP/WORLD coordinates.
    - location: center in meters (map/world)
    - extent: half sizes -> multiply by 2 for l,w,h
    - angle[1]: yaw in degrees -> radians
    """
    loc = label.get("location", [float("nan")] * 3)
    ext = label.get("extent",   [0.0, 0.0, 0.0])
    ang = label.get("angle",    [0.0, 0.0, 0.0])

    x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
    l = float(ext[0]) * 2.0
    w = float(ext[1]) * 2.0
    h = float(ext[2]) * 2.0
    yaw = math.radians(float(ang[1]))
    return np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

def angle_wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def rotation_matrix(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """
    Matches the rotation used in the codebase.
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([
        [cy*cp,              cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,              sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,                cp*sr,             cp*cr           ],
    ], dtype=np.float32)

def map_bbox_to_sensor_bbox(bbox_map: np.ndarray, lidar_pose: List[float]) -> np.ndarray:
    """
    Convert a global bbox [x,y,z,l,w,h,yaw] into the attacker's sensor frame.
    lidar_pose = [tx, ty, tz, roll_deg, yaw_deg, pitch_deg]
    """
    tx, ty, tz = float(lidar_pose[0]), float(lidar_pose[1]), float(lidar_pose[2])
    roll = math.radians(float(lidar_pose[3]))
    yaw  = math.radians(float(lidar_pose[4]))
    pitch= math.radians(float(lidar_pose[5]))

    R = rotation_matrix(roll, yaw, pitch)  # sensor->map
    t = np.array([tx, ty, tz], dtype=np.float32)

    # position: inverse rigid transform
    p_map = bbox_map[:3].astype(np.float32)
    p_sensor = R.T @ (p_map - t)

    # sizes unchanged
    l, w, h = bbox_map[3], bbox_map[4], bbox_map[5]

    # yaw relative to sensor's heading
    yaw_map = float(bbox_map[6])
    yaw_sensor = angle_wrap_pi(yaw_map - yaw)

    return np.array([p_sensor[0], p_sensor[1], p_sensor[2], l, w, h, yaw_sensor], dtype=np.float32)

def process_one_start(meta: Dict[str, Any], start_frame_id: int) -> Dict[str, Any]:
    """
    Returns a dict for a single start frame:
    { "scenario_id", "start_frame_id", "target_vehicle_id", "reference_vehicle_id", "positions": (10,7) }
    """
    label_frames = meta[SCENARIO_ID].get("label", {})
    data_frames  = meta[SCENARIO_ID].get("data",  {})

    # frames: start, start+2, ..., start+18 (10 frames)
    target_frames = [start_frame_id + 2 * i for i in range(10)]

    # Output (10,7): bbox of TARGET in the attacker's (REFERENCE) sensor frame
    positions_sensor = np.full((10, 7), np.nan, dtype=np.float32)

    for i, frame_id in enumerate(target_frames):
        frame_rec = data_frames.get(frame_id, {})
        rec_ref = frame_rec.get(REFERENCE_VEHICLE_ID, frame_rec.get(str(REFERENCE_VEHICLE_ID)))
        if not isinstance(rec_ref, dict):
            # missing attacker record in data for this frame
            continue
        # fetch lidar_pose from either calib or direct field
        calib_entry = rec_ref.get("calib", {})
        if isinstance(calib_entry, dict) and "lidar_pose" in calib_entry:
            lidar_pose = calib_entry["lidar_pose"]
        else:
            lidar_pose = rec_ref.get("lidar_pose", None)
        if lidar_pose is None:
            continue

        labels_frame = label_frames.get(frame_id)
        if labels_frame is None:
            continue
        try:
            label_target = get_label(labels_frame, TARGET_VEHICLE_ID)
            tgt_map = label_to_abs_bbx7(label_target)  # [x,y,z,l,w,h,yaw] in global

            tgt_in_attacker = map_bbox_to_sensor_bbox(tgt_map, lidar_pose)
            positions_sensor[i, :] = tgt_in_attacker

        except KeyError:
            # missing target label in this frame
            pass

    out_obj = {
        "scenario_id": SCENARIO_ID,
        "start_frame_id": start_frame_id,
        "target_vehicle_id": TARGET_VEHICLE_ID,
        "reference_vehicle_id": REFERENCE_VEHICLE_ID,
        "positions": positions_sensor,  # shape (10,7)
    }
    return out_obj

def main():
    meta = load_meta(DATADIR, DATASET)

    if SCENARIO_ID not in meta:
        sample_keys = list(meta.keys())[:10]
        raise KeyError(f"Scenario '{SCENARIO_ID}' not found. Examples: {sample_keys}")

    results: List[Dict[str, Any]] = []
    for start in START_FRAME_IDS:
        d = process_one_start(meta, start)
        results.append(d)
        print(f"Prepared entry for start={start} (frames {start}..{start+18} step2)")

    # Save all dicts (list) into one pickle file
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(results)} start-block(s) to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
