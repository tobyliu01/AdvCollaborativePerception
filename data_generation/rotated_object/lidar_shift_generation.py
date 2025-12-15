import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lidar spoof attack config from test_attacks_unique.pkl")
    parser.add_argument("--attacks", type=Path, default=Path("test_attacks_unique.pkl"), help="Source attacks PKL")
    parser.add_argument("--output", type=Path, default=Path("lidar_shift.pkl"), help="Destination PKL")
    parser.add_argument("--datadir", type=Path, default=Path("/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/data/OPV2V"), help="Dataset root containing split PKL files")
    parser.add_argument("--dataset", choices=["train", "validate", "test"], default="test", help="Dataset split to load")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rotation_matrix(roll, yaw, pitch):
    return np.array([
        [np.cos(yaw)*np.cos(pitch),
         np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll),
         np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
        [np.sin(yaw)*np.cos(pitch),
         np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll),
         np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
        [-np.sin(pitch),
         np.cos(pitch)*np.sin(roll),
         np.cos(pitch)*np.cos(roll)]
    ])


def bbox_map_to_sensor(bbox: np.ndarray, sensor_calib: np.ndarray) -> np.ndarray:
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180.0
    new_bbox = np.copy(bbox)
    if new_bbox.ndim == 1:
        new_bbox[:3] -= sensor_location
        new_bbox[:3] = np.linalg.inv(rotation_matrix(*sensor_rotation)) @ new_bbox[:3]
        new_bbox[6] -= sensor_rotation[1]
    elif new_bbox.ndim == 2:
        new_bbox[:, :3] -= sensor_location
        new_bbox[:, :3] = (np.linalg.inv(rotation_matrix(*sensor_rotation)) @ new_bbox[:, :3].T).T
        new_bbox[:, 6] -= sensor_rotation[1]
    else:
        raise ValueError("bbox must be 1D or 2D array")
    return new_bbox


def rotate_bbox_toward_target(bbox_sensor: np.ndarray, target_xy_sensor: np.ndarray) -> np.ndarray:
    """Rotate bbox yaw so it faces the target position (already in ego frame)."""
    rotated = np.copy(bbox_sensor)
    vec_to_target = target_xy_sensor - rotated[:2]
    rotated[6] = np.arctan2(vec_to_target[1], vec_to_target[0])
    return rotated


def label_to_bbox(label: Dict) -> np.ndarray:
    location = np.asarray(label["location"], dtype=float)
    extent = np.asarray(label["extent"], dtype=float) * 2.0
    yaw_rad = np.deg2rad(label["angle"][1])
    return np.array([location[0], location[1], location[2], extent[0], extent[1], extent[2], yaw_rad], dtype=float)


def get_lidar_pose(frame_data: Dict, vehicle_id: int) -> Optional[np.ndarray]:
    vehicle_rec = frame_data.get(vehicle_id)
    if not vehicle_rec:
        return None
    calib = vehicle_rec.get("calib")
    if isinstance(calib, dict) and calib.get("lidar_pose") is not None:
        pose = calib["lidar_pose"]
    else:
        pose = vehicle_rec.get("lidar_pose")
    if pose is None:
        return None
    return np.asarray(pose, dtype=float)


def compute_relative_bboxes(
    scenario_meta: Dict,
    frame_ids: List[int],
    object_id: int,
    ego_vehicle_id: int,
    victim_vehicle_id: int,
) -> Optional[np.ndarray]:
    rel_bboxes = []
    for fid in frame_ids:
        labels = scenario_meta["label"].get(fid, {})
        label = labels.get(object_id)
        if label is None:
            return None
        bbox_map = label_to_bbox(label)
        frame_data = scenario_meta["data"].get(fid, {})
        ego_pose = get_lidar_pose(frame_data, ego_vehicle_id)
        victim_pose = get_lidar_pose(frame_data, victim_vehicle_id)
        if ego_pose is None:
            return None
        if victim_pose is None:
            return None

        # Victim position relative to ego, expressed in ego frame.
        victim_rel = victim_pose[:3] - ego_pose[:3]
        ego_rot = rotation_matrix(*(ego_pose[3:] * np.pi / 180.0))
        victim_rel_sensor = (np.linalg.inv(ego_rot) @ victim_rel.reshape(3, 1)).flatten()[:2]

        bbox_sensor = bbox_map_to_sensor(bbox_map, ego_pose)
        bbox_sensor = rotate_bbox_toward_target(bbox_sensor, victim_rel_sensor)
        rel_bboxes.append(bbox_sensor)
    if len(rel_bboxes) != len(frame_ids):
        return None
    return np.vstack(rel_bboxes)


def generate_entries(cases, dataset_meta: Dict, dataset_split: str):
    entries = []
    fixed_frames = list(range(10))
    for case in cases:
        scenario_id = case.get("scenario_id")
        if scenario_id not in dataset_meta:
            continue
        scenario_meta = dataset_meta[scenario_id]
        frame_ids = case.get("frame_ids")
        vehicle_ids = case.get("vehicle_ids", [])
        victim_vehicle_id = case.get("victim_vehicle_id")
        object_id = case.get("object_id")
        case_id = case.get("case_id")
        pair_id = case.get("pair_id")
        if frame_ids is None or object_id is None or victim_vehicle_id is None:
            continue
        for ego_vehicle_id in vehicle_ids:
            rel_bboxes = compute_relative_bboxes(
                scenario_meta, frame_ids, object_id, ego_vehicle_id, victim_vehicle_id
            )
            if rel_bboxes is None:
                continue
            attack_opts = {
                "case_id": case_id,
                "pair_id": pair_id,
                "frame_ids": fixed_frames,
                "ego_vehicle_id": ego_vehicle_id,
                "victim_vehicle_id": victim_vehicle_id,
                "object_id": object_id,
                "bboxes": rel_bboxes,
            }
            attack_meta = {
                "case_id": case_id,
                "pair_id": pair_id,
                "scenario_id": scenario_id,
                "frame_ids": list(frame_ids),
                "ego_vehicle_id": ego_vehicle_id,
                "victim_vehicle_id": victim_vehicle_id,
                "object_id": object_id,
                "attack_frame_ids": fixed_frames,
                "vehicle_ids": list(vehicle_ids),
                "bboxes": rel_bboxes,
            }
            entries.append({"attack_opts": attack_opts, "attack_meta": attack_meta})
    return entries


def main():
    args = parse_args()
    attacks = load_pickle(args.attacks)
    dataset_meta = load_pickle(args.datadir / f"{args.dataset}.pkl")
    entries = generate_entries(attacks, dataset_meta, args.dataset)
    with args.output.open("wb") as handle:
        pickle.dump(entries, handle)
    print(f"Generated {len(entries)} entries and saved to {args.output}")


if __name__ == "__main__":
    main()
