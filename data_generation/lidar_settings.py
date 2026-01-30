import argparse
import os
import sys
import os
import pickle
from pathlib import Path

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root)
from mvp.data.util import read_pcd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute lidar height/azimuth stats from lidar_shift.pkl")
    parser.add_argument("--attacks", type=Path, default=Path("lidar_shift.pkl"), help="Source lidar_shift PKL")
    parser.add_argument(
        "--datadir",
        type=Path,
        default=Path("/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/data/OPV2V"),
        help="Dataset root containing split PKL files",
    )
    parser.add_argument("--dataset", choices=["train", "validate", "test"], default="test", help="Dataset split to load")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
    args = parse_args()
    attacks = load_pickle(args.attacks)
    dataset_meta = load_pickle(args.datadir / f"{args.dataset}.pkl")

    height_min = np.inf
    height_max = -np.inf
    height_sum = 0.0
    height_count = 0

    az_gap_min = np.inf
    az_gap_max = -np.inf
    az_gap_sum = 0.0
    az_gap_count = 0

    for entry in attacks:
        meta = entry.get("attack_meta", {})
        print(f"Handling case {meta['case_id']} pair {meta['pair_id']} vehicle {meta['ego_vehicle_id']}")
        scenario_id = meta.get("scenario_id")
        frame_ids = meta.get("frame_ids", [])
        ego_vehicle_id = meta.get("ego_vehicle_id")
        if scenario_id not in dataset_meta or ego_vehicle_id is None:
            continue

        scenario_meta = dataset_meta[scenario_id]
        for fid in frame_ids:
            frame_data = scenario_meta.get("data", {}).get(fid, {})
            vehicle_data = frame_data.get(ego_vehicle_id)
            if not vehicle_data:
                continue

            pose = vehicle_data.get("lidar_pose")
            if pose is None:
                calib = vehicle_data.get("calib", {})
                if isinstance(calib, dict):
                    pose = calib.get("lidar_pose")
            if pose is None:
                print("Missing lidar_pose for:", scenario_id, fid, ego_vehicle_id)
                print(vehicle_data)
                continue
            height = float(pose[2])
            height_min = min(height_min, height)
            height_max = max(height_max, height)
            height_sum += height
            height_count += 1

            pcd = vehicle_data.get("lidar")
            if isinstance(pcd, str):
                pcd = read_pcd(os.path.join(args.datadir, pcd))
            if pcd is None:
                pcd = vehicle_data.get("lidar_np")
            if pcd is None or len(pcd) == 0:
                continue

            # Direction unit vectors
            xyz = pcd[:, :3]
            norm = np.linalg.norm(xyz, axis=1)
            valid = norm > 0
            if not np.any(valid):
                continue
            xyz = xyz[valid]
            norm = norm[valid]
            directions = xyz / norm[:, None]

            # Azimuth and elevation angles
            az = np.arctan2(directions[:, 1], directions[:, 0])
            el = np.arctan2(directions[:, 2], np.sqrt(directions[:, 0] ** 2 + directions[:, 1] ** 2))

            # Bin by elevation into 64 channels
            el_min = float(el.min())
            el_max = float(el.max())
            if el_max == el_min:
                continue
            bin_edges = np.linspace(el_min, el_max, 65, endpoint=True)
            channel_ids = np.clip(np.digitize(el, bin_edges) - 1, 0, 63)

            # Azimuth gaps within each channel
            for ch in range(64):
                ch_mask = channel_ids == ch
                if np.sum(ch_mask) < 2:
                    continue
                ch_az = np.sort(az[ch_mask])
                gaps = np.diff(ch_az)
                wrap_gap = (ch_az[0] + 2 * np.pi) - ch_az[-1]
                gaps = np.hstack([gaps, wrap_gap])
                az_gap_min = min(az_gap_min, float(gaps.min()))
                az_gap_max = max(az_gap_max, float(gaps.max()))
                az_gap_sum += float(gaps.sum())
                az_gap_count += gaps.shape[0]

    height_avg = height_sum / height_count if height_count > 0 else float("nan")
    az_gap_avg = az_gap_sum / az_gap_count if az_gap_count > 0 else float("nan")

    print("Lidar height (z) stats:")
    print(f"  min: {height_min:.6f}, max: {height_max:.6f}, avg: {height_avg:.6f} (count={height_count})")
    print("Azimuth gap stats within channels (radians):")
    print(f"  min: {az_gap_min:.6f}, max: {az_gap_max:.6f}, avg: {az_gap_avg:.6f} (count={az_gap_count})")
    print("Azimuth gap stats within channels (degrees):")
    print(f"  min: {np.degrees(az_gap_min):.3f}, max: {np.degrees(az_gap_max):.3f}, avg: {np.degrees(az_gap_avg):.3f}")


if __name__ == "__main__":
    main()
