import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mvp.data.util import get_point_indices_in_bbox, rotation_matrix


FRAMES_PER_SCENARIO = 60
MIN_POINTS_IN_BBOX = 10
MAX_VICTIM_OBJECT_DISTANCE = 50.0
LOG_PATH = Path(__file__).resolve().parent / "test_attacks_log.txt"
scenarios = ["2021_08_22_21_41_24", "2021_08_23_16_06_26", "2021_08_23_21_07_10", "2021_08_24_07_45_41", "2021_08_23_12_58_19",\
             "2021_08_23_15_19_19", "2021_08_24_20_49_54", "2021_08_21_09_28_12", "2021_08_23_17_22_47", "2021_08_22_09_08_29",\
             "2021_08_22_07_52_02", "2021_08_20_21_10_24", "2021_08_24_20_09_18", "2021_08_23_21_47_19", "2021_08_18_19_48_05"]
starts = [69, 69, 69, 70, 68, 68, 68, 69, 69, 68, 71, 69, 68, 69, 68]

CaseSpec = Tuple[str, int]


def log_debug(message: str) -> None:
    """Append a single debug line to the log file."""
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def format_distance_value(dist: Optional[float]) -> str:
    return "NA" if dist is None else f"{dist:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate randomized attack cases from OPV2V metadata.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="../data/OPV2V",
        help="Dataset root that contains train/validate/test PKL files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        choices=["train", "validate", "test"],
        help="Which dataset split PKL to use.",
    )
    parser.add_argument(
        "--case-length",
        type=int,
        default=10,
        help="Number of frames per case.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=2,
        help="Spacing between consecutive frames (default matches OPV2V).",
    )
    parser.add_argument(
        "--samples-per-case",
        type=int,
        default=6,
        help="How many (victim, object) pairs to draw for every frame chunk.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test_attacks.pkl",
        help="Where to store the generated cases (PKL file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def load_meta(datadir: str, dataset: str) -> Dict[str, Dict]:
    pkl_path = os.path.join(datadir, f"{dataset}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing dataset pkl: {pkl_path}")
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)


def coerce_case_specs(args: argparse.Namespace) -> List[CaseSpec]:
    if len(scenarios) != len(starts):
        raise ValueError("--scenario and --start must contain the same number of items.")
    specs: List[CaseSpec] = []
    for scenario, start in zip(scenarios, starts):
        specs.append((scenario, int(start)))
    if not specs:
        raise ValueError("Provide at least one scenario/start pair.")
    return specs


def build_frame_chunks(
    start_frame: int,
    total_frames: int,
    case_len: int,
    interval: int,
) -> List[List[int]]:
    """Generate sequential frame windows for each case."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if total_frames % 10 != 0:
        raise ValueError("total_frames must be a multiplier of 10")
    frames = [start_frame + interval * i for i in range(total_frames)]
    if len(frames) < case_len:
        raise ValueError(
            f"Not enough frames ({len(frames)}) to form a case of length {case_len}.",
        )
    chunks: List[List[int]] = []
    for idx in range(0, len(frames), case_len):
        chunk = frames[idx : idx + case_len]
        if len(chunk) == case_len:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No complete frame chunks were produced.")
    return chunks


def frame_ids_exist(scenario_id: str, scenario_meta: Dict, frame_ids: Iterable[int]) -> None:
    missing = [fid for fid in frame_ids if fid not in scenario_meta["data"]]
    if missing:
        raise KeyError(f"Scenario {scenario_id} missing frames: {missing[:5]}...")


def select_object_id(
    labels: Dict[int, Dict],
    frame_ids: Sequence[int],
    excluded_ids: Sequence[int],
    rng: random.Random,
) -> int:
    excluded = set(int(v) for v in excluded_ids)
    shuffled_frames = list(frame_ids)
    rng.shuffle(shuffled_frames)
    for fid in shuffled_frames:
        frame_labels = labels.get(fid, {})
        if not frame_labels:
            continue
        candidates = list(frame_labels.keys())
        rng.shuffle(candidates)
        for cid in candidates:
            obj_id = int(cid)
            if obj_id not in excluded:
                return obj_id
    raise RuntimeError("Unable to find an object id outside the cooperative set.")


def label_to_bbox(label: Dict) -> np.ndarray:
    location = np.asarray(label["location"], dtype=float)
    extent = np.asarray(label["extent"], dtype=float) * 2.0
    yaw_rad = np.deg2rad(label["angle"][1])
    return np.array(
        [location[0], location[1], location[2], extent[0], extent[1], extent[2], yaw_rad],
        dtype=float,
    )


def load_vehicle_points_map(
    datadir: str,
    scenario_meta: Dict,
    frame_id: int,
    vehicle_id: int,
    cache: Dict[Tuple[int, int], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    cache_key = (frame_id, vehicle_id)
    if cache_key in cache:
        return cache[cache_key]

    frame_rec = scenario_meta["data"].get(frame_id, {})
    vehicle_rec = frame_rec.get(vehicle_id)
    if not vehicle_rec:
        cache[cache_key] = None
        return None

    pcd_path = vehicle_rec.get("lidar")
    if not pcd_path:
        cache[cache_key] = None
        return None
    if not os.path.isabs(pcd_path):
        pcd_path = os.path.join(datadir, pcd_path)
    if not os.path.exists(pcd_path):
        cache[cache_key] = None
        return None

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        cache[cache_key] = None
        return None

    calib_entry = vehicle_rec.get("calib")
    if isinstance(calib_entry, dict):
        lidar_pose = calib_entry.get("lidar_pose")
    else:
        lidar_pose = None
    if lidar_pose is None:
        lidar_pose = vehicle_rec.get("lidar_pose")
    if lidar_pose is None:
        cache[cache_key] = None
        return None

    pose = np.asarray(lidar_pose, dtype=float)
    R = rotation_matrix(*(np.deg2rad(pose[3:])))
    pts_map = (R @ pts.T).T + pose[:3]
    cache[cache_key] = pts_map
    return pts_map


def points_in_bbox_per_frame(
    datadir: str,
    scenario_meta: Dict,
    labels: Dict[int, Dict],
    frame_ids: Sequence[int],
    victim_vehicle_id: int,
    object_id: int,
    cache: Dict[Tuple[int, int], Optional[np.ndarray]],
) -> List[Tuple[int, int]]:
    """Count LiDAR points inside the object bbox for every frame in the chunk."""
    counts: List[Tuple[int, int]] = []
    for fid in frame_ids:
        frame_labels = labels.get(fid, {})
        label = frame_labels.get(object_id)
        if label is None:
            counts.append((fid, 0))
            continue
        pts_map = load_vehicle_points_map(datadir, scenario_meta, fid, victim_vehicle_id, cache)
        if pts_map is None:
            counts.append((fid, 0))
            continue
        bbox = label_to_bbox(label)
        bbox[3:6] += 0.2
        indices = get_point_indices_in_bbox(bbox, pts_map)
        counts.append((fid, int(len(indices))))
    return counts


def victim_object_distances(
    labels: Dict[int, Dict],
    frame_ids: Sequence[int],
    victim_vehicle_id: int,
    object_id: int,
) -> List[Tuple[int, Optional[float]]]:
    """Measure victim/object XY distance per frame using GT locations."""
    distances: List[Tuple[int, Optional[float]]] = []
    for fid in frame_ids:
        frame_labels = labels.get(fid, {})
        obj_label = frame_labels.get(object_id)
        victim_label = frame_labels.get(victim_vehicle_id)
        if obj_label is None or victim_label is None:
            distances.append((fid, None))
            continue
        obj_loc = np.asarray(obj_label["location"], dtype=float)
        victim_loc = np.asarray(victim_label["location"], dtype=float)
        dist = float(np.linalg.norm(obj_loc[:2] - victim_loc[:2]))
        distances.append((fid, dist))
    return distances


def generate_cases_for_spec(
    meta: Dict[str, Dict],
    spec: CaseSpec,
    case_len: int,
    interval: int,
    samples_per_case: int,
    rng: random.Random,
    start_case_id: int,
    frames_per_scenario: int,
    datadir: str,
) -> Tuple[List[Dict], int]:
    scenario_id, start_frame = spec
    if scenario_id not in meta:
        raise KeyError(f"Scenario {scenario_id} not found in dataset.")
    scenario_meta = meta[scenario_id]

    chunks = build_frame_chunks(start_frame, frames_per_scenario, case_len, interval)
    flat_frame_ids = [fid for chunk in chunks for fid in chunk]
    frame_ids_exist(scenario_id, scenario_meta, flat_frame_ids)

    vehicle_ids = [int(v) for v in scenario_meta.get("vehicle_ids", [])]
    vehicle_ids.sort()
    if not vehicle_ids:
        raise ValueError(f"No cooperative vehicle ids found for scenario {scenario_id}.")

    cases: List[Dict] = []
    case_id = start_case_id
    labels = scenario_meta.get("label", {})
    pcd_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}  # reuse transformed point clouds
    max_attempts = 50
    for chunk in chunks:
        frame_ids = list(chunk)
        sampled_count = 0
        chunk_failed = False
        while sampled_count < samples_per_case:
            for attempt in range(max_attempts):
                victim_vehicle_id = rng.choice(vehicle_ids)
                object_id = select_object_id(labels, frame_ids, vehicle_ids, rng)

                # Filter 1: ensure the object is within victim vehicle's visible range.
                distance_entries = victim_object_distances(
                    labels=labels,
                    frame_ids=frame_ids,
                    victim_vehicle_id=victim_vehicle_id,
                    object_id=object_id,
                )
                if not distance_entries or all(dist is None for _, dist in distance_entries):
                    log_debug(
                        "[DEBUG] scenario=%s frames=%s victim=%d object=%d distance_info=missing attempt=%d"
                        % (scenario_id, frame_ids, victim_vehicle_id, object_id, attempt + 1)
                    )
                    continue
                max_distance = max(dist for _, dist in distance_entries if dist is not None)
                min_distance = min(dist for _, dist in distance_entries if dist is not None)
                if max_distance >= MAX_VICTIM_OBJECT_DISTANCE:
                    distance_str = ",".join(
                        f"{fid}:{format_distance_value(dist)}" for fid, dist in distance_entries
                    )
                    log_debug(
                        "[DEBUG] scenario=%s frames=%s victim=%d object=%d per_frame=%s "
                        "min=%.2f max=%.2f distance_fail attempt=%d"
                        % (
                            scenario_id,
                            frame_ids,
                            victim_vehicle_id,
                            object_id,
                            distance_str,
                            min_distance,
                            max_distance,
                            attempt + 1,
                        )
                    )
                    continue

                # Filter 2: ensure lidar points are on the object from the victim's lidar view.
                frame_counts = points_in_bbox_per_frame(
                    datadir=datadir,
                    scenario_meta=scenario_meta,
                    labels=labels,
                    frame_ids=frame_ids,
                    victim_vehicle_id=victim_vehicle_id,
                    object_id=object_id,
                    cache=pcd_cache,
                )
                min_points = min((cnt for _, cnt in frame_counts), default=0)
                distance_str = ",".join(f"{fid}:{format_distance_value(dist)}" for fid, dist in distance_entries)
                log_debug(
                    "[DEBUG] scenario=%s frames=%s victim=%d object=%d per_frame=%s min_points=%d "
                    "distance_per_frame=%s min_distance=%.2f max_distance=%.2f attempt=%d"
                    % (
                        scenario_id,
                        frame_ids,
                        victim_vehicle_id,
                        object_id,
                        ",".join(f"{fid}:{cnt}" for fid, cnt in frame_counts),
                        min_points,
                        distance_str,
                        min_distance,
                        max_distance,
                        attempt + 1,
                    )
                )
                if not (frame_counts and all(count >= MIN_POINTS_IN_BBOX for _, count in frame_counts)):
                    continue

                # Filter 3: ensure at least one collaborator observes the object.
                collaborator_ok = False
                collab_details: List[str] = []
                for collaborator_id in vehicle_ids:
                    if collaborator_id == victim_vehicle_id:
                        continue
                    collab_counts = points_in_bbox_per_frame(
                        datadir=datadir,
                        scenario_meta=scenario_meta,
                        labels=labels,
                        frame_ids=frame_ids,
                        victim_vehicle_id=collaborator_id,
                        object_id=object_id,
                        cache=pcd_cache,
                    )
                    min_collab_points = min((cnt for _, cnt in collab_counts), default=0)
                    collab_details.append(f"{collaborator_id}:{min_collab_points}")
                    if min_collab_points >= MIN_POINTS_IN_BBOX:
                        collaborator_ok = True
                        break
                if not collaborator_ok:
                    log_debug(
                        "[DEBUG] scenario=%s frames=%s victim=%d object=%d details=%s collaborator_fail attempt=%d"
                        % (
                            scenario_id,
                            frame_ids,
                            victim_vehicle_id,
                            object_id,
                            ",".join(collab_details),
                            attempt + 1,
                        )
                    )
                    continue

                # Append case if all requirements are satisfied.
                cases.append(
                    {
                        "case_id": case_id,
                        "scenario_id": scenario_id,
                        "frame_ids": frame_ids.copy(),
                        "victim_vehicle_id": victim_vehicle_id,
                        "object_id": object_id,
                        "vehicle_ids": vehicle_ids.copy(),
                    },
                )
                sampled_count += 1
                break
            else:  # Failed to find a valid pair after max attempts, skip chunk.
                msg = (
                    f"[WARN] giving up on scenario={scenario_id} frame_chunk={frame_ids} "
                    f"after {max_attempts} attempts."
                )
                print(msg)
                log_debug(msg)
                chunk_failed = True
                break
        if chunk_failed:
            continue
        case_id += 1
    return cases, case_id


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    specs = coerce_case_specs(args)
    meta = load_meta(args.datadir, args.dataset)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")  # fresh log for each run

    all_cases: List[Dict] = []
    case_id = 0
    for spec in specs:
        cases_chunk, case_id = generate_cases_for_spec(
            meta=meta,
            spec=spec,
            case_len=args.case_length,
            interval=args.frame_interval,
            samples_per_case=args.samples_per_case,
            rng=rng,
            start_case_id=case_id,
            frames_per_scenario=FRAMES_PER_SCENARIO,
            datadir=args.datadir,
        )
        all_cases.extend(cases_chunk)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(all_cases, handle)

    print(f"Generated {len(all_cases)} cases across {len(specs)} spec(s).")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
