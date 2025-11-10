import argparse
import os
import pickle
import random
from typing import Dict, Iterable, List, Sequence, Tuple


CaseSpec = Tuple[str, int]
FRAMES_PER_SCENARIO = 60
scenarios = ["2021_08_22_21_41_24", "2021_08_23_16_06_26", "2021_08_23_21_07_10", "2021_08_24_07_45_41", "2021_08_23_12_58_19",\
             "2021_08_23_15_19_19", "2021_08_24_20_49_54", "2021_08_21_09_28_12", "2021_08_23_17_22_47", "2021_08_22_09_08_29",\
             "2021_08_22_07_52_02", "2021_08_20_21_10_24", "2021_08_24_20_09_18", "2021_08_23_21_47_19", "2021_08_24_11_37_54",\
             "2021_08_18_19_48_05"]
starts = [69, 69, 69, 70, 68, 68, 68, 69, 69, 68, 71, 69, 68, 69, 149, 68]


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
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
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


def frame_ids_exist(scenario_meta: Dict, frame_ids: Iterable[int]) -> None:
    missing = [fid for fid in frame_ids if fid not in scenario_meta["data"]]
    if missing:
        raise KeyError(f"Scenario missing frames: {missing[:5]}...")


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


def generate_cases_for_spec(
    meta: Dict[str, Dict],
    spec: CaseSpec,
    case_len: int,
    interval: int,
    samples_per_case: int,
    rng: random.Random,
    start_case_id: int,
    frames_per_scenario: int,
) -> Tuple[List[Dict], int]:
    scenario_id, start_frame = spec
    if scenario_id not in meta:
        raise KeyError(f"Scenario {scenario_id} not found in dataset.")
    scenario_meta = meta[scenario_id]

    chunks = build_frame_chunks(start_frame, frames_per_scenario, case_len, interval)
    flat_frame_ids = [fid for chunk in chunks for fid in chunk]
    frame_ids_exist(scenario_meta, flat_frame_ids)

    vehicle_ids = [int(v) for v in scenario_meta.get("vehicle_ids", [])]
    vehicle_ids.sort()
    if not vehicle_ids:
        raise ValueError(f"No cooperative vehicle ids found for scenario {scenario_id}.")

    cases: List[Dict] = []
    case_id = start_case_id
    labels = scenario_meta.get("label", {})
    for chunk in chunks:
        frame_ids = list(chunk)
        for _ in range(samples_per_case):
            victim_vehicle_id = rng.choice(vehicle_ids)
            object_id = select_object_id(labels, frame_ids, vehicle_ids, rng)
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
        case_id += 1
    return cases, case_id


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    specs = coerce_case_specs(args)
    meta = load_meta(args.datadir, args.dataset)

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
        )
        all_cases.extend(cases_chunk)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(all_cases, handle)

    print(f"Generated {len(all_cases)} cases across {len(specs)} spec(s).")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
