import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate new test_cases.pkl from test_cases_old.pkl and attacks list")
    parser.add_argument("--source", type=Path, default=Path("test_cases_old.pkl"), help="Path to the old test_cases.pkl")
    parser.add_argument("--attacks", type=Path, default=Path("test_attacks_unique.pkl"), help="Path to the deduplicated attacks PKL")
    parser.add_argument("--output", type=Path, default=Path("test_cases.pkl"), help="Where to store the new merged PKL")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_multi_frame(attacks):
    multi_frame = []
    seen = set()
    for case in attacks:
        scenario_id = case.get("scenario_id")
        frame_ids = case.get("frame_ids")
        if scenario_id is None or frame_ids is None:
            continue
        key = (scenario_id, tuple(frame_ids))
        if key in seen:
            continue
        seen.add(key)
        multi_frame.append({"scenario_id": scenario_id, "frame_ids": list(frame_ids)})
    return multi_frame


def main():
    args = parse_args()
    original = load_pickle(args.source)
    attacks = load_pickle(args.attacks)

    single_vehicle = original.get("single_vehicle")
    multi_vehicle = original.get("multi_vehicle")
    scenario = original.get("scenario")
    multi_frame = build_multi_frame(attacks)

    new_payload = {
        "single_vehicle": single_vehicle,
        "multi_vehicle": multi_vehicle,
        "multi_frame": multi_frame,
        "scenario": scenario,
    }

    with args.output.open("wb") as handle:
        pickle.dump(new_payload, handle)
    print(f"Saved {len(multi_frame)} multi-frame cases to {args.output}")


if __name__ == "__main__":
    main()
