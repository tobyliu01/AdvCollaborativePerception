import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate generated attack cases")
    parser.add_argument("--input", type=Path, default=Path("test_attacks.pkl"), help="Path to the input cases pkl")
    parser.add_argument("--output", type=Path, default=Path("test_attacks_unique.pkl"), help="Where to save the deduplicated pkl")
    return parser.parse_args()


def load_cases(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def deduplicate(cases):
    seen = set()
    unique = []
    for case in cases:
        key = (case.get("case_id"), case.get("victim_vehicle_id"), case.get("object_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    pair_counts = {}
    for case in unique:
        case_id = case.get("case_id")
        pair_id = pair_counts.get(case_id, 0)
        case["pair_id"] = pair_id
        pair_counts[case_id] = pair_id + 1
    return unique


def save_cases(path: Path, cases):
    with path.open("wb") as handle:
        pickle.dump(cases, handle)


def main():
    args = parse_args()
    cases = load_cases(args.input)
    unique_cases = deduplicate(cases)
    save_cases(args.output, unique_cases)
    print(f"Total cases read: {len(cases)}; after deduplication: {len(unique_cases)}; saved to {args.output}")


if __name__ == "__main__":
    main()
