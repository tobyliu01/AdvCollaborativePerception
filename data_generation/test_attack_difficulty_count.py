import argparse
import pickle
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count cases by difficulty.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test_attacks.pkl"),
        help="Path to the generated cases pkl.",
    )
    return parser.parse_args()


def load_cases(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input)
    counts = Counter(case.get("difficulty") for case in cases)

    print(f"Total cases: {len(cases)}")
    for difficulty in sorted(counts):
        count = counts[difficulty]
        portion = count / len(cases) if cases else 0.0
        print(f"difficulty={difficulty}: {count} ({portion:.2%})")


if __name__ == "__main__":
    main()
