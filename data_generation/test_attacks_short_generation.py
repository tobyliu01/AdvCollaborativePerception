"""Generate a shortened attack list that only keeps case_id 0 or 1."""

import os
import pickle


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "test_attacks_unique.pkl")
    output_path = os.path.join(base_dir, "test_attacks_short.pkl")

    with open(input_path, "rb") as f:
        attacks = pickle.load(f)

    if not isinstance(attacks, list):
        raise ValueError("Expected a list in test_attacks_unique.pkl")

    filtered_attacks = [a for a in attacks if a.get("case_id") in (0, 1)]

    with open(output_path, "wb") as f:
        pickle.dump(filtered_attacks, f)

    print(
        "Saved {} attacks with case_id in {{0,1}} to {}".format(
            len(filtered_attacks), output_path
        )
    )


if __name__ == "__main__":
    main()
