import os
import pickle
import numpy as np

# =========================
# PARAMETERS (edit these)
# =========================
NEW_POSITIONS_PKL = "positions_output.pkl"   # from previous step (list of dicts)
ATTACK_INPUT_PKL  = "../data/OPV2V/attack/lidar_spoof_new.pkl"   # list of dicts (attack_opts, attack_meta)
OUTPUT_PKL        = "../data/OPV2V/attack/lidar_spoof_new1.pkl"  # destination file
# =========================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {path}")

def main():
    # --- Load new positions ---
    new_positions_list = load_pkl(NEW_POSITIONS_PKL)
    if not isinstance(new_positions_list, list):
        raise TypeError("Expected NEW_POSITIONS_PKL to contain a list of dicts.")
    print(f"Loaded {len(new_positions_list)} position blocks from {NEW_POSITIONS_PKL}")

    # --- Load the attack configuration list ---
    attack_list = load_pkl(ATTACK_INPUT_PKL)
    if not isinstance(attack_list, list):
        raise TypeError("Expected the attack file to be a list of dicts.")

    modified_count = 0

    # --- Iterate over all new position blocks ---
    for new_block in new_positions_list:
        scenario_id_new = new_block.get("scenario_id")
        start_frame_id  = new_block.get("start_frame_id")
        positions       = new_block.get("positions", None)

        if scenario_id_new is None or start_frame_id is None or positions is None:
            print("Skipping invalid block (missing keys)")
            continue

        # --- Find matching attack entries ---
        for entry in attack_list:
            if not isinstance(entry, dict):
                continue

            attack_meta = entry.get("attack_meta", {})
            attack_opts = entry.get("attack_opts", {})

            scenario = attack_meta.get("scenario_id", None)
            frame_ids = attack_meta.get("frame_ids", [])
            first_frame_id = frame_ids[0] if isinstance(frame_ids, (list, tuple)) and len(frame_ids) > 0 else None

            # Match by both scenario ID and first frame ID
            if scenario == scenario_id_new and first_frame_id == start_frame_id:
                attack_opts["positions"] = positions
                attack_meta["bboxes"] = positions
                entry["attack_opts"] = attack_opts
                entry["attack_meta"] = attack_meta
                modified_count += 1
                print(f"Updated scenario {scenario}, start_frame {start_frame_id}")

    print(f"Total modified entries: {modified_count}")

    # --- Save updated attack file ---
    save_pkl(attack_list, OUTPUT_PKL)

if __name__ == "__main__":
    main()
