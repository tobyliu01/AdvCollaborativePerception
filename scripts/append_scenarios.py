# import os
# import pickle
# from typing import Any, Dict, List

# # ============= PARAMETERS =============
# INPUT_PKL = "../data/OPV2V/attack/lidar_spoof_new.pkl"   # original attack list
# OUTPUT_PKL = "../data/OPV2V/attack/lidar_spoof_new_dup.pkl"
# TARGET_SCENARIO_ID = "2021_08_22_21_41_24"
# COPY_TIMES = 2
# # ======================================

# def load_pkl(path: str) -> Any:
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def save_pkl(obj: Any, path: str) -> None:
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#     with open(path, "wb") as f:
#         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"Saved: {path}")

# def replicate_by_scenario(attack_list: List[Dict], scenario_id: str, copies: int) -> List[Dict]:
#     """
#     For each element in attack_list:
#       - if element is a dict and element['attack_meta']['scenario_id'] == scenario_id,
#         append that element (1 + copies) times to the output list.
#       - otherwise append it once.
#     """
#     out = []
#     matched = 0
#     for idx, entry in enumerate(attack_list):
#         if not isinstance(entry, dict):
#             # If entry is not a dict, just preserve it
#             out.append(entry)
#             continue
#         attack_meta = entry.get("attack_meta")
#         if isinstance(attack_meta, dict) and attack_meta.get("scenario_id") == scenario_id:
#             # replicate (original + copies)
#             for _ in range(1 + copies):
#                 out.append(entry.copy() if isinstance(entry, dict) else entry)
#             matched += 1
#         else:
#             out.append(entry)
#     print(f"Total entries in input: {len(attack_list)}. Matched scenario_id '{scenario_id}': {matched}.")
#     print(f"Total entries in output: {len(out)}.")
#     return out

# def main():
#     attack_list = load_pkl(INPUT_PKL)
#     if not isinstance(attack_list, list):
#         raise TypeError(f"Expected a list at top-level in {INPUT_PKL}, got {type(attack_list)}")

#     new_list = replicate_by_scenario(attack_list, TARGET_SCENARIO_ID, COPY_TIMES)
#     save_pkl(new_list, OUTPUT_PKL)

# if __name__ == "__main__":
#     main()




# =========================================================================================================================================================================================


import pickle
import numpy as np
import copy  # <-- important

# ========= PARAMETERS =========
POSITIONS_PKL     = "positions_output.pkl"
ATTACK_PKL_IN     = "../data/OPV2V/attack/lidar_spoof_new_dup.pkl"
ATTACK_PKL_OUT    = "../data/OPV2V/attack/lidar_spoof_new_dup1.pkl"
SCENARIO_ID       = "2021_08_22_21_41_24"
# ==============================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {path}")

def ensure_10x7(positions):
    arr = np.asarray(positions, dtype=float)
    if arr.shape != (10, 7):
        raise ValueError(f"'positions' must be shape (10,7), got {arr.shape}")
    return arr

def main():
    pos_list = load_pkl(POSITIONS_PKL)
    atk_list = load_pkl(ATTACK_PKL_IN)

    if not isinstance(pos_list, list):
        raise TypeError("positions_output.pkl must be a list of dicts.")
    if not isinstance(atk_list, list):
        raise TypeError("attack pkl must be a list of dicts.")

    # Filter positions by scenario, preserve order
    pos_filtered = [d for d in pos_list
                    if isinstance(d, dict) and d.get("scenario_id") == SCENARIO_ID]

    # Collect indices in attack list with matching scenario, preserve order
    atk_indices = []
    for i, e in enumerate(atk_list):
        if not isinstance(e, dict):
            continue
        am = e.get("attack_meta", {})
        if isinstance(am, dict) and am.get("scenario_id") == SCENARIO_ID:
            atk_indices.append(i)

    if len(pos_filtered) != len(atk_indices):
        raise ValueError(
            f"Count mismatch for scenario '{SCENARIO_ID}': "
            f"{len(pos_filtered)} in positions_output.pkl vs {len(atk_indices)} in attack file."
        )

    modified = 0
    for k, idx in enumerate(atk_indices):
        src = pos_filtered[k]

        # deep copy the target entry
        dst = copy.deepcopy(atk_list[idx])

        positions = ensure_10x7(src.get("positions")).copy()  # copy NumPy array too
        ref_vid   = src.get("reference_vehicle_id")
        if ref_vid is None:
            raise KeyError("positions entry missing 'reference_vehicle_id'.")

        attack_opts = dict(dst.get("attack_opts") or {})
        attack_meta = dict(dst.get("attack_meta") or {})

        # Replace arrays
        attack_opts["positions"] = positions
        attack_meta["bboxes"]    = positions

        # Replace attacker/victim ids
        attack_opts["attacker_vehicle_id"] = ref_vid
        attack_meta["attacker_vehicle_id"] = ref_vid
        attack_meta["victim_vehicle_id"]   = ref_vid

        dst["attack_opts"] = attack_opts
        dst["attack_meta"] = attack_meta

        # Write back the de-aliased, modified entry
        atk_list[idx] = dst
        modified += 1

    print(f"Modified entries (scenario={SCENARIO_ID}): {modified}")
    save_pkl(atk_list, ATTACK_PKL_OUT)

if __name__ == "__main__":
    main()

