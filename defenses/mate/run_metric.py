from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


TRUST_SCORE_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result/defense/mate"
BASELINE_MESH_MODEL = "real_car"
COMPARISON_MESH_MODEL = "adv_real_car_with_plane_victim_03"
RESULT_LOG_FILE = os.path.join(TRUST_SCORE_ROOT, "result.log")


def _load_trust_records(mesh_model_name: str) -> List[dict]:
    file_path = os.path.join(TRUST_SCORE_ROOT, "{}.pkl".format(mesh_model_name))
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Trust score file not found: {}".format(file_path))
    with open(file_path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise ValueError("Expected list records in {}, got {}".format(file_path, type(records)))
    return records


def _to_record_map(records: List[dict]) -> Dict[Tuple[int, int], dict]:
    record_map: Dict[Tuple[int, int], dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if "case_id" not in record or "pair_id" not in record:
            continue
        key = (int(record["case_id"]), int(record["pair_id"]))
        record_map[key] = record
    return record_map


def _normalize_trust_map(trust_map: Any) -> Dict[str, float]:
    if not isinstance(trust_map, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in trust_map.items():
        out[str(key)] = float(value)
    return out


def _mean_or_nan(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def main() -> None:
    baseline_records = _load_trust_records(BASELINE_MESH_MODEL)
    comparison_records = _load_trust_records(COMPARISON_MESH_MODEL)

    baseline_map = _to_record_map(baseline_records)
    comparison_map = _to_record_map(comparison_records)

    common_keys = sorted(set(baseline_map.keys()).intersection(set(comparison_map.keys())))

    evaluated_cases: List[Tuple[int, int]] = []
    success_cases: List[Tuple[int, int]] = []
    victim_drops: List[float] = []
    non_victim_drops: List[float] = []

    for case_id, pair_id in common_keys:
        baseline_record = baseline_map[(case_id, pair_id)]
        comparison_record = comparison_map[(case_id, pair_id)]

        victim_vehicle_id = baseline_record.get("victim_vehicle_id", comparison_record.get("victim_vehicle_id"))
        if victim_vehicle_id is None:
            continue
        victim_key = str(victim_vehicle_id)

        baseline_trust = _normalize_trust_map(baseline_record.get("final_agent_trust", {}))
        comparison_trust = _normalize_trust_map(comparison_record.get("final_agent_trust", {}))
        if victim_key not in baseline_trust or victim_key not in comparison_trust:
            continue

        baseline_victim_score = baseline_trust[victim_key]
        comparison_victim_score = comparison_trust[victim_key]
        evaluated_cases.append((case_id, pair_id))
        victim_drop = baseline_victim_score - comparison_victim_score
        victim_drops.append(victim_drop)
        if baseline_victim_score > comparison_victim_score:
            success_cases.append((case_id, pair_id))

        common_agents = sorted(set(baseline_trust.keys()).intersection(set(comparison_trust.keys())))
        for agent_key in common_agents:
            if agent_key == victim_key:
                continue
            non_victim_drops.append(baseline_trust[agent_key] - comparison_trust[agent_key])

    total_cases = len(victim_drops)
    success_count = len(success_cases)
    success_rate = float(success_count) / float(total_cases) if total_cases > 0 else 0.0
    failed_cases = [key for key in evaluated_cases if key not in set(success_cases)]

    os.makedirs(TRUST_SCORE_ROOT, exist_ok=True)
    output_lines = [
        "============================================================================",
        "Baseline mesh model: {}".format(BASELINE_MESH_MODEL),
        "Comparison mesh model: {}".format(COMPARISON_MESH_MODEL),
        "Failed cases: {}".format(failed_cases),
        "Total cases: {}".format(total_cases),
        "Success rate: {:.4f}".format(success_rate),
        "Average victim drop: {:.4f}".format(_mean_or_nan(victim_drops)),
        "Average non-victim drop: {:.4f}".format(_mean_or_nan(non_victim_drops)),
        "============================================================================",
        "",
    ]
    with open(RESULT_LOG_FILE, "a") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
