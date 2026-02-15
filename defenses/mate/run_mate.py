#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from mate.estimator import MATEConfig, MATEEstimator
from mate.io import load_scenarios_from_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MATE trust estimation on OpenCOOD-style data.")
    parser.add_argument("--input-pkl", required=True, help="Path to pickle containing scenario/frame data.")
    parser.add_argument(
        "--output-json",
        default="mate_trust_scores.json",
        help="Output JSON path for final trust scores and histories.",
    )
    parser.add_argument("--assignment-distance", type=float, default=2.0)
    parser.add_argument("--track-neg-bias", type=float, default=1.0)
    parser.add_argument("--agent-neg-bias", type=float, default=6.0)
    parser.add_argument("--prop-omega", type=float, default=0.02)
    parser.add_argument(
        "--disable-unmatched-pred-penalty",
        action="store_true",
        help="Disable agent penalty for local predictions unmatched to GT/AGG tracks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios_from_pickle(args.input_pkl)

    config = MATEConfig(
        assignment_distance_m=args.assignment_distance,
        track_negativity_bias=args.track_neg_bias,
        agent_negativity_bias=args.agent_neg_bias,
        propagation_omega=args.prop_omega,
        penalize_unmatched_predictions=not args.disable_unmatched_pred_penalty,
    )
    estimator = MATEEstimator(config=config)
    results = estimator.run_scenarios(scenarios)

    output: Dict[str, Any] = {}
    for scenario_id, result in results.items():
        output[scenario_id] = {
            "final_agent_trust": {str(k): float(v) for k, v in result.final_agent_trust.items()},
            "final_track_trust": {str(k): float(v) for k, v in result.final_track_trust.items()},
            "agent_trust_history": {
                str(k): [float(x) for x in history]
                for k, history in result.agent_trust_history.items()
            },
            "num_frames": len(next(iter(result.agent_trust_history.values()), [])),
        }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Final per-scenario per-CAV trust:")
    for scenario_id, scenario_data in output.items():
        print(f"- {scenario_id}")
        for cav_id, trust in scenario_data["final_agent_trust"].items():
            print(f"  CAV {cav_id}: {trust:.4f}")
    print(f"\nSaved detailed output to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
