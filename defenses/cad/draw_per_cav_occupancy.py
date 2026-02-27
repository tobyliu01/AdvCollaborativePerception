from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running this file directly (python draw_per_cav_occupancy.py).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from defenses.cad.visualization import collect_bounds, draw_polygons

# Configure constants here.
OCCUPANCY_MAP_FILE = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result/defense/cad/occupancy_map/car_side_8/0/0/frame2.pkl"
OUTPUT_FILE_PREFIX = None  # e.g. "/tmp/frame2_per_cav"
OUTPUT_DPI = 300


def _load_payload(occupancy_map_file: str) -> Dict[str, Any]:
    with open(occupancy_map_file, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid occupancy payload: expected dict, got {}".format(type(payload)))
    return payload


def _default_output_prefix(occupancy_map_file: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(occupancy_map_file))
    return "{}_per_cav".format(base)


def draw_per_cav_occupancy_map(
    *,
    occupancy_map_file: str,
    output_file_prefix: str,
    dpi: int = 150,
) -> list[str]:
    payload = _load_payload(occupancy_map_file)
    regions = payload.get("regions", {})
    per_cav = regions.get("per_cav", {})
    if not isinstance(per_cav, dict) or len(per_cav) == 0:
        raise ValueError("No per_cav occupancy regions found in {}".format(occupancy_map_file))

    sorted_cav_items = sorted(per_cav.items(), key=lambda x: str(x[0]))

    all_geoms = []
    for _, cav_regions in sorted_cav_items:
        all_geoms.extend(cav_regions.get("free_areas", []))
        all_geoms.extend(cav_regions.get("occupied_areas", []))
        all_geoms.append(cav_regions.get("ego_area"))
    bounds = collect_bounds(all_geoms)

    case_id = payload.get("case_id", -1)
    pair_id = payload.get("pair_id", -1)
    frame_id = payload.get("frame_id", -1)

    output_files: list[str] = []
    for cav_id, cav_regions in sorted_cav_items:
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_polygons(
            ax=ax,
            polygons=cav_regions.get("free_areas", []),
            color="tab:blue",
            alpha=0.18,
            fill=True,
            border=False,
        )
        draw_polygons(
            ax=ax,
            polygons=cav_regions.get("occupied_areas", []),
            color="tab:green",
            alpha=0.28,
            fill=True,
            border=False,
        )
        draw_polygons(
            ax=ax,
            polygons=[cav_regions.get("ego_area")],
            color="black",
            alpha=0.8,
            fill=False,
            border=True,
            linewidth=1.2,
        )

        ax.set_title(
            "CAV {} | case {:06d} pair {:02d} frame {:02d}".format(
                cav_id,
                int(case_id),
                int(pair_id),
                int(frame_id),
            )
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if len(bounds) == 4:
            margin = 2.0
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

        out_file = os.path.abspath("{}_cav{}.png".format(output_file_prefix, cav_id))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_file, dpi=int(dpi))
        plt.close(fig)
        output_files.append(out_file)

    return output_files


def main() -> None:
    occupancy_map_file = os.path.abspath(OCCUPANCY_MAP_FILE)
    if not os.path.isfile(occupancy_map_file):
        raise FileNotFoundError("occupancy map file not found: {}".format(occupancy_map_file))

    output_file_prefix = OUTPUT_FILE_PREFIX
    if output_file_prefix is None:
        output_file_prefix = _default_output_prefix(occupancy_map_file)

    saved_files = draw_per_cav_occupancy_map(
        occupancy_map_file=occupancy_map_file,
        output_file_prefix=output_file_prefix,
        dpi=int(OUTPUT_DPI),
    )
    for file_path in saved_files:
        print("Saved per-CAV occupancy visualization: {}".format(file_path))


if __name__ == "__main__":
    main()
