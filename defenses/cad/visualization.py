from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .config import DEBUG_MODE


def geometry_to_polygons(geom: Any) -> List[Polygon]:
    if geom is None:
        return []
    try:
        if geom.is_empty:
            return []
    except Exception:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty and poly.area > 0]
    if isinstance(geom, GeometryCollection):
        polygons: List[Polygon] = []
        for sub_geom in geom.geoms:
            polygons.extend(geometry_to_polygons(sub_geom))
        return polygons
    return []


def draw_polygons(
    ax: Any,
    polygons: Iterable[Any],
    color: str,
    alpha: float,
    fill: bool,
    border: bool,
    linewidth: float = 1.0,
) -> None:
    for geom in polygons:
        for polygon in geometry_to_polygons(geom):
            x, y = polygon.exterior.coords.xy
            if fill:
                ax.fill(x, y, color=color, alpha=alpha)
            if border:
                ax.plot(x, y, color=color, alpha=max(alpha, 0.2), linewidth=linewidth)


def collect_bounds(polygons: Iterable[Any]) -> List[float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for geom in polygons:
        for polygon in geometry_to_polygons(geom):
            x0, y0, x1, y1 = polygon.bounds
            min_x = min(min_x, float(x0))
            min_y = min(min_y, float(y0))
            max_x = max(max_x, float(x1))
            max_y = max(max_y, float(y1))
    if min_x == float("inf"):
        return []
    return [min_x, min_y, max_x, max_y]


def save_fused_occupancy_visualization(
    *,
    visualization_root: str,
    case_id: int,
    pair_id: int,
    frame_id: int,
    frame_occupancy: Dict[Any, Dict[str, Any]],
    frame_metric: Dict[str, Any],
    logger: Any,
) -> None:
    save_dir = os.path.join(
        visualization_root,
        "case{:06d}".format(int(case_id)),
        "pair{:02d}".format(int(pair_id)),
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "frame{:02d}.png".format(int(frame_id)))

    fig, ax = plt.subplots(figsize=(14, 14))

    for _, vehicle_data in sorted(frame_occupancy.items(), key=lambda x: str(x[0])):
        draw_polygons(
            ax=ax,
            polygons=vehicle_data.get("free_areas", []),
            color="tab:blue",
            alpha=0.05,
            fill=True,
            border=False,
        )
        draw_polygons(
            ax=ax,
            polygons=vehicle_data.get("occupied_areas", []),
            color="tab:green",
            alpha=0.08,
            fill=True,
            border=False,
        )
        draw_polygons(
            ax=ax,
            polygons=[vehicle_data.get("ego_area")],
            color="black",
            alpha=0.35,
            fill=False,
            border=True,
            linewidth=1.0,
        )

    draw_polygons(
        ax=ax,
        polygons=[frame_metric.get("_fused_free_geom")],
        color="tab:blue",
        alpha=0.15,
        fill=True,
        border=False,
    )
    draw_polygons(
        ax=ax,
        polygons=[frame_metric.get("_fused_occupied_geom")],
        color="tab:green",
        alpha=0.25,
        fill=True,
        border=False,
    )
    draw_polygons(
        ax=ax,
        polygons=frame_metric.get("_conflicted_geoms", []),
        color="tab:red",
        alpha=0.6,
        fill=True,
        border=True,
        linewidth=1.2,
    )

    bounds = collect_bounds(
        list(frame_metric.get("_conflicted_geoms", []))
        + [frame_metric.get("_fused_occupied_geom"), frame_metric.get("_fused_free_geom")]
    )
    if len(bounds) == 4:
        margin = 2.0
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "CAD Fused Occupancy case {:06d} pair {:02d} frame {:02d} | conflicted={} area={:.3f}".format(
            int(case_id),
            int(pair_id),
            int(frame_id),
            bool(frame_metric.get("conflicted", False)),
            float(frame_metric.get("conflicted_area_total", 0.0)),
        )
    )
    fig.savefig(save_file, dpi=150)
    plt.close(fig)

    if DEBUG_MODE:
        logger.info("[CAD_DEBUG] Saved fused occupancy visualization: %s", save_file)
