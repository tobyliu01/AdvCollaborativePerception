from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


def _safe_geom(geom: Any):
    if geom is None:
        return GeometryCollection()
    try:
        if geom.is_empty:
            return GeometryCollection()
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        return GeometryCollection()
    return geom


def _iter_polygons(geom: Any) -> List[Polygon]:
    geom = _safe_geom(geom)
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if not poly.is_empty and poly.area > 0]
    if isinstance(geom, GeometryCollection):
        polygons: List[Polygon] = []
        for sub_geom in geom.geoms:
            polygons.extend(_iter_polygons(sub_geom))
        return polygons
    return []


def _union_polygons(polygons: Iterable[Any]):
    polygon_list: List[Polygon] = []
    for polygon in polygons:
        polygon_list.extend(_iter_polygons(polygon))
    if len(polygon_list) == 0:
        return GeometryCollection()
    return _safe_geom(unary_union(polygon_list))


class CADPerceptionDefender(object):
    def __init__(self, conflict_area_threshold: float = 0.6):
        self.name = "cad"
        self.conflict_area_threshold = float(conflict_area_threshold)

    @staticmethod
    def _empty_frame_metric(frame_id: int) -> Dict[str, Any]:
        return {
            "frame_id": int(frame_id),
            "vehicle_ids": [],
            "conflicted": False,
            "conflicted_count": 0,
            "conflicted_area_total": 0.0,
            "fused_occupied_area": 0.0,
            "fused_free_area": 0.0,
            "conflicted_regions": [],
            "conflict_pair_details": [],
            "_fused_occupied_geom": GeometryCollection(),
            "_fused_free_geom": GeometryCollection(),
            "_conflicted_geoms": [],
        }

    def _run_frame(self, frame_id: int, frame_data: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        metric = self._empty_frame_metric(frame_id)
        if not isinstance(frame_data, dict) or len(frame_data) == 0:
            return metric

        occupied_by_cav: "OrderedDict[Any, Any]" = OrderedDict()
        free_by_cav: "OrderedDict[Any, Any]" = OrderedDict()

        sorted_vehicle_ids = sorted(frame_data.keys(), key=lambda x: str(x))
        for vehicle_id in sorted_vehicle_ids:
            vehicle_data = frame_data[vehicle_id]
            occupied_polygons: List[Polygon] = []
            for occupied_area in vehicle_data.get("occupied_areas", []):
                occupied_polygons.extend(_iter_polygons(occupied_area))

            ego_area = _safe_geom(vehicle_data.get("ego_area"))
            if not ego_area.is_empty:
                occupied_polygons.extend(_iter_polygons(ego_area))

            free_polygons: List[Polygon] = []
            for free_area in vehicle_data.get("free_areas", []):
                free_polygons.extend(_iter_polygons(free_area))

            occupied_union = _union_polygons(occupied_polygons)
            free_union = _union_polygons(free_polygons)
            if not ego_area.is_empty:
                free_union = _safe_geom(free_union.difference(ego_area))

            occupied_by_cav[vehicle_id] = occupied_union
            free_by_cav[vehicle_id] = free_union

        fused_occupied = _union_polygons(occupied_by_cav.values())
        fused_free = _union_polygons(free_by_cav.values())

        conflict_pair_details: List[Dict[str, Any]] = []
        conflict_polygons: List[Polygon] = []
        vehicle_ids = list(occupied_by_cav.keys())
        for occupied_vehicle_id in vehicle_ids:
            occupied_geom = occupied_by_cav[occupied_vehicle_id]
            if occupied_geom.is_empty:
                continue
            for free_vehicle_id in vehicle_ids:
                if free_vehicle_id == occupied_vehicle_id:
                    continue
                free_geom = free_by_cav[free_vehicle_id]
                if free_geom.is_empty:
                    continue

                overlap = _safe_geom(occupied_geom.intersection(free_geom))
                overlap_polygons = _iter_polygons(overlap)
                if len(overlap_polygons) == 0:
                    continue

                overlap_area = float(sum(float(polygon.area) for polygon in overlap_polygons))
                conflict_polygons.extend(overlap_polygons)
                conflict_pair_details.append(
                    {
                        "occupied_vehicle_id": occupied_vehicle_id,
                        "free_vehicle_id": free_vehicle_id,
                        "area": float(overlap_area),
                    }
                )

        merged_conflict = _union_polygons(conflict_polygons)
        merged_conflict_polygons = _iter_polygons(merged_conflict)
        conflicted_regions: List[Dict[str, Any]] = []
        region_id = 0
        for polygon in merged_conflict_polygons:
            if float(polygon.area) < self.conflict_area_threshold:
                continue
            centroid = polygon.centroid
            conflicted_regions.append(
                {
                    "region_id": int(region_id),
                    "label": "conflicted",
                    "area": float(polygon.area),
                    "centroid": [float(centroid.x), float(centroid.y)],
                    "bounds": [float(v) for v in polygon.bounds],
                }
            )
            region_id += 1

        metric["vehicle_ids"] = vehicle_ids
        metric["conflicted"] = len(conflicted_regions) > 0
        metric["conflicted_count"] = len(conflicted_regions)
        metric["conflicted_area_total"] = float(
            sum(region["area"] for region in conflicted_regions)
        )
        metric["fused_occupied_area"] = float(fused_occupied.area)
        metric["fused_free_area"] = float(fused_free.area)
        metric["conflicted_regions"] = conflicted_regions
        metric["conflict_pair_details"] = conflict_pair_details
        metric["_fused_occupied_geom"] = fused_occupied
        metric["_fused_free_geom"] = fused_free
        metric["_conflicted_geoms"] = [
            polygon for polygon in merged_conflict_polygons
            if float(polygon.area) >= self.conflict_area_threshold
        ]
        return metric

    def run(self, occupancy_feature: List[Dict[Any, Dict[str, Any]]], frame_ids: List[int]):
        metrics = [self._empty_frame_metric(frame_id) for frame_id in range(len(occupancy_feature))]
        for frame_id in frame_ids:
            if frame_id < 0 or frame_id >= len(occupancy_feature):
                continue
            metrics[frame_id] = self._run_frame(frame_id, occupancy_feature[frame_id])
        return metrics
