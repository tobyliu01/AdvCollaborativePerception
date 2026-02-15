from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_centers_xy(bboxes: np.ndarray) -> np.ndarray:
    if bboxes.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return bboxes[:, :2].astype(np.float32, copy=False)


def pairwise_l2_cost(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.size == 0 or rhs.size == 0:
        return np.empty((lhs.shape[0], rhs.shape[0]), dtype=np.float32)
    d = lhs[:, None, :] - rhs[None, :, :]
    return np.linalg.norm(d, axis=-1)


@dataclass(slots=True)
class AssignmentResult:
    matched_pairs: List[Tuple[int, int]]
    unmatched_left: List[int]
    unmatched_right: List[int]


def jvc_distance_assignment(
    left_boxes: np.ndarray,
    right_boxes: np.ndarray,
    max_distance_m: float,
) -> AssignmentResult:
    """
    JVC-style linear assignment with distance cost.
    """
    n_left = int(left_boxes.shape[0])
    n_right = int(right_boxes.shape[0])

    if n_left == 0 and n_right == 0:
        return AssignmentResult([], [], [])
    if n_left == 0:
        return AssignmentResult([], [], list(range(n_right)))
    if n_right == 0:
        return AssignmentResult([], list(range(n_left)), [])

    left_centers = box_centers_xy(left_boxes)
    right_centers = box_centers_xy(right_boxes)
    cost = pairwise_l2_cost(left_centers, right_centers)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs: List[Tuple[int, int]] = []
    matched_left = set()
    matched_right = set()
    for li, ri in zip(row_ind.tolist(), col_ind.tolist()):
        if float(cost[li, ri]) <= max_distance_m:
            matched_pairs.append((li, ri))
            matched_left.add(li)
            matched_right.add(ri)

    unmatched_left = [i for i in range(n_left) if i not in matched_left]
    unmatched_right = [i for i in range(n_right) if i not in matched_right]
    return AssignmentResult(matched_pairs, unmatched_left, unmatched_right)


# Backward-compatible alias.
# hungarian_distance_assignment = jvc_distance_assignment


class TrackIdManager:
    """
    Assign stable integer IDs to per-frame GT boxes when dataset GT IDs are absent.
    """

    def __init__(self, match_distance_m: float = 2.0, max_missing_frames: int = 2):
        self.match_distance_m = match_distance_m
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._active: Dict[int, np.ndarray] = {}
        self._last_seen_frame: Dict[int, int] = {}

    def assign(
        self,
        frame_idx: int,
        gt_boxes: np.ndarray,
        gt_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if gt_ids is not None:
            ids = gt_ids.astype(np.int64, copy=False)
            for tid, box in zip(ids.tolist(), gt_boxes):
                self._active[int(tid)] = box
                self._last_seen_frame[int(tid)] = frame_idx
                self._next_track_id = max(self._next_track_id, int(tid) + 1)
            self._prune(frame_idx)
            return ids

        if gt_boxes.size == 0:
            self._prune(frame_idx)
            return np.empty((0,), dtype=np.int64)

        active_ids = list(self._active.keys())
        if not active_ids:
            ids = self._new_ids(gt_boxes.shape[0])
            self._update_active(ids, gt_boxes, frame_idx)
            self._prune(frame_idx)
            return np.array(ids, dtype=np.int64)

        active_boxes = np.stack([self._active[tid] for tid in active_ids], axis=0)
        result = jvc_distance_assignment(
            left_boxes=gt_boxes,
            right_boxes=active_boxes,
            max_distance_m=self.match_distance_m,
        )

        ids = np.full((gt_boxes.shape[0],), fill_value=-1, dtype=np.int64)
        for cur_idx, active_idx in result.matched_pairs:
            ids[cur_idx] = active_ids[active_idx]

        if result.unmatched_left:
            new_ids = self._new_ids(len(result.unmatched_left))
            for cur_idx, tid in zip(result.unmatched_left, new_ids):
                ids[cur_idx] = tid

        self._update_active(ids.tolist(), gt_boxes, frame_idx)
        self._prune(frame_idx)
        return ids

    def _new_ids(self, n: int) -> List[int]:
        ids = list(range(self._next_track_id, self._next_track_id + n))
        self._next_track_id += n
        return ids

    def _update_active(self, ids: List[int], boxes: np.ndarray, frame_idx: int) -> None:
        for tid, box in zip(ids, boxes):
            itid = int(tid)
            self._active[itid] = box
            self._last_seen_frame[itid] = frame_idx

    def _prune(self, frame_idx: int) -> None:
        stale = [
            tid
            for tid, seen in self._last_seen_frame.items()
            if frame_idx - seen > self.max_missing_frames
        ]
        for tid in stale:
            self._last_seen_frame.pop(tid, None)
            self._active.pop(tid, None)
