from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
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
