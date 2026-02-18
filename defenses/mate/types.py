from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional

import numpy as np


AgentId = Hashable
TrackId = int


@dataclass
class PSM:
    """Trust pseudomeasurement: (value, confidence), both in [0, 1]."""

    value: float
    confidence: float


@dataclass
class CAVFramePrediction:
    """
    Predictions for one CAV in one frame.

    Boxes: [x, y, z, l, w, h, yaw]
    """

    pred_bboxes: np.ndarray
    pred_scores: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    visible_gt_ids: Optional[np.ndarray] = None  # Ground truth track IDs of this CAV in the current frame.
    bboxes_in_global: bool = False


@dataclass
class FrameData:
    """One scenario frame with ground truth and all CAV predictions."""

    frame_id: int
    gt_bboxes: np.ndarray
    cavs: Dict[AgentId, CAVFramePrediction]
    gt_ids: Optional[np.ndarray] = None


@dataclass
class ScenarioData:
    """One scenario containing all frames."""

    scenario_id: str
    frames: List[FrameData]


@dataclass
class ScenarioTrustResult:
    """Outputs of MATE for one scenario."""

    scenario_id: str
    final_agent_trust: Dict[AgentId, float]
    final_track_trust: Dict[TrackId, float]
    agent_trust_history: Dict[AgentId, List[float]]
    track_trust_history: Dict[TrackId, List[float]] = field(default_factory=dict)
