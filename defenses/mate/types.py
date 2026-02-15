from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional

import numpy as np


AgentId = Hashable
TrackId = int


@dataclass(slots=True)
class PSM:
    """Trust pseudomeasurement as (value, confidence), both in [0, 1]."""

    value: float
    confidence: float
    reason: str = ""


@dataclass(slots=True)
class CAVFramePrediction:
    """
    Predictions for one CAV in one frame.

    Boxes are expected in [x, y, z, l, w, h, yaw] format.
    """

    pred_bboxes: np.ndarray
    pred_scores: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    boxes_in_global: bool = False


@dataclass(slots=True)
class FrameData:
    """One scenario frame with GT and all CAV predictions."""

    frame_id: int
    gt_bboxes: np.ndarray
    cavs: Dict[AgentId, CAVFramePrediction]
    gt_ids: Optional[np.ndarray] = None


@dataclass(slots=True)
class ScenarioData:
    """One scenario containing all frames."""

    scenario_id: str
    frames: List[FrameData]


@dataclass(slots=True)
class ScenarioTrustResult:
    """Outputs of MATE for one scenario."""

    scenario_id: str
    final_agent_trust: Dict[AgentId, float]
    final_track_trust: Dict[TrackId, float]
    agent_trust_history: Dict[AgentId, List[float]]
    track_trust_history: Dict[TrackId, List[float]] = field(default_factory=dict)


BoxTransformFn = Callable[[AgentId, int, np.ndarray], np.ndarray]
