from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
from .trust import BetaTrustState
from .types import (
    AgentId,
    CAVFramePrediction,
    PSM,
    ScenarioData,
    ScenarioTrustResult,
    TrackId,
)
from .visibility import RangeVisibilityModel


@dataclass
class MATEConfig:
    # Step 0: priors
    # prior_agent_alpha/beta: initial trust prior for each agent, Beta(alpha, beta)
    # prior_track_alpha/beta: initial trust prior for each track, Beta(alpha, beta)
    prior_agent_alpha: float = 1.0
    prior_agent_beta: float = 1.0
    prior_track_alpha: float = 2.0
    prior_track_beta: float = 2.0

    # Step 1: trust propagation
    # propagation_omega: PropP interpolation weight towards initial prior each frame.
    # Small value -> slow drift back to prior when no strong evidence exists.
    propagation_omega: float = 0.01
    # propagation_delta_mu / target_mu: PropE for expectation drift.
    # Larger delta_mu -> slower drift of trust mean toward target_mu.
    propagation_delta_mu: Optional[float] = None
    propagation_target_mu: float = 0.5
    # propagation_delta_nu / target_nu: PropV for precision drift.
    # Larger delta_nu -> slower drift of precision toward target_nu.
    propagation_delta_nu: Optional[float] = None
    propagation_target_nu: float = 4.0

    # Step 2: assignment / visibility
    # Max center-distance to accept a pred<->GT/AGG match.
    assignment_distance_m: float = 2.0
    # Maximum lidar range
    lidar_visibility_range_m: float = 120.0

    # Step 3: weighted Beta update
    # track_negativity_bias / threshold: extra penalty weight for negative track-side PSMs.
    # agent_negativity_bias / threshold: extra penalty weight for negative agent-side PSMs.
    track_negativity_bias: float = 1.0
    track_negativity_threshold: float = 0.6
    agent_negativity_bias: float = 12.0
    agent_negativity_threshold: float = 0.6

    # Flag to penalize unmatched tracks of CAV's prediction.
    penalize_unmatched_predictions: bool = False


@dataclass
class _FrameAssociation:
    matched_track_ids: List[TrackId]
    missed_track_ids_in_fov: List[TrackId]
    self_reward_track_ids: List[TrackId]
    unmatched_prediction_scores: np.ndarray


class MATEEstimator:
    """
    Multi-Agent Trust Estimator implementation.
    """

    def __init__(
        self,
        visibility_model: RangeVisibilityModel,
        config: Optional[MATEConfig] = None,
        box_transform: Optional[Any] = None,
    ):
        self.config = config or MATEConfig()
        self.box_transform = box_transform
        self.visibility_model = visibility_model

    def run_scenario(self, scenario: ScenarioData) -> ScenarioTrustResult:
        # Create empty agent/track states and trust history.
        agent_states: Dict[AgentId, BetaTrustState] = {}
        track_states: Dict[TrackId, BetaTrustState] = {}
        agent_trust_history: Dict[AgentId, List[float]] = {}
        track_trust_history: Dict[TrackId, List[float]] = {}

        # Initialize each agent state with prior.
        all_agent_ids = self._all_agent_ids(scenario)
        for agent_id in all_agent_ids:
            agent_states[agent_id] = self._new_agent_state()
            agent_trust_history[agent_id] = []

        # Iterate all frames in one scenario.
        for frame_idx, frame in enumerate(scenario.frames):
            # Step 1: propagate all currently active trust states.
            for state in agent_states.values():
                self._propagate(state)
            for state in track_states.values():
                self._propagate(state)

            # Iterate all ground truth track IDs in the current frame.
            gt_bboxes = self._as_boxes(frame.gt_bboxes)
            gt_track_ids = self._frame_gt_track_ids(frame)
            # Initialize new track state with prior.
            for track_id in gt_track_ids.tolist():
                int_track_id = int(track_id)
                if int_track_id not in track_states:
                    track_states[int_track_id] = self._new_track_state()
                    track_trust_history[int_track_id] = []

            # Build per-agent matched results for the current frame.
            frame_associations = self._build_frame_associations(frame_idx, frame.cavs, gt_bboxes, gt_track_ids)

            # Update track trust from agent trust.
            track_psms = self._build_track_psms(frame_associations, agent_states)
            for track_id, psms in track_psms.items():
                track_states[track_id].update_from_psms(
                    psms,
                    negativity_bias=self.config.track_negativity_bias,
                    negativity_threshold=self.config.track_negativity_threshold,
                )

            # Update agent trust from track trust.
            agent_psms = self._build_agent_psms(frame_associations, track_states)
            for agent_id, psms in agent_psms.items():
                agent_states[agent_id].update_from_psms(
                    psms,
                    negativity_bias=self.config.agent_negativity_bias,
                    negativity_threshold=self.config.agent_negativity_threshold,
                )

            # Append the agent/track trust history.
            for agent_id, state in agent_states.items():
                agent_trust_history[agent_id].append(state.mean)
            for track_id, state in track_states.items():
                track_trust_history.setdefault(track_id, []).append(state.mean)

        # Get the final agent/track trust scores.
        final_agent_trust = {agent_id: state.mean for agent_id, state in agent_states.items()}
        final_track_trust = {track_id: state.mean for track_id, state in track_states.items()}
        return ScenarioTrustResult(
            scenario_id=scenario.scenario_id,
            final_agent_trust=final_agent_trust,
            final_track_trust=final_track_trust,
            agent_trust_history=agent_trust_history,
            track_trust_history=track_trust_history,
        )

    # Collect the union of agent IDs that appear in any frame of the scenario.
    def _all_agent_ids(self, scenario: ScenarioData) -> List[AgentId]:
        seen: Dict[AgentId, None] = {}
        for frame in scenario.frames:
            for agent_id in frame.cavs.keys():
                seen[agent_id] = None
        return list(seen.keys())

    # Build per-agent matched and unmatched track sets used to form frame-level trust evidence.
    def _build_frame_associations(
        self,
        frame_idx: int,
        cavs: Dict[Hashable, CAVFramePrediction],
        gt_boxes: np.ndarray,
        gt_track_ids: np.ndarray,
    ) -> Dict[AgentId, _FrameAssociation]:
        frame_association: Dict[AgentId, _FrameAssociation] = {}
        for agent_id, cav in cavs.items():
            pred_boxes_global = self._to_global_boxes(agent_id, frame_idx, cav)
            pred_scores = self._prediction_scores(cav, pred_boxes_global.shape[0])

            matched_pairs = getattr(cav, "assignment_matched_pairs", None)
            unmatched_left = getattr(cav, "assignment_unmatched_left", None)
            unmatched_right = getattr(cav, "assignment_unmatched_right", None)
            if (
                matched_pairs is None
                or unmatched_left is None
                or unmatched_right is None
            ):
                raise ValueError(
                    "CAVFramePrediction is missing precomputed assignment fields."
                )

            matched_pairs = list(matched_pairs)
            unmatched_left = list(unmatched_left)
            unmatched_right = list(unmatched_right)

            matched_track_ids: List[TrackId] = []
            for _, gt_idx in matched_pairs:
                matched_track_ids.append(int(gt_track_ids[gt_idx]))

            visible_gt_ids = None
            if cav.visible_gt_ids is not None:
                try:
                    visible_gt_ids = set(np.asarray(cav.visible_gt_ids, dtype=np.int64).tolist())
                except Exception:
                    visible_gt_ids = None

            missed_track_ids_in_fov: List[TrackId] = []
            self_reward_track_ids: List[TrackId] = []
            for gt_idx in unmatched_right:
                track_id = int(gt_track_ids[gt_idx])
                if visible_gt_ids is not None and track_id not in visible_gt_ids:
                    # A CAV cannot see itself. Skip the agent PSM, but reward its own track.
                    if track_id == agent_id:
                        self_reward_track_ids.append(track_id)
                    continue
                gt_box = gt_boxes[gt_idx]
                visibility_override = getattr(cav, "visibility_override_by_track_id", None)
                if isinstance(visibility_override, dict) and track_id in visibility_override:
                    is_visible = bool(visibility_override[track_id])
                else:
                    is_visible = bool(self.visibility_model.is_visible(cav, gt_box, frame_idx))
                if is_visible:
                    missed_track_ids_in_fov.append(track_id)

            unmatched_scores = (
                pred_scores[np.array(unmatched_left, dtype=np.int64)]
                if unmatched_left
                else np.empty((0,), dtype=np.float32)
            )
            frame_association[agent_id] = _FrameAssociation(
                matched_track_ids=matched_track_ids,
                missed_track_ids_in_fov=missed_track_ids_in_fov,
                self_reward_track_ids=self_reward_track_ids,
                unmatched_prediction_scores=unmatched_scores,
            )
        return frame_association

    # Convert each agent's frame association into PSMs that update track trust states.
    def _build_track_psms(
        self,
        frame_associations: Dict[AgentId, _FrameAssociation],
        agent_states: Dict[AgentId, BetaTrustState],
    ) -> Dict[TrackId, List[PSM]]:
        track_psms: Dict[TrackId, List[PSM]] = {}
        for agent_id, association in frame_associations.items():
            agent_mean = agent_states[agent_id].mean
            confidence = self._bounded_conf(agent_mean)

            # PSM for matched tracks.
            for track_id in association.matched_track_ids:
                track_psms.setdefault(track_id, []).append(
                    PSM(value=1.0, confidence=confidence)
                )

            # PSM for unmatched track.
            for track_id in association.missed_track_ids_in_fov:
                track_psms.setdefault(track_id, []).append(
                    PSM(value=0.0, confidence=confidence)
                )

            # PSM for tracks which are ego CAVs.
            for track_id in association.self_reward_track_ids:
                track_psms.setdefault(track_id, []).append(
                    PSM(value=1.0, confidence=confidence)
                )
        return track_psms

    # Convert matched/unmatched tracks into per-agent PSMs using current track trust statistics.
    def _build_agent_psms(
        self,
        frame_associations: Dict[AgentId, _FrameAssociation],
        track_states: Dict[TrackId, BetaTrustState],
    ) -> Dict[AgentId, List[PSM]]:
        agent_psms: Dict[AgentId, List[PSM]] = {}
        for agent_id, association in frame_associations.items():
            psms: List[PSM] = []
            
            # PSM for matched tracks.
            for track_id in association.matched_track_ids:
                track_state = track_states[track_id]
                psms.append(
                    PSM(
                        value=track_state.mean,
                        confidence=self._bounded_conf(1.0 - track_state.variance),
                    )
                )

            # PSM for unmatched tracks in AGG.
            for track_id in association.missed_track_ids_in_fov:
                track_state = track_states[track_id]
                psms.append(
                    PSM(
                        value=1.0 - track_state.mean,
                        confidence=self._bounded_conf(1.0 - track_state.variance),
                    )
                )

            # PSM for unmatched tracks in the current CAV.
            if self.config.penalize_unmatched_predictions:
                for score in association.unmatched_prediction_scores.tolist():
                    psms.append(
                        PSM(
                            value=0.0,
                            confidence=self._bounded_conf(float(score)),
                        )
                    )
            agent_psms[agent_id] = psms
        return agent_psms

    # Return prediction boxes in global frame.
    def _to_global_boxes(self, agent_id: AgentId, frame_idx: int, cav: CAVFramePrediction) -> np.ndarray:
        pred_boxes = self._as_boxes(cav.pred_bboxes)
        if cav.bboxes_in_global:
            return pred_boxes
        if self.box_transform is None:
            raise ValueError(
                "Prediction boxes are not global but no box_transform callback was provided."
            )
        transformed = self.box_transform(agent_id, frame_idx, pred_boxes)
        return self._as_boxes(transformed)

    # Normalize prediction scores to a fixed length equal to the number of predicted boxes.
    def _prediction_scores(self, cav: CAVFramePrediction, n_preds: int) -> np.ndarray:
        if cav.pred_scores is None:
            return np.ones((n_preds,), dtype=np.float32)
        scores = np.asarray(cav.pred_scores, dtype=np.float32)
        if scores.shape[0] == n_preds:
            return scores
        if scores.shape[0] == 0:
            return np.ones((n_preds,), dtype=np.float32)
        resized = np.ones((n_preds,), dtype=np.float32)
        m = min(n_preds, scores.shape[0])
        resized[:m] = scores[:m]
        return resized

    # Validate and return frame ground truth track IDs aligned one-to-one with ground truth boxes.
    @staticmethod
    def _frame_gt_track_ids(frame) -> np.ndarray:
        gt_boxes = np.asarray(frame.gt_bboxes, dtype=np.float32)
        n_gt = int(gt_boxes.shape[0]) if gt_boxes.ndim == 2 else 0
        if n_gt == 0:
            return np.empty((0,), dtype=np.int64)
        if frame.gt_ids is None:
            raise ValueError(
                "frame.gt_ids is required when GT boxes are present; manual temporal GT tracking is disabled."
            )
        gt_ids = np.asarray(frame.gt_ids, dtype=np.int64).reshape(-1)
        if gt_ids.shape[0] != n_gt:
            raise ValueError(
                "gt_ids length {} does not match number of GT boxes {}.".format(
                    gt_ids.shape[0], n_gt
                )
            )
        return gt_ids

    # Create a new agent trust state initialized from configured agent priors.
    def _new_agent_state(self) -> BetaTrustState:
        return BetaTrustState(
            alpha=self.config.prior_agent_alpha,
            beta=self.config.prior_agent_beta,
            prior_alpha=self.config.prior_agent_alpha,
            prior_beta=self.config.prior_agent_beta,
        )

    # Create a new track trust state initialized from configured track priors.
    def _new_track_state(self) -> BetaTrustState:
        return BetaTrustState(
            alpha=self.config.prior_track_alpha,
            beta=self.config.prior_track_beta,
            prior_alpha=self.config.prior_track_alpha,
            prior_beta=self.config.prior_track_beta,
        )

    # Apply configured temporal propagation terms to one Beta trust state.
    def _propagate(self, state: BetaTrustState) -> None:
        state.propagate_prior_interpolation(self.config.propagation_omega)
        if self.config.propagation_delta_mu is not None:
            state.propagate_expectation(
                delta_mu=self.config.propagation_delta_mu,
                target_mu=self.config.propagation_target_mu,
            )
        if self.config.propagation_delta_nu is not None:
            state.propagate_precision(
                delta_nu=self.config.propagation_delta_nu,
                target_nu=self.config.propagation_target_nu,
            )

    # Clamp a raw confidence value to the valid [0, 1] range.
    def _bounded_conf(self, raw_conf: float) -> float:
        conf = float(raw_conf)
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return conf

    # Convert input boxes to (N, 7) and validate expected shape.
    @staticmethod
    def _as_boxes(boxes: np.ndarray) -> np.ndarray:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 7), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError(f"Expected bbox array shaped (N,7+), got {arr.shape}")
        return arr[:, :7]
