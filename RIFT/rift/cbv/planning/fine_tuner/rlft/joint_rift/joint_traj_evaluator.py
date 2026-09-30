"""Joint candidate evaluation built on the existing RIFT ``TrajEvaluator``.

The unary rollout, lane relation, comfort, off-road, and background collision
logic remains in the established evaluator.  This module only adds the three
Joint-RIFT concerns: PDM nominal-Ego collision, event-time interaction
pressure, and pair-candidate collision/coalition aggregation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from rift.cbv.planning.fine_tuner.rlft.joint_rift.interaction_topology import (
    discounted_interaction_return,
    get_interaction_events,
)
from rift.cbv.planning.fine_tuner.rlft.traj_eval.traj_evaluator import (
    TrajEvaluator,
    compute_agents_vertices,
)


ArrayLike = Union[np.ndarray, Tensor]


@dataclass
class UnaryEvaluation:
    """One controlled-CBV evaluation, excluding the other controlled CBV."""

    rollout_return: np.ndarray       # [G]
    interaction_trace: np.ndarray    # [G, T], onset impulses
    interaction_pressure: np.ndarray # [G]
    rollout_vertices: np.ndarray     # [G, T, 4, 2]
    collision_matrix: np.ndarray     # [G, T], background + nominal Ego
    off_road_matrix: np.ndarray      # [G, T]


class JointTrajEvaluator(TrajEvaluator):
    """Compose two RIFT unary evaluations into one candidate-pair objective."""

    def __init__(
        self,
        dt: float = 0.1,
        num_frames: int = 40,
        gamma: float = 0.98,
        lambda_pressure: float = 1.0,
        lambda_coalition: float = 1.0,
        lambda_pair_collision: float = 1.0,
        pair_collision_chunk_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt, num_frames=num_frames, **kwargs)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if pair_collision_chunk_size < 1:
            raise ValueError("pair_collision_chunk_size must be positive")
        self.gamma = gamma
        self.lambda_pressure = lambda_pressure
        self.lambda_coalition = lambda_coalition
        self.lambda_pair_collision = lambda_pair_collision
        self.pair_collision_chunk_size = pair_collision_chunk_size

    def evaluate_unary(
        self,
        center_history_states: Sequence[Any],
        raw_trajectories: Tensor,
        reference_line_positions: Tensor,
        reference_line_orientations: Tensor,
        reference_line_valid_mask: Tensor,
        background_actors: Sequence[Any],
        ego_nominal_trajectory: ArrayLike,
        ego_history_states: Sequence[Any],
    ) -> UnaryEvaluation:
        """Evaluate one full candidate group using RIFT's existing rollout.

        ``background_actors`` must already exclude both the Ego and the other
        controlled CBV.  The supplied PDM nominal trajectory replaces the
        ordinary Ego forward prediction in the unary collision term.
        """
        self._validate_unary_inputs(
            raw_trajectories,
            reference_line_positions,
            reference_line_orientations,
            reference_line_valid_mask,
        )
        trajectories = raw_trajectories[:, :self.num_frames]
        reference_positions = [
            position[mask]
            for position, mask in zip(reference_line_positions, reference_line_valid_mask)
        ]
        reference_orientations = [
            orientation[mask]
            for orientation, mask in zip(reference_line_orientations, reference_line_valid_mask)
        ]
        if any(position.shape[0] == 0 for position in reference_positions):
            raise ValueError("every candidate needs at least one valid reference-line point")

        # TrajEvaluator expects [R, M, T, C].  Treat every flattened Joint-RIFT
        # candidate as an R entry with a single mode; this reuses its lane and
        # rollout machinery without copying it.
        grouped_trajectories = trajectories.unsqueeze(1)
        delta_dis, delta_angle = self.get_ref_line_info(
            grouped_trajectories, reference_positions, reference_orientations
        )
        self.center_rollout_model.batch_pid_controller.reset()
        (
            rollout_center,
            rollout_angle,
            rollout_speed,
            rollout_acc,
            rollout_angular_vel,
            rollout_angular_acc,
            rollout_vertices,
        ) = self.get_center_rollout(grouped_trajectories, center_history_states)
        # TrackPropagate has an 80-step internal default in the existing RIFT
        # evaluator.  Joint-RIFT's documented horizon is 40, so explicitly
        # align every downstream signal to this evaluator's configured horizon.
        rollout_center = rollout_center[:, :self.num_frames]
        rollout_angle = rollout_angle[:, :self.num_frames]
        rollout_speed = rollout_speed[:, :self.num_frames]
        rollout_acc = rollout_acc[:, :self.num_frames]
        rollout_angular_vel = rollout_angular_vel[:, :self.num_frames]
        rollout_angular_acc = rollout_angular_acc[:, :self.num_frames]
        rollout_vertices = rollout_vertices[:, :self.num_frames]

        background_vertices = self.get_other_vehicle_rollout(
            background_actors, num_future_frames=self.num_frames
        )
        aligned_ego_nominal = self._align_nominal_to_rollout(
            ego_nominal_trajectory, ego_history_states[-1]
        )
        ego_vertices = self._nominal_ego_vertices(
            aligned_ego_nominal, ego_history_states[-1]
        )
        collision_obstacles = np.concatenate((background_vertices, ego_vertices), axis=0)
        collision_matrix = self.get_collision_matrix(rollout_vertices, collision_obstacles)
        off_road_matrix = self.get_off_road_matrix(rollout_center, center_history_states[-1])
        rollout_return = self.get_rollout_return(
            delta_dis,
            delta_angle,
            rollout_speed,
            rollout_acc,
            rollout_angular_vel,
            rollout_angular_acc,
            collision_matrix,
            off_road_matrix,
            gamma=self.gamma,
        )

        rollout_center_tensor = torch.as_tensor(rollout_center, dtype=trajectories.dtype)
        ego_nominal_tensor = torch.as_tensor(
            aligned_ego_nominal[:, :2],
            dtype=trajectories.dtype,
        )
        events = get_interaction_events(rollout_center_tensor, ego_nominal_tensor)
        interaction_trace = torch.nn.functional.pad(
            events.clustered_event_mask.to(dtype=trajectories.dtype), (0, 1)
        )
        interaction_pressure = discounted_interaction_return(interaction_trace, self.gamma)
        return UnaryEvaluation(
            rollout_return=np.asarray(rollout_return, dtype=np.float64),
            interaction_trace=interaction_trace.cpu().numpy(),
            interaction_pressure=interaction_pressure.cpu().numpy(),
            rollout_vertices=rollout_vertices,
            collision_matrix=collision_matrix,
            off_road_matrix=off_road_matrix,
        )

    def evaluate_joint(
        self,
        unary_1_kwargs: Dict[str, Any],
        unary_2_kwargs: Dict[str, Any],
        valid_mask_1: Optional[np.ndarray] = None,
        valid_mask_2: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run both independent unary rollouts and aggregate candidate pairs."""
        unary_1 = self.evaluate_unary(**unary_1_kwargs)
        unary_2 = self.evaluate_unary(**unary_2_kwargs)
        return self.combine_unary_evaluations(
            unary_1, unary_2, valid_mask_1=valid_mask_1, valid_mask_2=valid_mask_2
        )

    def combine_unary_evaluations(
        self,
        unary_1: UnaryEvaluation,
        unary_2: UnaryEvaluation,
        valid_mask_1: Optional[np.ndarray] = None,
        valid_mask_2: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Broadcast unary results into ``Q[i,j]`` without pairwise rollouts."""
        trace_1 = np.asarray(unary_1.interaction_trace, dtype=np.float64)
        trace_2 = np.asarray(unary_2.interaction_trace, dtype=np.float64)
        if trace_1.ndim != 2 or trace_2.ndim != 2 or trace_1.shape[1] != trace_2.shape[1]:
            raise ValueError("unary interaction traces must have shapes [G, T] with a shared horizon")
        group_1, horizon = trace_1.shape
        group_2 = trace_2.shape[0]
        if unary_1.rollout_return.shape != (group_1,) or unary_2.rollout_return.shape != (group_2,):
            raise ValueError("unary return shape must match its interaction trace")
        valid_mask_1 = _normalise_candidate_mask(valid_mask_1, group_1)
        valid_mask_2 = _normalise_candidate_mask(valid_mask_2, group_2)
        valid_mask = valid_mask_1[:, None] & valid_mask_2[None, :]
        if not valid_mask.any():
            raise ValueError("at least one candidate pair must be valid")

        temporal_weight = np.power(self.gamma, np.arange(horizon, dtype=np.float64))
        pressure_1 = (trace_1 * temporal_weight).sum(axis=-1)
        pressure_2 = (trace_2 * temporal_weight).sum(axis=-1)
        union_trace = 1.0 - (1.0 - trace_1[:, None, :]) * (1.0 - trace_2[None, :, :])
        p12 = (union_trace * temporal_weight).sum(axis=-1)
        coalition_gain = p12 - np.maximum(pressure_1[:, None], pressure_2[None, :])
        pair_collision = self.pair_collision_from_vertices(
            unary_1.rollout_vertices,
            unary_2.rollout_vertices,
            chunk_size=self.pair_collision_chunk_size,
        )
        joint_q = (
            np.asarray(unary_1.rollout_return)[:, None]
            + np.asarray(unary_2.rollout_return)[None, :]
            + self.lambda_pressure * p12
            + self.lambda_coalition * coalition_gain
            - self.lambda_pair_collision * pair_collision.astype(np.float64)
        )
        joint_advantage = _masked_standardise(joint_q, valid_mask)
        return {
            "joint_advantage": joint_advantage,
            "joint_q": joint_q,
            "p12": p12,
            "coalition_gain": coalition_gain,
            "pair_collision": pair_collision,
            "valid_mask": valid_mask,
            "unary_return_1": np.asarray(unary_1.rollout_return),
            "unary_return_2": np.asarray(unary_2.rollout_return),
            "interaction_trace_1": trace_1,
            "interaction_trace_2": trace_2,
            "p1": pressure_1,
            "p2": pressure_2,
        }

    @staticmethod
    def pair_collision_from_vertices(
        vertices_1: np.ndarray,
        vertices_2: np.ndarray,
        chunk_size: int = 32,
    ) -> np.ndarray:
        """Return exact oriented-box pair collisions with chunked SAT tests.

        The only loop is over first-group chunks.  No ``(i, j)`` pair is
        re-rolled-out or evaluated in Python.
        """
        vertices_1 = np.asarray(vertices_1, dtype=np.float64)
        vertices_2 = np.asarray(vertices_2, dtype=np.float64)
        if vertices_1.ndim != 4 or vertices_2.ndim != 4:
            raise ValueError("vertices must have shapes [G, T, 4, 2]")
        if vertices_1.shape[1:] != vertices_2.shape[1:]:
            raise ValueError("both candidate groups need matching [T, 4, 2] vertices")
        if vertices_1.shape[2:] != (4, 2):
            raise ValueError("collision geometry requires four 2D box vertices")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        group_1, horizon = vertices_1.shape[:2]
        group_2 = vertices_2.shape[0]
        collision = np.zeros((group_1, group_2), dtype=np.bool_)
        for start in range(0, group_1, chunk_size):
            stop = min(start + chunk_size, group_1)
            first = np.broadcast_to(
                vertices_1[start:stop, None], (stop - start, group_2, horizon, 4, 2)
            )
            second = np.broadcast_to(
                vertices_2[None], (stop - start, group_2, horizon, 4, 2)
            )
            axes = np.concatenate((_box_axes(first), _box_axes(second)), axis=-2)
            first_projection = np.einsum("...vc,...ac->...va", first, axes)
            second_projection = np.einsum("...vc,...ac->...va", second, axes)
            separated = (
                (first_projection.max(axis=-2) < second_projection.min(axis=-2))
                | (second_projection.max(axis=-2) < first_projection.min(axis=-2))
            )
            collision[start:stop] = (~separated.any(axis=-1)).any(axis=-1)
        return collision

    def _nominal_ego_vertices(self, ego_nominal_trajectory: ArrayLike, ego_state: Any) -> np.ndarray:
        nominal = _to_numpy(ego_nominal_trajectory)
        if nominal.ndim != 2 or nominal.shape[0] < self.num_frames or nominal.shape[1] < 3:
            raise ValueError("ego nominal trajectory must have shape [T>=num_frames, x,y,heading,...]")
        centers = nominal[:self.num_frames, :2][None]
        headings = nominal[:self.num_frames, 2][None]
        ego_shape = np.array(
            [[ego_state.car_footprint.width, ego_state.car_footprint.length]], dtype=np.float64
        )
        return compute_agents_vertices(centers, headings, ego_shape)

    def _align_nominal_to_rollout(
        self,
        ego_nominal_trajectory: ArrayLike,
        ego_state: Any,
    ) -> np.ndarray:
        """Align PDM's first future sample with RIFT rollout's current sample.

        ``TrackPropagate`` includes the current CBV state at index zero, while
        the Step 1 PDM adapter intentionally starts one ``dt`` in the future.
        Prepending the current right-handed Ego rear-axle state avoids a
        one-tick phase error in both nominal collision and topology events.
        """
        nominal = _to_numpy(ego_nominal_trajectory)
        if nominal.ndim != 2 or nominal.shape[0] < self.num_frames or nominal.shape[1] < 3:
            raise ValueError("ego nominal trajectory must have shape [T>=num_frames, x,y,heading,...]")
        aligned = nominal[:self.num_frames].copy()
        aligned[1:] = nominal[:self.num_frames - 1]
        aligned[0, :2] = ego_state.rear_axle.array[:2]
        aligned[0, 2] = ego_state.rear_axle.heading
        if aligned.shape[1] > 3:
            aligned[0, 3] = ego_state.dynamic_car_state.speed
        return aligned

    def _validate_unary_inputs(
        self,
        raw_trajectories: Tensor,
        reference_line_positions: Tensor,
        reference_line_orientations: Tensor,
        reference_line_valid_mask: Tensor,
    ) -> None:
        if raw_trajectories.ndim != 3 or raw_trajectories.shape[-1] < 6:
            raise ValueError("raw_trajectories must have shape [G, T, 6]")
        if raw_trajectories.shape[1] < self.num_frames:
            raise ValueError("raw trajectories are shorter than the evaluator horizon")
        group_size = raw_trajectories.shape[0]
        if reference_line_positions.shape[:2] != reference_line_orientations.shape:
            raise ValueError("reference-line positions and orientations must share [G, P]")
        if reference_line_valid_mask.shape != reference_line_orientations.shape:
            raise ValueError("reference-line valid mask must have shape [G, P]")
        if reference_line_positions.shape[0] != group_size or reference_line_positions.shape[-1] != 2:
            raise ValueError("reference-line positions must have shape [G, P, 2]")


def _box_axes(vertices: np.ndarray) -> np.ndarray:
    """Two separating axes per rectangle, perpendicular to adjacent edges."""
    edges = np.stack((vertices[..., 1, :] - vertices[..., 0, :], vertices[..., 2, :] - vertices[..., 1, :]), axis=-2)
    return np.stack((-edges[..., 1], edges[..., 0]), axis=-1)


def _normalise_candidate_mask(mask: Optional[np.ndarray], group_size: int) -> np.ndarray:
    if mask is None:
        return np.ones(group_size, dtype=np.bool_)
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.shape != (group_size,):
        raise ValueError("candidate mask must have shape [G]")
    return mask


def _masked_standardise(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """State-local normalisation; invalid pairs remain exactly zero."""
    valid_values = values[valid_mask]
    mean = valid_values.mean()
    std = valid_values.std()
    standardised = np.zeros_like(values, dtype=np.float64)
    standardised[valid_mask] = (valid_values - mean) / (std + 1e-5)
    return standardised


def _to_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
