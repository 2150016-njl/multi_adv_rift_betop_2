"""Audited Joint-RIFT evaluator with time-aware interaction pressure."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_traj_evaluator import (
    JointTrajEvaluator,
    UnaryEvaluation,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.temporal_interaction import (
    get_temporally_weighted_interaction_events,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.interaction_topology import (
    discounted_interaction_return,
)


class AuditedJointTrajEvaluator(JointTrajEvaluator):
    """Keep RIFT unary returns but require temporal proximity at conflict points."""

    def __init__(
        self,
        *args,
        arrival_time_tau: float = 1.0,
        both_necessary_epsilon: float = 0.1,
        **kwargs,
    ) -> None:
        interaction_dt = float(kwargs.get("dt", 0.1))
        super().__init__(*args, **kwargs)
        if arrival_time_tau <= 0:
            raise ValueError("arrival_time_tau must be positive")
        if both_necessary_epsilon < 0:
            raise ValueError("both_necessary_epsilon must be non-negative")
        self.arrival_time_tau = float(arrival_time_tau)
        self.both_necessary_epsilon = float(both_necessary_epsilon)
        self.interaction_dt = interaction_dt

    def evaluate_unary(
        self,
        center_history_states: Sequence[Any],
        raw_trajectories: torch.Tensor,
        reference_line_positions: torch.Tensor,
        reference_line_orientations: torch.Tensor,
        reference_line_valid_mask: torch.Tensor,
        background_actors: Sequence[Any],
        ego_nominal_trajectory,
        ego_history_states: Sequence[Any],
    ) -> UnaryEvaluation:
        """Evaluate one group with the original RIFT rollout and audited topology.

        The code deliberately mirrors ``JointTrajEvaluator.evaluate_unary`` so
        lane, comfort, off-road, background collision, and nominal-Ego collision
        semantics stay identical. Only the interaction trace changes from a
        binary spatial crossing to an arrival-time-weighted crossing.
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
            for position, mask in zip(
                reference_line_positions, reference_line_valid_mask
            )
        ]
        reference_orientations = [
            orientation[mask]
            for orientation, mask in zip(
                reference_line_orientations, reference_line_valid_mask
            )
        ]
        if any(position.shape[0] == 0 for position in reference_positions):
            raise ValueError("every candidate needs at least one valid reference-line point")

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
        off_road_matrix = self.get_off_road_matrix(
            rollout_center, center_history_states[-1]
        )
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

        rollout_center_tensor = torch.as_tensor(
            rollout_center,
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        ego_nominal_tensor = torch.as_tensor(
            aligned_ego_nominal[:, :2],
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        events = get_temporally_weighted_interaction_events(
            rollout_center_tensor,
            ego_nominal_tensor,
            dt=self.interaction_dt,
            arrival_time_tau=self.arrival_time_tau,
        )
        interaction_trace = torch.nn.functional.pad(events.weighted_onset, (0, 1))
        interaction_pressure = discounted_interaction_return(
            interaction_trace, self.gamma
        )
        return UnaryEvaluation(
            rollout_return=np.asarray(rollout_return, dtype=np.float64),
            interaction_trace=interaction_trace.detach().cpu().numpy(),
            interaction_pressure=interaction_pressure.detach().cpu().numpy(),
            rollout_vertices=rollout_vertices,
            collision_matrix=collision_matrix,
            off_road_matrix=off_road_matrix,
        )

    def combine_unary_evaluations(
        self,
        unary_1: UnaryEvaluation,
        unary_2: UnaryEvaluation,
        valid_mask_1: Optional[np.ndarray] = None,
        valid_mask_2: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        result = super().combine_unary_evaluations(
            unary_1,
            unary_2,
            valid_mask_1=valid_mask_1,
            valid_mask_2=valid_mask_2,
        )
        p1 = np.asarray(result["p1"], dtype=np.float64)
        p2 = np.asarray(result["p2"], dtype=np.float64)
        p12 = np.asarray(result["p12"], dtype=np.float64)
        valid_mask = np.asarray(result["valid_mask"], dtype=bool)
        marginal_1 = p12 - p2[None, :]
        marginal_2 = p12 - p1[:, None]
        both_necessary = (
            valid_mask
            & (marginal_1 > self.both_necessary_epsilon)
            & (marginal_2 > self.both_necessary_epsilon)
        )
        result.update(
            {
                "marginal_1": marginal_1,
                "marginal_2": marginal_2,
                "both_necessary": both_necessary,
            }
        )
        return result
