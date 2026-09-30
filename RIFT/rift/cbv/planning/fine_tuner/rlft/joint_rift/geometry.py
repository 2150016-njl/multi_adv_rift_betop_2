"""Coordinate transforms shared by all future Joint-RIFT components.

The common frame is Ego-centric and right-handed: positions are expressed
relative to the Ego rear axle and rotated by negative Ego heading; headings
are measured relative to the Ego heading.  Trajectory channels after heading
(for example speed) are preserved unchanged.
"""

import math
from typing import Union

import numpy as np
import torch


Trajectory = Union[np.ndarray, torch.Tensor]


def carla_global_to_right_handed_trajectory(trajectory: Trajectory) -> Trajectory:
    """Convert CARLA-global ``[..., x, y, heading, ...]`` to RIFT's frame.

    CARLA uses a left-handed world convention, while ``CarlaAgentState`` and
    Pluto rollout geometry use right-handed coordinates (``y`` and heading
    sign flipped).  PDM-Lite nominal plans are the only Joint-RIFT input still
    in CARLA's raw global frame, so centralise that conversion here.
    """
    _validate_trajectory(trajectory)
    if isinstance(trajectory, torch.Tensor):
        converted = trajectory.clone()
        converted[..., 1] = -converted[..., 1]
        converted[..., 2] = _wrap_angle(-converted[..., 2])
        return converted

    converted = trajectory.copy()
    converted[..., 1] = -converted[..., 1]
    converted[..., 2] = _wrap_angle(-converted[..., 2])
    return converted


def local_to_global_trajectory(trajectory: Trajectory, source_state) -> Trajectory:
    """Transform a source-actor local ``[..., x, y, heading, ...]`` trajectory."""
    _validate_trajectory(trajectory)
    origin, heading = _state_origin_heading(source_state)

    if isinstance(trajectory, torch.Tensor):
        origin = trajectory.new_tensor(origin)
        heading = trajectory.new_tensor(heading)
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        local_xy = trajectory[..., :2]
        global_xy = torch.stack(
            (
                local_xy[..., 0] * cos_heading - local_xy[..., 1] * sin_heading,
                local_xy[..., 0] * sin_heading + local_xy[..., 1] * cos_heading,
            ),
            dim=-1,
        ) + origin
        global_heading = _wrap_angle(trajectory[..., 2] + heading)
        return torch.cat((global_xy, global_heading.unsqueeze(-1), trajectory[..., 3:]), dim=-1)

    cos_heading, sin_heading = np.cos(heading), np.sin(heading)
    local_xy = trajectory[..., :2]
    global_xy = np.stack(
        (
            local_xy[..., 0] * cos_heading - local_xy[..., 1] * sin_heading,
            local_xy[..., 0] * sin_heading + local_xy[..., 1] * cos_heading,
        ),
        axis=-1,
    ) + origin
    global_heading = _wrap_angle(trajectory[..., 2] + heading)
    return np.concatenate((global_xy, global_heading[..., None], trajectory[..., 3:]), axis=-1)


def global_to_common_frame_trajectory(trajectory: Trajectory, ego_state) -> Trajectory:
    """Transform a global ``[..., x, y, heading, ...]`` trajectory to Ego frame."""
    _validate_trajectory(trajectory)
    ego_origin, ego_heading = _state_origin_heading(ego_state)

    if isinstance(trajectory, torch.Tensor):
        ego_origin = trajectory.new_tensor(ego_origin)
        ego_heading = trajectory.new_tensor(ego_heading)
        cos_heading, sin_heading = torch.cos(ego_heading), torch.sin(ego_heading)
        delta_xy = trajectory[..., :2] - ego_origin
        common_xy = torch.stack(
            (
                delta_xy[..., 0] * cos_heading + delta_xy[..., 1] * sin_heading,
                -delta_xy[..., 0] * sin_heading + delta_xy[..., 1] * cos_heading,
            ),
            dim=-1,
        )
        common_heading = _wrap_angle(trajectory[..., 2] - ego_heading)
        return torch.cat((common_xy, common_heading.unsqueeze(-1), trajectory[..., 3:]), dim=-1)

    cos_heading, sin_heading = np.cos(ego_heading), np.sin(ego_heading)
    delta_xy = trajectory[..., :2] - ego_origin
    common_xy = np.stack(
        (
            delta_xy[..., 0] * cos_heading + delta_xy[..., 1] * sin_heading,
            -delta_xy[..., 0] * sin_heading + delta_xy[..., 1] * cos_heading,
        ),
        axis=-1,
    )
    common_heading = _wrap_angle(trajectory[..., 2] - ego_heading)
    return np.concatenate((common_xy, common_heading[..., None], trajectory[..., 3:]), axis=-1)


def to_common_frame_trajectory(
    trajectory: Trajectory,
    source_state,
    ego_state,
) -> Trajectory:
    """Transform a CBV-local candidate trajectory into the Ego common frame."""
    return global_to_common_frame_trajectory(
        local_to_global_trajectory(trajectory, source_state), ego_state
    )


def _state_origin_heading(state):
    rear_axle = state.rear_axle
    return np.asarray(rear_axle.array[:2], dtype=np.float32), float(rear_axle.heading)


def _validate_trajectory(trajectory: Trajectory) -> None:
    if trajectory.ndim < 1 or trajectory.shape[-1] < 3:
        raise ValueError("trajectory must have shape [..., C] with C >= 3")


def _wrap_angle(angle: Trajectory) -> Trajectory:
    if isinstance(angle, torch.Tensor):
        return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
    return np.remainder(angle + math.pi, 2 * math.pi) - math.pi

