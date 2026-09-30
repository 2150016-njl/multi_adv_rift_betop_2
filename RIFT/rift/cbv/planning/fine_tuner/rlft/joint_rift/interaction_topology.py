"""Event-time trajectory topology utilities for Joint-RIFT.

This is a small, self-contained segment-crossing implementation inspired by
BeTop's low-level braid signal.  It intentionally preserves the candidate
segment time axis instead of reducing all crossings to one trajectory-level
edge.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class InteractionEvents:
    """Raw and de-duplicated candidate-vs-Ego crossing events."""

    raw_event_mask: Tensor        # [G, T - 1]
    clustered_event_mask: Tensor  # [G, T - 1], one impulse per True cluster


def get_interaction_events(
    cbv_traj: Tensor,
    ego_nominal: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> InteractionEvents:
    """Find spatial segment crossings for each candidate trajectory.

    A candidate segment at time ``t`` is checked against every Ego nominal
    segment.  This intentionally detects a shared conflict point even when
    the two vehicles arrive at it at different times; synchronous occupancy
    conflicts remain part of the rollout collision evaluator.  Consecutive
    raw hits are collapsed to their first (onset) time.
    """
    _validate_trajectories(cbv_traj, ego_nominal)
    intersections, _, _ = _all_segment_intersections(cbv_traj, ego_nominal)
    raw_event_mask = intersections.any(dim=-1)

    if valid_mask is not None:
        if valid_mask.shape != raw_event_mask.shape[:1]:
            raise ValueError("valid_mask must have shape [G]")
        raw_event_mask = raw_event_mask & valid_mask.to(
            dtype=torch.bool, device=raw_event_mask.device
        ).unsqueeze(-1)

    previous = torch.zeros_like(raw_event_mask)
    previous[:, 1:] = raw_event_mask[:, :-1]
    clustered_event_mask = raw_event_mask & ~previous
    return InteractionEvents(raw_event_mask, clustered_event_mask)


def discounted_interaction_return(
    event_mask: Tensor,
    gamma: float = 0.98,
) -> Tensor:
    """Accumulate onset events using the RIFT-compatible temporal discount."""
    if event_mask.ndim != 2:
        raise ValueError("event_mask must have shape [G, T]")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    dtype = event_mask.dtype if event_mask.is_floating_point() else torch.float32
    weights = torch.pow(
        torch.tensor(gamma, device=event_mask.device, dtype=dtype),
        torch.arange(event_mask.shape[-1], device=event_mask.device, dtype=dtype),
    )
    return (event_mask.to(dtype=weights.dtype) * weights).sum(dim=-1)


def get_arrival_order_label(
    cbv_traj: Tensor,
    ego_nominal: Tensor,
    valid_mask: Optional[Tensor] = None,
    arrival_tolerance_steps: float = 1.0,
) -> Tensor:
    """Label each candidate crossing as early (+1), late (-1), or tied (0).

    The output is ``[G, T-1]`` and is deliberately diagnostic-only.  If a
    candidate segment crosses multiple Ego segments, the earliest Ego arrival
    is used.  Non-events are zero.
    """
    if arrival_tolerance_steps < 0:
        raise ValueError("arrival_tolerance_steps must be non-negative")
    _validate_trajectories(cbv_traj, ego_nominal)
    intersections, candidate_fraction, ego_fraction = _all_segment_intersections(
        cbv_traj, ego_nominal
    )
    ego_segment_index = torch.arange(
        ego_nominal.shape[0] - 1, device=cbv_traj.device, dtype=cbv_traj.dtype
    )
    ego_arrival = ego_segment_index[None, None, :] + ego_fraction
    ego_arrival = ego_arrival.masked_fill(~intersections, torch.inf)
    earliest_ego_arrival, earliest_ego_index = ego_arrival.min(dim=-1)
    has_event = intersections.any(dim=-1)

    candidate_segment_index = torch.arange(
        cbv_traj.shape[1] - 1, device=cbv_traj.device, dtype=cbv_traj.dtype
    )
    candidate_fraction = candidate_fraction.gather(
        dim=-1, index=earliest_ego_index.unsqueeze(-1)
    ).squeeze(-1)
    candidate_arrival = candidate_segment_index[None, :] + candidate_fraction
    candidate_arrival = torch.where(has_event, candidate_arrival, torch.zeros_like(candidate_arrival))
    gap = candidate_arrival - earliest_ego_arrival
    label = torch.where(
        gap < -arrival_tolerance_steps,
        torch.ones_like(gap, dtype=torch.int8),
        torch.where(
            gap > arrival_tolerance_steps,
            -torch.ones_like(gap, dtype=torch.int8),
            torch.zeros_like(gap, dtype=torch.int8),
        ),
    )
    label = torch.where(has_event, label, torch.zeros_like(label))
    if valid_mask is not None:
        if valid_mask.shape != label.shape[:1]:
            raise ValueError("valid_mask must have shape [G]")
        label = torch.where(
            valid_mask.to(dtype=torch.bool, device=label.device).unsqueeze(-1),
            label,
            torch.zeros_like(label),
        )
    return label


def _all_segment_intersections(cbv_traj: Tensor, ego_nominal: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return crossings and along-segment fractions, all shaped ``[G,T-1,T-1]``."""
    candidate_start = cbv_traj[:, :-1, None, :2]
    candidate_end = cbv_traj[:, 1:, None, :2]
    ego_start = ego_nominal[None, None, :-1, :2]
    ego_end = ego_nominal[None, None, 1:, :2]

    candidate_vector = candidate_end - candidate_start
    ego_vector = ego_end - ego_start
    offset = ego_start - candidate_start
    determinant = _cross_2d(candidate_vector, ego_vector)
    non_parallel = determinant.abs() > torch.finfo(cbv_traj.dtype).eps
    safe_determinant = torch.where(non_parallel, determinant, torch.ones_like(determinant))
    candidate_fraction = _cross_2d(offset, ego_vector) / safe_determinant
    ego_fraction = _cross_2d(offset, candidate_vector) / safe_determinant
    intersections = (
        non_parallel
        & (candidate_fraction >= 0)
        & (candidate_fraction <= 1)
        & (ego_fraction >= 0)
        & (ego_fraction <= 1)
    )
    return intersections, candidate_fraction, ego_fraction


def _validate_trajectories(cbv_traj: Tensor, ego_nominal: Tensor) -> None:
    if cbv_traj.ndim != 3 or cbv_traj.shape[-1] < 2:
        raise ValueError("cbv_traj must have shape [G, T, C] with C >= 2")
    if ego_nominal.ndim != 2 or ego_nominal.shape[-1] < 2:
        raise ValueError("ego_nominal must have shape [T, C] with C >= 2")
    if cbv_traj.shape[1] != ego_nominal.shape[0]:
        raise ValueError("CBV and Ego trajectories must have the same horizon")
    if cbv_traj.shape[1] < 2:
        raise ValueError("trajectory horizon must contain at least two points")
    if not cbv_traj.is_floating_point() or not ego_nominal.is_floating_point():
        raise TypeError("trajectory tensors must use a floating-point dtype")
    if cbv_traj.device != ego_nominal.device:
        raise ValueError("CBV and Ego trajectories must be on the same device")


def _cross_2d(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
