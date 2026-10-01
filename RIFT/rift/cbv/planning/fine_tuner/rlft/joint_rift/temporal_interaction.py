"""Time-aware path-interaction events for audited Joint-RIFT.

A spatial crossing is only a strong interaction when the CBV and nominal Ego
reach the conflict point at similar times.  This module keeps the original
multi-crossing/event-onset semantics while weighting each event by arrival-time
closeness.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class TemporalInteractionEvents:
    raw_event_mask: Tensor          # [G, T-1]
    raw_temporal_weight: Tensor     # [G, T-1], in [0, 1]
    clustered_event_mask: Tensor    # [G, T-1]
    weighted_onset: Tensor          # [G, T-1], event weight at cluster onset
    closest_gap_seconds: Tensor     # [G, T-1], inf for non-events


def get_temporally_weighted_interaction_events(
    cbv_traj: Tensor,
    ego_nominal: Tensor,
    dt: float = 0.1,
    arrival_time_tau: float = 1.0,
    valid_mask: Optional[Tensor] = None,
) -> TemporalInteractionEvents:
    """Return de-duplicated spatial crossings weighted by arrival-time gap.

    For every CBV segment ``t`` all intersecting Ego nominal segments are
    considered.  The temporally closest realization of the shared conflict
    point is used and receives weight ``exp(-|delta_t| / tau)``.  Consecutive
    CBV segments that hit the same conflict region are collapsed to one onset;
    the cluster keeps its maximum temporal weight while the temporal discount
    is anchored at the first segment of that cluster.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if arrival_time_tau <= 0:
        raise ValueError("arrival_time_tau must be positive")
    _validate_trajectories(cbv_traj, ego_nominal)

    intersections, candidate_fraction, ego_fraction = _all_segment_intersections(
        cbv_traj, ego_nominal
    )
    group_size, candidate_segments, ego_segments = intersections.shape

    candidate_index = torch.arange(
        candidate_segments, device=cbv_traj.device, dtype=cbv_traj.dtype
    )[None, :, None]
    ego_index = torch.arange(
        ego_segments, device=cbv_traj.device, dtype=cbv_traj.dtype
    )[None, None, :]
    candidate_arrival = candidate_index + candidate_fraction
    ego_arrival = ego_index + ego_fraction
    absolute_gap_seconds = (candidate_arrival - ego_arrival).abs() * dt
    absolute_gap_seconds = absolute_gap_seconds.masked_fill(~intersections, torch.inf)

    closest_gap_seconds, closest_index = absolute_gap_seconds.min(dim=-1)
    raw_event_mask = intersections.any(dim=-1)
    raw_temporal_weight = torch.exp(-closest_gap_seconds / arrival_time_tau)
    raw_temporal_weight = torch.where(
        raw_event_mask, raw_temporal_weight, torch.zeros_like(raw_temporal_weight)
    )

    if valid_mask is not None:
        if valid_mask.shape != (group_size,):
            raise ValueError("valid_mask must have shape [G]")
        valid_mask = valid_mask.to(dtype=torch.bool, device=cbv_traj.device)
        raw_event_mask = raw_event_mask & valid_mask[:, None]
        raw_temporal_weight = torch.where(
            valid_mask[:, None], raw_temporal_weight, torch.zeros_like(raw_temporal_weight)
        )
        closest_gap_seconds = torch.where(
            valid_mask[:, None],
            closest_gap_seconds,
            torch.full_like(closest_gap_seconds, torch.inf),
        )

    previous = torch.zeros_like(raw_event_mask)
    previous[:, 1:] = raw_event_mask[:, :-1]
    clustered_event_mask = raw_event_mask & ~previous
    # Use the temporal weight at the cluster onset. This preserves the original
    # RIFT/BeTop-inspired onset semantics and remains fully vectorized on GPU.
    weighted_onset = torch.where(
        clustered_event_mask, raw_temporal_weight, torch.zeros_like(raw_temporal_weight)
    )

    return TemporalInteractionEvents(
        raw_event_mask=raw_event_mask,
        raw_temporal_weight=raw_temporal_weight,
        clustered_event_mask=clustered_event_mask,
        weighted_onset=weighted_onset,
        closest_gap_seconds=closest_gap_seconds,
    )


def _all_segment_intersections(
    cbv_traj: Tensor, ego_nominal: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
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
