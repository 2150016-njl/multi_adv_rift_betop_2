"""Fixed-Ego-frame lateral/time braids with signed longitudinal order.

Inspired by BeTop's lateral/time projection. Adds an explicit locality gate;
it is not a copy of BeTopNet, nor evidence of actual yielding/causal response.
"""

from dataclasses import dataclass
import torch


@dataclass
class NominalBraidEvents:
    raw_event_mask: torch.Tensor
    weighted_onset: torch.Tensor
    signed_onset: torch.Tensor


def nominal_braid_events(cbv, ego, ego_heading, radius=20., distance_tau=5.):
    if cbv.ndim != 3 or ego.ndim != 2 or cbv.shape[1] != ego.shape[0] or cbv.shape[1] < 2:
        raise ValueError("expected [G,T,C], [T,C] with matching T>=2")
    if radius <= 0 or distance_tau <= 0:
        raise ValueError("braid locality scales must be positive")
    if cbv.device != ego.device or not cbv.is_floating_point() or not ego.is_floating_point():
        raise ValueError("braid inputs must be floating tensors on one device")
    if not torch.isfinite(cbv).all() or not torch.isfinite(ego).all():
        raise ValueError("nonfinite braid trajectories")
    heading = torch.as_tensor(ego_heading, dtype=cbv.dtype, device=cbv.device)
    relative = cbv[..., :2] - ego[None, :, :2]
    longitudinal = relative[..., 0] * heading.cos() + relative[..., 1] * heading.sin()
    lateral = -relative[..., 0] * heading.sin() + relative[..., 1] * heading.cos()
    start, end = lateral[:, :-1], lateral[:, 1:]
    denominator = start - end
    moving = denominator.abs() > torch.finfo(cbv.dtype).eps
    fraction = start / torch.where(moving, denominator, torch.ones_like(denominator))
    at_crossing = longitudinal[:, :-1] + fraction * (longitudinal[:, 1:] - longitudinal[:, :-1])
    raw = moving & (fraction >= 0) & (fraction <= 1) & (at_crossing.abs() <= radius)
    previous = torch.zeros_like(raw)
    previous[:, 1:] = raw[:, :-1]
    onset = raw & ~previous
    weight = torch.exp(-at_crossing.abs() / distance_tau)
    weight = torch.where(onset, weight, torch.zeros_like(weight))
    sign = torch.where(onset, at_crossing.sign(), torch.zeros_like(at_crossing))
    return NominalBraidEvents(raw, weight, sign)
