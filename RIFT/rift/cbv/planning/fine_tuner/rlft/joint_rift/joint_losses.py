# Simulator-independent Joint-RIFT group objectives.
from typing import Dict, Tuple
import torch
from torch import Tensor

def masked_joint_log_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
    """Categorical log-probabilities over valid flattened joint actions only."""
    if logits.shape != valid_mask.shape or logits.ndim != 3:
        raise ValueError("logits and valid_mask must both have shape [B, G1, G2]")
    if not valid_mask.flatten(1).any(dim=1).all():
        raise ValueError("every joint state requires at least one valid pair")
    flattened_logits = logits.masked_fill(~valid_mask, -torch.inf).flatten(1)
    return torch.log_softmax(flattened_logits, dim=-1).view_as(logits)


def joint_dual_clip_loss(
    new_joint_logits: Tensor,
    old_joint_logits: Tensor,
    joint_advantage: Tensor,
    valid_mask: Tensor,
    state_weight: Tensor = None,
    clip_ratio: float = 0.2,
    dual_clip: float = 3.0,
    pair_weighting: str = "uniform",
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute state-normalized RIFT dual-clip PPO.

    ``state_weight`` is optional and applies only to complete joint states;
    it never changes the relative weighting of candidate pairs within a state.
    """
    if not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must be in (0, 1)")
    if dual_clip < 1:
        raise ValueError("dual_clip must be at least one")
    if not (
        new_joint_logits.shape == old_joint_logits.shape == joint_advantage.shape == valid_mask.shape
    ):
        raise ValueError("joint logits, advantages, and valid mask must share [B, G1, G2]")

    new_logp = masked_joint_log_softmax(new_joint_logits, valid_mask)
    old_logp = masked_joint_log_softmax(old_joint_logits, valid_mask)
    # Avoid forming ``-inf - -inf`` for padding entries.  Masking only the
    # result can leave a finite forward value while still poisoning autograd.
    safe_new_logp = torch.where(valid_mask, new_logp, torch.zeros_like(new_logp))
    safe_old_logp = torch.where(valid_mask, old_logp, torch.zeros_like(old_logp))
    logp_delta = safe_new_logp - safe_old_logp
    ratio = torch.exp(logp_delta)
    unclipped = joint_advantage * ratio
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    clipped = joint_advantage * clipped_ratio
    minimum = torch.minimum(unclipped, clipped)
    dual_clipped = torch.maximum(minimum, joint_advantage * dual_clip)
    objective = torch.where(joint_advantage < 0, dual_clipped, minimum)
    objective = torch.where(valid_mask, objective, torch.zeros_like(objective))

    valid_count = valid_mask.sum(dim=(1, 2)).clamp_min(1)
    if pair_weighting == "uniform":
        state_objective = objective.sum(dim=(1, 2)) / valid_count
    elif pair_weighting == "old_policy":
        old_probability = torch.where(valid_mask, safe_old_logp.exp(), torch.zeros_like(safe_old_logp)).detach()
        state_objective = (objective * old_probability).sum(dim=(1, 2))
    else:
        raise ValueError("pair_weighting must be uniform or old_policy")
    if state_weight is None:
        state_weight = torch.ones_like(state_objective)
    else:
        state_weight = state_weight.to(
            device=state_objective.device, dtype=state_objective.dtype
        ).reshape(-1)
        if state_weight.shape != state_objective.shape:
            raise ValueError("state_weight must have shape [B]")
        if not torch.isfinite(state_weight).all() or (state_weight <= 0).any():
            raise ValueError("state_weight must be finite and strictly positive")
    loss = -(state_objective * state_weight).sum() / state_weight.sum()

    probability = torch.where(valid_mask, safe_new_logp.exp(), torch.zeros_like(safe_new_logp))
    entropy = -(probability * safe_new_logp).sum(dim=(1, 2)).mean()
    valid_ratio = ratio[valid_mask]
    metrics = {
        "policy_entropy": entropy.detach(),
        "ratio_mean": valid_ratio.mean().detach(),
        "ratio_clip_fraction": (
            (valid_ratio - valid_ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)).abs() > 0
        ).float().mean().detach(),
        "state_weight_mean": state_weight.mean().detach(),
    }
    return loss, metrics


def soft_target_loss(logits, target_score, valid_mask, temperature=0.25, state_weight=None):
    """Dense ranking auxiliary; target is detached and no oracle is executed."""
    if temperature <= 0 or logits.shape != target_score.shape:
        raise ValueError("positive temperature and matching target shapes required")
    logp = masked_joint_log_softmax(logits, valid_mask)
    teacher_logp = masked_joint_log_softmax(target_score.detach() / temperature, valid_mask)
    safe_logp = torch.where(valid_mask, logp, torch.zeros_like(logp))
    teacher = torch.where(valid_mask, teacher_logp.exp(), torch.zeros_like(teacher_logp))
    per_state = -(teacher * safe_logp).sum(dim=(1, 2))
    weight = torch.ones_like(per_state) if state_weight is None else state_weight.to(per_state).reshape(-1)
    if weight.shape != per_state.shape or not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("state weights must be finite, positive, and shaped [B]")
    return (per_state * weight).sum() / weight.sum()


