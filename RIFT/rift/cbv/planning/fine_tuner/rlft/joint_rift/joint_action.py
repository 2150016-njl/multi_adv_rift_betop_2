"""Pure, mask-safe joint action selection used by collection and evaluation."""

from typing import Tuple

import torch
from torch import Tensor


def select_joint_pair(joint_logits: Tensor, valid_mask: Tensor) -> Tuple[int, int]:
    """Return the argmax ``(i, j)`` over exactly the valid candidate pairs."""
    if joint_logits.ndim != 2 or valid_mask.shape != joint_logits.shape:
        raise ValueError("joint_logits and valid_mask must both have shape [G1, G2]")
    valid_mask = valid_mask.to(dtype=torch.bool, device=joint_logits.device)
    if not valid_mask.any():
        raise RuntimeError("cannot select an action without a valid candidate pair")
    masked_logits = joint_logits.masked_fill(~valid_mask, -torch.inf)
    flat_index = int(masked_logits.reshape(-1).argmax())
    group_2 = joint_logits.shape[1]
    return flat_index // group_2, flat_index % group_2
