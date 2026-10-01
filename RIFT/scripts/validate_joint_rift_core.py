#!/usr/bin/env python
"""Fast CPU-only checks for the audited Joint-RIFT math.

Run from the RIFT repository root:
    python scripts/validate_joint_rift_core.py
"""

import numpy as np
import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_diagnostics import (
    masked_additive_residual_ratio,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_score_head import (
    JointResidualScorer,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.temporal_interaction import (
    get_temporally_weighted_interaction_events,
)


def check_temporal_weighting():
    # Ego crosses x=0 near t=1; two CBV candidates follow the same vertical
    # conflict path, one crossing near the same time and one much later.
    horizon = 10
    ego = torch.stack(
        (torch.arange(horizon, dtype=torch.float32) - 1.0, torch.zeros(horizon)),
        dim=-1,
    )
    near_y = torch.arange(horizon, dtype=torch.float32) - 1.0
    late_y = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 2.0])
    cbv = torch.stack(
        (
            torch.stack((torch.zeros(horizon), near_y), dim=-1),
            torch.stack((torch.zeros(horizon), late_y), dim=-1),
        ),
        dim=0,
    )
    events = get_temporally_weighted_interaction_events(
        cbv, ego, dt=1.0, arrival_time_tau=1.0
    )
    near_weight = float(events.weighted_onset[0].max())
    late_weight = float(events.weighted_onset[1].max())
    assert near_weight > 0.9, (near_weight, late_weight)
    assert late_weight < near_weight * 0.1, (near_weight, late_weight)


def check_zero_init_factorization_and_symmetry():
    torch.manual_seed(7)
    scorer = JointResidualScorer(
        candidate_dim=8,
        representation_dim=16,
        pair_dim=8,
        trajectory_frames=40,
        trajectory_stride=5,
    )
    b, g1, g2, t = 1, 4, 5, 40
    f1 = torch.randn(b, g1, 8)
    f2 = torch.randn(b, g2, 8)
    c1 = torch.randn(b, g1, t, 3)
    c2 = torch.randn(b, g2, t, 3)
    r1 = torch.randn(b, g1, t, 6)
    r2 = torch.randn(b, g2, t, 6)
    z1 = torch.randn(b, g1)
    z2 = torch.randn(b, g2)

    out12 = scorer(f1, c1, r1, z1, f2, c2, r2, z2)
    out21 = scorer(f2, c2, r2, z2, f1, c1, r1, z1)
    assert torch.allclose(out12.delta, torch.zeros_like(out12.delta), atol=1e-7)
    assert torch.allclose(out12.joint_logits, out12.base_logits, atol=1e-7)
    assert torch.allclose(
        out12.joint_logits, out21.joint_logits.transpose(1, 2), atol=1e-6
    )
    flat = int(out12.joint_logits[0].argmax())
    joint_argmax = (flat // g2, flat % g2)
    independent_argmax = (int(z1[0].argmax()), int(z2[0].argmax()))
    assert joint_argmax == independent_argmax


def check_nonfactorizable_diagnostic():
    additive = np.array([[1.0, 2.0], [2.0, 3.0]])
    mask = np.ones_like(additive, dtype=bool)
    assert masked_additive_residual_ratio(additive, mask) < 1e-10
    xor_like = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert masked_additive_residual_ratio(xor_like, mask) > 0.9


def main():
    check_temporal_weighting()
    check_zero_init_factorization_and_symmetry()
    check_nonfactorizable_diagnostic()
    print("Joint-RIFT core validation: PASS")


if __name__ == "__main__":
    main()
