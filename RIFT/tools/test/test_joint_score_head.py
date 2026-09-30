"""Contract tests for the Step 8 Joint-RIFT residual scorer."""

import unittest

import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_score_head import (
    JointResidualScorer,
)


class TestJointResidualScorer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.scorer = JointResidualScorer(
            candidate_dim=4,
            representation_dim=8,
            pair_dim=4,
            geometry_hidden_dim=4,
        )

    @staticmethod
    def _group(group_size):
        return (
            torch.randn(1, group_size, 4),
            torch.randn(1, group_size, 40, 3),
            torch.randn(1, group_size, 40, 6),
            torch.randn(1, group_size),
        )

    def test_zero_initial_residual_preserves_factorized_log_probabilities(self):
        group_1 = self._group(2)
        group_2 = self._group(3)
        output = self.scorer(*group_1, *group_2)

        expected = (
            torch.log_softmax(group_1[3], dim=-1).unsqueeze(-1)
            + torch.log_softmax(group_2[3], dim=-1).unsqueeze(-2)
        )
        self.assertTrue(torch.equal(output.raw_delta, torch.zeros_like(output.raw_delta)))
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(output.delta)))
        self.assertTrue(torch.allclose(output.joint_logits, expected))
        self.assertTrue(torch.equal(output.residual_scale, torch.zeros_like(output.residual_scale)))

    def test_shared_pair_scorer_is_symmetric_under_member_swap(self):
        group_1 = self._group(2)
        group_2 = self._group(3)
        with torch.no_grad():
            self.scorer.residual_projection.weight.copy_(torch.tensor([[0.75, -0.25]]))
            self.scorer.residual_projection.bias.fill_(0.1)

        forward = self.scorer(*group_1, *group_2)
        swapped = self.scorer(*group_2, *group_1)
        self.assertTrue(torch.allclose(forward.raw_delta, swapped.raw_delta.transpose(-2, -1)))
        self.assertTrue(torch.allclose(forward.joint_logits, swapped.joint_logits.transpose(-2, -1)))

    def test_invalid_candidates_cannot_form_joint_actions(self):
        group_1 = self._group(2)
        group_2 = self._group(2)
        output = self.scorer(
            *group_1,
            *group_2,
            valid_mask_1=torch.tensor([[True, False]]),
            valid_mask_2=torch.tensor([[False, True]]),
        )
        self.assertTrue(output.valid_pair_mask[0, 0, 1])
        self.assertFalse(output.valid_pair_mask[0, 0, 0])
        self.assertTrue(torch.isneginf(output.joint_logits[0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
