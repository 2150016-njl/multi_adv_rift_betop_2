"""Numerical contracts for Step 14's masked, state-balanced PPO objective."""

import math
import unittest

import torch
from torch import nn

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_trainer import (
    JointRIFTLightningTrainer,
    _local_raw_to_common,
    joint_dual_clip_loss,
    masked_joint_log_softmax,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_score_head import JointResidualScorer


class _TinyPlanningModel(nn.Module):
    """Enough of Pluto's module topology to test the Stage-2 freeze contract."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.planning_decoder = nn.Module()
        self.planning_decoder.loc_head = nn.Linear(2, 2)
        self.planning_decoder.pi_head = nn.Linear(2, 1)


class TestJointRIFTTrainer(unittest.TestCase):
    def test_loss_weights_states_equally_instead_of_weighting_large_grids_more(self):
        logits = torch.zeros((2, 2, 2), requires_grad=True)
        old_logits = torch.zeros((2, 2, 2))
        valid_mask = torch.tensor(
            [[[True, False], [False, False]], [[True, True], [True, True]]]
        )
        advantage = torch.tensor(
            [[[2.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]
        )

        loss, _ = joint_dual_clip_loss(logits, old_logits, advantage, valid_mask)

        # State 1 contributes 2; state 2 contributes mean(1, 1, 1, 1)=1.
        self.assertTrue(torch.allclose(loss, torch.tensor(-1.5)))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_padding_never_creates_nan_probability_ratios_or_gradients(self):
        new_logits = torch.tensor(
            [[[0.0, -torch.inf], [-torch.inf, -torch.inf]]], requires_grad=True
        )
        old_logits = torch.tensor([[[0.0, -torch.inf], [-torch.inf, -torch.inf]]])
        valid_mask = torch.tensor([[[True, False], [False, False]]])
        advantage = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

        log_probability = masked_joint_log_softmax(new_logits, valid_mask)
        loss, metrics = joint_dual_clip_loss(new_logits, old_logits, advantage, valid_mask)
        loss.backward()

        self.assertTrue(torch.isneginf(log_probability[0, 0, 1]))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(torch.isfinite(new_logits.grad).all())

    def test_local_raw_trajectory_reconstructs_the_collection_common_frame(self):
        raw = torch.tensor([[[[2.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]])
        frame = {
            "source_origin": torch.tensor([[10.0, 1.0]]),
            "source_heading": torch.tensor([math.pi / 2]),
            "ego_origin": torch.tensor([[10.0, 1.0]]),
            "ego_heading": torch.tensor([math.pi / 2]),
        }

        common = _local_raw_to_common(raw, frame)

        self.assertTrue(torch.allclose(common, torch.tensor([[[[2.0, 0.0, 0.0]]]]), atol=1e-6))

    def test_stage_two_unfreezes_only_the_single_shared_pi_head_with_its_own_lr(self):
        model = _TinyPlanningModel()
        joint_head = JointResidualScorer(candidate_dim=2, representation_dim=4, pair_dim=2)
        trainer = JointRIFTLightningTrainer(
            model=model,
            joint_head=joint_head,
            lr=1e-4,
            joint_lr=1e-4,
            pi_lr=1e-5,
            train_pi_head=True,
            cl_lr_decay=0.9,
            weight_decay=1e-5,
            epochs=2,
            warmup_epochs=1,
            frame_rate=10,
        )
        optimizer = trainer.configure_optimizers()[0][0]

        self.assertTrue(all(parameter.requires_grad for parameter in model.planning_decoder.pi_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.planning_decoder.loc_head.parameters()))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-5)
        self.assertFalse(any(key.startswith("pi_head.") for key in trainer.state_dict()))

    def test_scope_guard_rejects_any_attempt_to_unfreeze_encoder_or_trajectory_layers(self):
        with self.assertRaisesRegex(ValueError, "trainable_layers must remain empty"):
            JointRIFTLightningTrainer(
                model=_TinyPlanningModel(),
                joint_head=JointResidualScorer(candidate_dim=2, representation_dim=4, pair_dim=2),
                lr=1e-4,
                cl_lr_decay=0.9,
                weight_decay=1e-5,
                epochs=2,
                warmup_epochs=1,
                frame_rate=10,
                trainable_layers=["encoder"],
            )


if __name__ == "__main__":
    unittest.main()
