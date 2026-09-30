"""Contracts for Step 11 selection and Step 12--13 state batching."""

import unittest

import numpy as np
import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_action import select_joint_pair
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_buffer import JointRolloutBuffer
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_datamodule import JointRIFTCollate


def state(group_1, group_2, pair_ids=(11, 22), stage=1, checkpoint="stage-1.ckpt"):
    logits = np.arange(group_1 * group_2, dtype=np.float32).reshape(group_1, group_2)
    return {
        "feature_1": {"feature": pair_ids[0]},
        "feature_2": {"feature": pair_ids[1]},
        "old_joint_logits": logits,
        "joint_advantage": logits / 10.0,
        "valid_mask_1": np.ones(group_1, dtype=bool),
        "valid_mask_2": np.ones(group_2, dtype=bool),
        "pair_ids": pair_ids,
        "frame_1": {
            "source_origin": np.array([0.0, 0.0]), "source_heading": 0.0,
            "ego_origin": np.array([0.0, 0.0]), "ego_heading": 0.0,
        },
        "frame_2": {
            "source_origin": np.array([1.0, 0.0]), "source_heading": 0.0,
            "ego_origin": np.array([0.0, 0.0]), "ego_heading": 0.0,
        },
        "collection_stage": stage,
        "collection_checkpoint": checkpoint,
        "diagnostics": {
            "p12": logits,
            "coalition_gain": logits,
            "pair_collision": np.zeros_like(logits, dtype=bool),
            "unary_return_1": np.zeros(group_1, dtype=np.float32),
            "unary_return_2": np.zeros(group_2, dtype=np.float32),
            "independent_same_as_joint_rate": 1.0,
            "oracle_gap_if_debug": 0.0,
        },
    }


class TestJointBufferAndDataModule(unittest.TestCase):
    def test_joint_action_ignores_invalid_high_logits(self):
        logits = torch.tensor([[1.0, 99.0], [2.0, 3.0]])
        valid_mask = torch.tensor([[True, False], [True, True]])
        self.assertEqual(select_joint_pair(logits, valid_mask), (1, 1))

    def test_buffer_appends_complete_states_without_waiting_for_done(self):
        buffer = JointRolloutBuffer(1, "train_cbv", {"buffer_capacity": 2})
        buffer.store({"joint_rollout_states": [state(1, 2)]})
        self.assertEqual(len(buffer), 1)
        self.assertFalse(buffer.buffer_full)
        self.assertEqual(buffer.sample(0)["pair_ids"], (11, 22))

    def test_buffer_rejects_stage_one_data_for_stage_two_training(self):
        buffer = JointRolloutBuffer(1, "train_cbv", {"buffer_capacity": 1})
        buffer.store({"joint_rollout_states": [state(1, 1, stage=1)]})
        with self.assertRaisesRegex(RuntimeError, "freshly collected Stage 2"):
            buffer.assert_on_policy_collection(2, "stage-1.ckpt")

    def test_buffer_rejects_mixed_checkpoint_collection(self):
        buffer = JointRolloutBuffer(1, "train_cbv", {"buffer_capacity": 2})
        buffer.store({"joint_rollout_states": [state(1, 1, checkpoint="first.ckpt")]})
        with self.assertRaisesRegex(RuntimeError, "cannot mix"):
            buffer.store({"joint_rollout_states": [state(1, 1, checkpoint="second.ckpt")]})

    def test_collate_pads_each_candidate_axis_and_rebuilds_outer_product_mask(self):
        first = state(1, 2)
        second = state(2, 1, pair_ids=(33, 44))
        second["valid_mask_1"] = np.array([True, False])
        collate = JointRIFTCollate(feature_collate=lambda features: features)

        output = collate([first, second])

        self.assertEqual(output["old_joint_logits"].shape, (2, 2, 2))
        self.assertEqual(output["joint_advantage"].shape, (2, 2, 2))
        self.assertTrue(output["joint_valid_mask"][0, 0, :2].all())
        self.assertFalse(output["joint_valid_mask"][0, 1].any())
        self.assertTrue(output["joint_valid_mask"][1, 0, 0])
        self.assertFalse(output["joint_valid_mask"][1, 1].any())
        self.assertTrue(torch.isneginf(output["old_joint_logits"][0, 1, 0]))
        self.assertEqual(output["pair_ids"], [(11, 22), (33, 44)])
        self.assertEqual(output["frame_1"]["source_origin"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
