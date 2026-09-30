"""Pure aggregation contracts for the Step 10 joint trajectory evaluator."""

import unittest

import numpy as np

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_traj_evaluator import (
    JointTrajEvaluator,
    UnaryEvaluation,
)


def _boxes(centers):
    """Axis-aligned [G,T,4,2] test boxes with width/length 2."""
    centers = np.asarray(centers, dtype=np.float64)
    offsets = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
    return centers[:, :, None, :] + offsets[None, None]


class TestJointTrajEvaluator(unittest.TestCase):
    def _evaluator_without_carla_rollout(self):
        evaluator = JointTrajEvaluator.__new__(JointTrajEvaluator)
        evaluator.gamma = 0.98
        evaluator.lambda_pressure = 1.0
        evaluator.lambda_coalition = 1.0
        evaluator.lambda_pair_collision = 3.0
        evaluator.pair_collision_chunk_size = 1
        return evaluator

    def test_pair_collision_uses_candidate_pairs_not_any_other_candidate(self):
        vertices_1 = _boxes([[[0, 0], [0, 0]], [[10, 0], [10, 0]]])
        vertices_2 = _boxes([[[0.5, 0], [0.5, 0]], [[20, 0], [20, 0]]])
        collision = JointTrajEvaluator.pair_collision_from_vertices(vertices_1, vertices_2, chunk_size=1)
        self.assertEqual(collision.tolist(), [[True, False], [False, False]])

    def test_joint_q_uses_union_pressure_coalition_and_masked_normalisation(self):
        evaluator = self._evaluator_without_carla_rollout()
        vertices_1 = _boxes([[[0, 0], [0, 0], [0, 0]], [[10, 0], [10, 0], [10, 0]]])
        vertices_2 = _boxes([[[20, 0], [20, 0], [20, 0]], [[30, 0], [30, 0], [30, 0]]])
        unary_1 = UnaryEvaluation(
            rollout_return=np.array([1.0, 2.0]),
            interaction_trace=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            interaction_pressure=np.array([1.0, 0.98 ** 2]),
            rollout_vertices=vertices_1,
            collision_matrix=np.zeros((2, 3), dtype=bool),
            off_road_matrix=np.zeros((2, 3), dtype=bool),
        )
        unary_2 = UnaryEvaluation(
            rollout_return=np.array([3.0, 4.0]),
            interaction_trace=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            interaction_pressure=np.array([0.98, 0.98 ** 2]),
            rollout_vertices=vertices_2,
            collision_matrix=np.zeros((2, 3), dtype=bool),
            off_road_matrix=np.zeros((2, 3), dtype=bool),
        )
        result = evaluator.combine_unary_evaluations(
            unary_1, unary_2, valid_mask_1=np.array([True, False])
        )
        self.assertTrue(np.allclose(result['p12'][0], np.array([1.0 + 0.98, 1.0 + 0.98 ** 2])))
        self.assertTrue(np.allclose(result['coalition_gain'][0], np.array([0.98, 0.98 ** 2])))
        self.assertTrue(result['valid_mask'][0].all())
        self.assertFalse(result['valid_mask'][1].any())
        self.assertTrue(np.all(result['joint_advantage'][1] == 0.0))


if __name__ == '__main__':
    unittest.main()
