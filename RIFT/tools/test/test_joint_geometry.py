"""Geometry invariants for the Joint-RIFT Ego-centric common frame."""

from types import SimpleNamespace
import unittest

import numpy as np

from rift.cbv.planning.fine_tuner.rlft.joint_rift.geometry import (
    carla_global_to_right_handed_trajectory,
    global_to_common_frame_trajectory,
    to_common_frame_trajectory,
)


def state(x, y, heading):
    return SimpleNamespace(
        rear_axle=SimpleNamespace(
            array=np.array([x, y], dtype=np.float32), heading=heading
        )
    )


class JointGeometryTest(unittest.TestCase):
    def test_carla_global_nominal_converts_to_rift_right_handed_frame(self):
        carla_trajectory = np.array([[4.0, 3.0, np.pi / 2, 7.0]], dtype=np.float32)

        converted = carla_global_to_right_handed_trajectory(carla_trajectory)

        np.testing.assert_allclose(converted, [[4.0, -3.0, -np.pi / 2, 7.0]])

    def test_source_and_ego_same_pose_preserves_local_trajectory(self):
        source = state(10.0, -3.0, np.pi / 2)
        trajectory = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

        common = to_common_frame_trajectory(trajectory, source, source)

        np.testing.assert_allclose(common, trajectory, atol=1e-6)

    def test_source_local_pose_is_rotated_into_ego_frame(self):
        source = state(10.0, 0.0, np.pi / 2)
        ego = state(0.0, 0.0, 0.0)
        trajectory = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        common = to_common_frame_trajectory(trajectory, source, ego)

        np.testing.assert_allclose(common, [[10.0, 1.0, np.pi / 2]], atol=1e-6)

    def test_global_nominal_plan_keeps_non_geometry_channels(self):
        ego = state(5.0, 2.0, np.pi / 2)
        nominal = np.array([[5.0, 3.0, np.pi / 2, 7.5]], dtype=np.float32)

        common = global_to_common_frame_trajectory(nominal, ego)

        np.testing.assert_allclose(common, [[1.0, 0.0, 0.0, 7.5]], atol=1e-6)


if __name__ == '__main__':
    unittest.main()

