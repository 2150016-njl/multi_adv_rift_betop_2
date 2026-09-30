"""Unit tests for the dependency-free Joint-RIFT pair lifecycle."""

import unittest

from rift.cbv.planning.fine_tuner.rlft.joint_rift.pair_manager import PairManager


class PairManagerTest(unittest.TestCase):
    def test_first_pair_is_sorted_and_deterministic(self):
        manager = PairManager()

        self.assertEqual(manager.update(0, [17, 4, 9]), (4, 9))
        self.assertEqual(manager.get_pair(0), (4, 9))

    def test_pair_is_locked_while_its_members_stay_active(self):
        manager = PairManager()
        manager.update(0, [4, 9])

        self.assertEqual(manager.update(0, [1, 4, 9]), (4, 9))

    def test_missing_member_ends_then_reinitializes_segment(self):
        manager = PairManager()
        manager.update(0, [4, 9])

        self.assertEqual(manager.update(0, [4, 7]), (4, 7))
        self.assertEqual(manager.get_pair(0), (4, 7))

    def test_fewer_than_two_active_cbvs_clears_pair(self):
        manager = PairManager()
        manager.update(0, [4, 9])

        self.assertIsNone(manager.update(0, [4]))
        self.assertIsNone(manager.get_pair(0))


if __name__ == '__main__':
    unittest.main()

