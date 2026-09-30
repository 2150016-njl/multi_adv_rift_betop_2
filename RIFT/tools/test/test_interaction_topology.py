"""Unit contracts for Step 9 event-time segment crossing."""

import unittest

import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.interaction_topology import (
    discounted_interaction_return,
    get_arrival_order_label,
    get_interaction_events,
)


class TestInteractionTopology(unittest.TestCase):
    def setUp(self):
        # Ego's horizontal route has four segments at y=0.
        self.ego = torch.tensor(
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        )

    def test_consecutive_crossings_are_collapsed_to_one_onset(self):
        # The first two vertical candidate segments both cross x=-0.5, y=0.
        cbv = torch.tensor(
            [[[-0.5, -1.0], [-0.5, 1.0], [-0.5, -1.0], [3.0, -1.0], [3.0, -1.0]]]
        )
        events = get_interaction_events(cbv, self.ego)
        self.assertEqual(events.raw_event_mask.tolist(), [[True, True, False, False]])
        self.assertEqual(events.clustered_event_mask.tolist(), [[True, False, False, False]])

    def test_disjoint_crossings_are_both_retained_and_discounted(self):
        cbv = torch.tensor(
            [[[-0.5, -1.0], [-0.5, 1.0], [0.5, 1.0], [0.5, -1.0], [2.5, -1.0]]]
        )
        events = get_interaction_events(cbv, self.ego)
        self.assertEqual(events.clustered_event_mask.tolist(), [[True, False, True, False]])
        interaction_return = discounted_interaction_return(
            events.clustered_event_mask, gamma=0.98
        )
        self.assertTrue(torch.allclose(interaction_return, torch.tensor([1.0 + 0.98 ** 2])))

    def test_arrival_order_is_diagnostic_and_respects_valid_mask(self):
        cbv = torch.tensor(
            [[[-0.5, -1.0], [-0.5, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0]]]
        )
        labels = get_arrival_order_label(cbv, self.ego, arrival_tolerance_steps=0.1)
        self.assertEqual(labels[0, 0].item(), 1)
        masked_labels = get_arrival_order_label(
            cbv, self.ego, valid_mask=torch.tensor([False]), arrival_tolerance_steps=0.1
        )
        self.assertTrue(torch.equal(masked_labels, torch.zeros_like(masked_labels)))


if __name__ == '__main__':
    unittest.main()
