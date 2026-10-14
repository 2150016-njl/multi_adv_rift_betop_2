"""Independent numeric contracts for offline evidence analysis."""

import importlib.util
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np


PATH = Path(__file__).resolve().parents[2] / "scripts/audit_joint_rift_evidence.py"
SPEC = importlib.util.spec_from_file_location("evidence_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class EvidenceAuditTest(unittest.TestCase):
    def test_cartesian_additive_target_has_no_interaction(self):
        q = np.arange(3)[:, None] + 2 * np.arange(4)[None, :]
        residual, ratio = AUDIT.additive_residual(q, np.ones_like(q, dtype=bool))
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertAlmostEqual(ratio, 0)

    def test_additive_target_stays_additive_with_safety_holes(self):
        q = np.arange(3)[:, None] + 2 * np.arange(4)[None, :]
        mask = np.ones_like(q, dtype=bool)
        mask[0, 0] = False
        mask[1, 2] = False
        residual, ratio = AUDIT.additive_residual(q, mask)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertLess(ratio, 1e-20)

    def test_xor_signal_is_pure_interaction(self):
        q = np.array([[0., 1.], [1., 0.]])
        self.assertAlmostEqual(AUDIT.additive_residual(q, np.ones((2, 2), dtype=bool))[1], 1)

    def test_safe_oracle_cannot_choose_masked_high_reward(self):
        q = np.array([[1., 1000.], [2., 3.]])
        mask = np.array([[True, False], [True, True]])
        self.assertEqual(AUDIT.argmax_pair(q, mask), (1, 1))

    def test_no_safe_pairs_and_invalid_numeric_values_are_explicit(self):
        with self.assertRaises(ValueError):
            AUDIT.argmax_pair(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            AUDIT.additive_residual(np.array([[np.nan]]), np.ones((1, 1), dtype=bool))

    def test_repetitions_remain_separate_in_matched_results(self):
        def record(rep, score):
            return {"route_id": "RouteScenario_1_rep" + str(rep), "status": "Completed",
                    "scores": {k: score for k in ("score_composed", "score_route", "score_penalty")},
                    "meta": {k: 0 for k in ("cbv_collision_count", "cbv_off_road_game_time", "cbv_uncomfortable_game_time", "cbv_total_game_time", "duration_game")}}
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for action, scores in (("policy", (10, 20)), ("independent", (9, 18))):
                payload = {"_checkpoint": {"records": [record(i, score) for i, score in enumerate(scores)], "global_record": {"scores_mean": {"score_composed": np.mean(scores)}}}}
                archive.writestr("evidence/outputs/holdout10_uniform_" + action + "_seed0/eval/simulation_results.json", json.dumps(payload))
        data.seek(0)
        _, pairs = AUDIT.formal_result_audit(data)
        self.assertEqual(len(pairs), 2)
        self.assertEqual([r["score_composed_delta"] for r in pairs], [1, 2])

    def test_unmatched_result_routes_fail_instead_of_partial_comparison(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            payload = {"_checkpoint": {"records": [], "global_record": {"scores_mean": {}}}}
            archive.writestr("evidence/outputs/holdout10_uniform_policy_seed0/eval/simulation_results.json", json.dumps(payload))
        data.seek(0)
        with self.assertRaises(ValueError):
            AUDIT.formal_result_audit(data)

    def test_same_unary_and_full_q_optimum_does_not_rule_out_pressure_opportunity(self):
        base = np.array([[0., -1.], [-1., -2.]])
        h = np.array([[0., 1.], [1., 0.]])
        cg = np.array([[0., .25], [.25, 0.]])
        unary = np.array([0., 100.])
        arrays = {
            "joint_q": unary[:, None] + unary[None, :] + h,
            "valid_pair_mask": np.ones((2, 2), dtype=bool),
            "pair_collision": np.zeros((2, 2), dtype=bool),
            "base_logits": base, "joint_logits": base,
            "joint_residual": np.zeros((2, 2)),
            "unary_return_1": unary, "unary_return_2": unary,
            "p12": h - cg, "coalition_gain": cg,
        }
        result, _ = AUDIT.audit_state(arrays, {"route_id": "synthetic", "repetition_index": 0, "time_step": 0}, [1])
        self.assertEqual(result["safe_oracle"], result["safe_unary_oracle"])
        self.assertEqual(result["safe_pressure_gain_no_unary_sum_loss"], 1)
        self.assertEqual(result["safe_nominal_both_no_unary_sum_loss_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
