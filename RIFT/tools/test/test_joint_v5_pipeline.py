"""CPU tests of real V5 math, exporter, and cached scorer; no CARLA stubs.

The exporter method is AST-loaded without its simulator class imports.
This is method-level verification, NOT an end-to-end CARLA/Lightning test.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT_ROOT = Path(os.environ.get(
    "JOINT_RIFT_TEST_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "joint_rift_validation")
)) / "v5"
MODULES = ROOT / "rift/cbv/planning/fine_tuner/rlft/joint_rift"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OFFLINE = load("v5_offline", ROOT / "scripts/joint_rift_offline.py")


def evaluation():
    return {"valid_mask": np.ones((3, 3), dtype=bool), "p12": np.array([[0., .1, .2], [.1, .2, 1.], [.2, .3, .1]]),
            "coalition_gain": np.zeros((3, 3)), "unary_return_1": np.zeros(3), "unary_return_2": np.zeros(3),
            "pair_collision": np.zeros((3, 3), dtype=bool),
            **{key: np.zeros(3, dtype=bool) for key in ("unary_collision_1", "unary_collision_2", "unary_offroad_1", "unary_offroad_2")},
            "p1": np.zeros(3), "p2": np.zeros(3), "interaction_trace_1": np.zeros((3, 40)), "interaction_trace_2": np.zeros((3, 40))}


def fixture(directory):
    """Exercise the actual production exporter on synthetic head inputs."""
    torch.manual_seed(17)
    cfg = dict(candidate_dim=4, representation_dim=8, pair_dim=4, geometry_hidden_dim=4,
               lambda_delta=4., trajectory_frames=40, trajectory_stride=5,
               base_temperature=2., residual_scale_floor=1., geometry_horizon=40)
    scorer = OFFLINE.Scorer(**cfg)
    checkpoint = directory / "base.ckpt"
    source = {"model.synthetic_backbone": torch.ones(1)}
    source.update({"joint_head." + k: v for k, v in scorer.state_dict().items()})
    torch.save({"state_dict": source}, checkpoint)
    candidates = []
    for axis in (1, 2):
        common = torch.randn(3, 80, 3)
        candidates.append(SimpleNamespace(candidate_features=torch.randn(3, 4),
            raw_trajectories=torch.randn(3, 80, 6), common_trajectories=common,
            global_trajectories=common, logits=torch.tensor([2., 0., -1.])))
    args = []
    for candidate in candidates:
        args.extend(v[None] for v in (candidate.candidate_features, candidate.common_trajectories, candidate.raw_trajectories, candidate.logits))
    output = scorer(*args)
    data = evaluation()
    data["joint_q"] = data["p12"].copy()
    batch = SimpleNamespace(candidates=candidates, pair_ids=(7, 9),
                            ego_nominal_common=np.zeros((40, 4)), ego_nominal_global=np.zeros((40, 4)))
    tree = ast.parse((MODULES / "joint_rift_audited.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuditedJointRIFTPluto")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_dump_candidate_state")
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_forensic_target_eligible")
    ns = dict(np=np, torch=torch, Path=Path, json=json, re=re, hashlib=hashlib,
              Dict=dict, Tuple=tuple, JointCandidateBatch=object)
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), "exporter", "exec"), ns)
    owner = SimpleNamespace(candidate_dump={"enabled": True, "max_states": 10, "stride_steps": 1,
        "max_states_per_episode": 3, "capture_all": True, "target_route_ids": set(),
        "target_time_windows": {}, "target_repetitions": {}, "include_training_inputs": True,
        "output_dir": str(directory / "cache")},
        _candidate_dump_total=0, _dump_episode_counts={}, _candidate_dump_target_counts={},
        _candidate_dump_non_high_value_count=0, joint_scorer=scorer, checkpoint=str(checkpoint),
        mode="eval", config={"joint": {"traj_sample_interval": 5}}, cooperation_config={},
        _checkpoint_hashes={}, _numpy_array=lambda v: v.detach().numpy() if torch.is_tensor(v) else np.asarray(v))
    info = {"route_id": "1", "repetition_index": 0, "scenario_index": 0, "env_id": 0, "time_step": 10}
    ns[method.name](owner, info, batch, output, data, {}, (0, 0), (0, 0), (1, 2), (0, 0))
    return checkpoint, scorer


class V5Test(unittest.TestCase):
    def test_temperature_alone_preserves_argmax_under_base_std_scaling(self):
        first = OFFLINE.Scorer(candidate_dim=4, representation_dim=8, pair_dim=4)
        second = OFFLINE.Scorer(candidate_dim=4, representation_dim=8, pair_dim=4, base_temperature=3.)
        fixed = OFFLINE.Scorer(candidate_dim=4, representation_dim=8, pair_dim=4,
                               base_temperature=3., residual_scale_mode="fixed", fixed_residual_scale=2.)
        with torch.no_grad():
            first.residual_projection.weight.fill_(.5)
        second.load_state_dict(first.state_dict())
        fixed.load_state_dict(first.state_dict())
        a = [torch.randn(1, 2, 4), torch.randn(1, 2, 80, 3), torch.randn(1, 2, 80, 6), torch.randn(1, 2)]
        b = [torch.randn(1, 3, 4), torch.randn(1, 3, 80, 3), torch.randn(1, 3, 80, 6), torch.randn(1, 3)]
        x, y, z = first(*a, *b), second(*a, *b), fixed(*a, *b)
        torch.testing.assert_close(y.delta, x.delta / 3.)
        self.assertEqual(x.joint_logits.argmax().item(), y.joint_logits.argmax().item())
        self.assertFalse(torch.allclose(z.delta, y.delta))

    def test_outcome_statistics_match_repetitions_and_cluster_routes(self):
        comparator = load("v5_comparator", ROOT / "scripts/compare_joint_v5_outcomes.py")
        def record(score):
            return {"scores": {k: score for k in ("score_composed", "score_route", "score_penalty")},
                    "status": "Completed", "meta": {"cbv_total_game_time": 10.,
                    "cbv_off_road_game_time": 0., "cbv_uncomfortable_game_time": 0., "cbv_collision_count": 0}}
        keys = [(0, "RouteScenario_1_rep0"), (0, "RouteScenario_1_rep1"), (1, "RouteScenario_2_rep0")]
        policy = {key: record(10.) for key in keys}
        independent = {key: record(9.) for key in keys}
        report = comparator.compare(policy, independent, bootstrap=100)
        self.assertEqual(report["matched_records"], 3)
        self.assertEqual(report["metrics"]["score_composed"]["routes"], 2)
        self.assertEqual(report["metrics"]["score_composed"]["route_cluster_bootstrap_95_percent"], [1., 1.])
        del independent[keys[0]]
        with self.assertRaises(ValueError):
            comparator.compare(policy, independent)

    def test_legacy_loss_matches_git_baseline_value_and_gradient(self):
        try:
            source = subprocess.check_output(["git", "show", "HEAD:RIFT/rift/cbv/planning/fine_tuner/rlft/joint_rift/joint_trainer.py"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("Git HEAD unavailable; historical loss regression skipped")
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"masked_joint_log_softmax", "joint_dual_clip_loss"}]
        from typing import Dict, Tuple
        ns = dict(torch=torch, Tensor=torch.Tensor, Dict=Dict, Tuple=Tuple)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "baseline", "exec"), ns)
        torch.manual_seed(3)
        old = torch.randn(2, 3, 4)
        advantage = torch.randn_like(old)
        mask = torch.ones_like(old, dtype=torch.bool)
        mask[0, 2] = False
        new = (old + .3 * torch.randn_like(old)).requires_grad_()
        baseline, _ = ns["joint_dual_clip_loss"](new, old, advantage, mask)
        current, _ = OFFLINE.losses.joint_dual_clip_loss(new, old, advantage, mask)
        torch.testing.assert_close(current, baseline)
        torch.testing.assert_close(torch.autograd.grad(current, new)[0], torch.autograd.grad(baseline, new)[0])

    def test_online_step_auxiliary_ignores_empty_feasibility_and_has_finite_gradient(self):
        tree = ast.parse((MODULES / "joint_trainer.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JointRIFTLightningTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_step")
        std = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_masked_pair_std")
        ns = dict(torch=torch, Tensor=torch.Tensor, Dict=dict,
                  joint_dual_clip_loss=OFFLINE.losses.joint_dual_clip_loss,
                  soft_target_loss=OFFLINE.losses.soft_target_loss)
        exec(compile(ast.Module(body=[method, std], type_ignores=[]), "online_step", "exec"), ns)
        mask = torch.tensor([[[True, True, False], [True, True, False]], [[True, True, True], [False, False, False]]])
        logits = torch.randn(2, 2, 3, requires_grad=True)
        result = SimpleNamespace(delta=logits, base_std=torch.ones(2), residual_scale=torch.ones(2))
        owner = SimpleNamespace(_forward_joint_logits=lambda _: (logits, result), clip_ratio=.2,
            dual_clip=3., pair_weighting="uniform", auxiliary_weight=.2, teacher_temperature=.25,
            log=lambda *args, **kwargs: None)
        batch = {"old_joint_logits": logits.detach(), "joint_advantage": torch.zeros_like(logits),
                 "joint_valid_mask": mask, "state_weight": torch.ones(2),
                 "cooperation_score": torch.randn_like(logits), "cooperation_feasible": mask.clone()}
        batch["cooperation_feasible"][1] = False
        loss = ns[method.name](owner, batch, "val")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))

    def test_route_split_never_splits_one_route_across_sets(self):
        states = [{"metadata": {"route_id": str(route)}} for route in range(5) for _ in range(4)]
        train, val, _ = OFFLINE.split_states(states)
        self.assertFalse({s["metadata"]["route_id"] for s in train} & {s["metadata"]["route_id"] for s in val})

    def test_forward_braid_sign_locality_and_noninteraction(self):
        braid = load("v5_braid", MODULES / "nominal_braid.py")
        ego = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        near_ahead = torch.tensor([[5., -1.], [6., -.5], [7., .5], [8., 1.]])
        near_behind = near_ahead.clone()
        near_behind[:, 0] -= 10
        far = near_ahead.clone()
        far[:, 0] += 100
        parallel = near_ahead.clone()
        parallel[:, 1] = 1
        result = braid.nominal_braid_events(torch.stack((near_ahead, near_behind, far, parallel)), ego, 0.)
        self.assertEqual(result.raw_event_mask.any(-1).tolist(), [True, True, False, False])
        self.assertGreater(result.signed_onset[0].sum().item(), 0.)
        self.assertLess(result.signed_onset[1].sum().item(), 0.)

    def test_production_diagnostic_method_handles_legacy_and_cooperation(self):
        tree = ast.parse((MODULES / "joint_rift_audited.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuditedJointRIFTPluto")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_feasibility_diagnostics")
        diagnostic = load("v5_diagnostic", MODULES / "joint_diagnostics.py")
        ns = dict(np=np, Dict=dict, Tuple=tuple, JointCandidateBatch=object, build_feasibility_diagnostics=diagnostic.build_feasibility_diagnostics,
                  cooperation_objective=OFFLINE.objective_fn)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "diagnostics", "exec"), ns)
        owner = SimpleNamespace(cooperation_config={}, joint_traj_evaluator=SimpleNamespace(both_necessary_epsilon=.1))
        batch = SimpleNamespace(candidates=(SimpleNamespace(logits=torch.zeros(3)), SimpleNamespace(logits=torch.zeros(3))))
        data = evaluation()
        data["joint_q"] = data["p12"].copy()
        self.assertNotIn("coop_gain", ns[method.name](owner, batch, data, (0, 0), (0, 0)))
        owner.cooperation_config = {"enabled": True}
        self.assertEqual(ns[method.name](owner, batch, data, (0, 0), (0, 0))["coop_opportunity"], 1.)

    def test_unary_is_a_budget_not_an_extra_ranking_reward(self):
        data = evaluation()
        data["unary_return_1"][2] = 1000
        result = OFFLINE.objective_fn(data, (0, 0))
        self.assertEqual(result["oracle"], (1, 2))
        self.assertGreater(result["target_score"][1, 2], result["target_score"][2, 2])

    def test_strict_two_member_support_rejects_strong_single_member(self):
        data = evaluation()
        data["coalition_gain"][2, 0] = .2
        result = OFFLINE.objective_fn(data, (0, 0), {"require_both": True})
        self.assertEqual(result["oracle"], (2, 0))
        self.assertFalse(result["feasible_mask"][1, 2])

    def test_each_vehicle_quality_and_hard_safety_are_checked(self):
        data = evaluation()
        data["unary_return_1"][1] = -1
        data["unary_return_2"][2] = 100
        data["unary_offroad_2"][1] = True
        result = OFFLINE.objective_fn(data, (0, 0))
        self.assertFalse(result["feasible_mask"][1, 2])
        self.assertFalse(result["feasible_mask"][:, 1].any())

    def test_missing_safety_flags_fail_and_empty_support_is_explicit(self):
        data = evaluation()
        del data["unary_collision_1"]
        with self.assertRaises(ValueError):
            OFFLINE.objective_fn(data, (0, 0))
        data = evaluation()
        data["pair_collision"][:] = True
        result = OFFLINE.objective_fn(data, (0, 0))
        self.assertIsNone(result["oracle"])
        np.testing.assert_array_equal(result["advantage"], 0)

    def test_probability_weighting_matches_exact_gradient_at_reference(self):
        old = torch.tensor([[[4., 0.], [-2., -3.]]])
        new = old.clone().requires_grad_()
        advantage = torch.tensor([[[1., 2.], [3., -2.]]])
        mask = torch.ones_like(old, dtype=torch.bool)
        loss, _ = OFFLINE.losses.joint_dual_clip_loss(new, old, advantage, mask, pair_weighting="old_policy")
        actual = torch.autograd.grad(loss, new)[0]
        exact = -(new.flatten(1).softmax(-1) * advantage.flatten(1)).sum()
        torch.testing.assert_close(actual, torch.autograd.grad(exact, new)[0])

    def test_floor_allows_learning_under_uniform_prior_and_geometry_obeys_horizon(self):
        scorer = OFFLINE.Scorer(candidate_dim=4, representation_dim=8, pair_dim=4,
                                residual_scale_floor=1., geometry_horizon=40)
        a = [torch.randn(1, 2, 4), torch.randn(1, 2, 80, 3), torch.randn(1, 2, 80, 6), torch.zeros(1, 2)]
        b = [torch.randn(1, 3, 4), torch.randn(1, 3, 80, 3), torch.randn(1, 3, 80, 6), torch.zeros(1, 3)]
        with torch.no_grad():
            scorer.residual_projection.weight.fill_(.3)
        first = scorer(*a, *b)
        self.assertGreater(first.delta.abs().max().item(), 0)
        a[1][:, :, 40:] += 1000
        torch.testing.assert_close(first.delta, scorer(*a, *b).delta)
        swapped = scorer(*b, *a)
        torch.testing.assert_close(first.delta, swapped.delta.transpose(1, 2))

    def test_real_export_verify_and_real_head_overfit(self):
        directory = TEST_OUTPUT_ROOT / uuid.uuid4().hex
        directory.mkdir(parents=True)
        checkpoint, scorer = fixture(directory)
        states = OFFLINE.load_cache(directory / "cache")
        self.assertLess(OFFLINE.verify(scorer, states, 5e-4), 5e-4)
        with self.assertRaises(ValueError):
            OFFLINE.split_states(states)
        initial = OFFLINE.metrics(scorer, states)["regret"]
        optimizer = torch.optim.Adam(scorer.parameters(), lr=.01)
        target = torch.as_tensor(states[0]["objective"]["target_score"], dtype=torch.float32)[None]
        for _ in range(150):
            optimizer.zero_grad()
            output = OFFLINE.forward(scorer, states[0])
            loss = OFFLINE.losses.soft_target_loss(output.joint_logits, target, output.valid_pair_mask, .1)
            loss.backward()
            optimizer.step()
        final = OFFLINE.metrics(scorer, states)["regret"]
        self.assertLess(final, initial)
        self.assertAlmostEqual(final, 0., places=6)
        (directory / "verification.json").write_text(json.dumps({"initial_regret": initial,
            "final_regret": final, "checkpoint": str(checkpoint), "scope": "synthetic inputs, actual scorer/exporter; no simulator"}, indent=2))
        print("V5 synthetic cache:", directory)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
