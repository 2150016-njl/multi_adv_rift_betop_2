"""V6 math, production scorer binding, and real offline export tests; no CARLA."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT_ROOT = Path(os.environ.get(
    "JOINT_RIFT_TEST_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "joint_rift_validation")
)) / "v6"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PIPE = load("v6_test_pipeline", ROOT / "scripts/joint_rift_v6.py")
BOX = PIPE.load_module("box_collision")
PAIR = load("v6_pair_proposal", ROOT / "rift/cbv/recognition/rule_based/interaction_pair_selection.py")


def evaluation():
    p1, p2 = np.array([.9, .45]), np.array([.1, .45])
    p12 = np.array([[.91, .945], [.55, .9]])
    return {"valid_mask": np.ones((2, 2), bool), "p1": p1, "p2": p2,
            "p12": p12, "coalition_gain": p12 - np.maximum(p1[:, None], p2[None]),
            "pair_collision": np.zeros((2, 2), bool), "unary_return_1": np.zeros(2), "unary_return_2": np.zeros(2),
            **{key: np.zeros(2, bool) for key in ("unary_collision_1", "unary_collision_2", "unary_offroad_1", "unary_offroad_2")}}


def inputs(dim=4):
    args = []
    for _ in (1, 2):
        args.extend((torch.randn(1, 2, dim), torch.randn(1, 2, 80, 3), torch.randn(1, 2, 80, 6), torch.tensor([[3., 0.]])))
    return args, torch.randn(1, 40, 4)


class V6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_sat_rejects_aabb_false_positive_and_keeps_real_overlap(self):
        rectangle = np.array([[-2., -.2], [2., -.2], [2., .2], [-2., .2]])
        angle = np.pi / 4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        first = rectangle @ rotation.T
        normal = np.array([-np.sin(angle), np.cos(angle)])
        separate, overlap = first + .8 * normal, first + .1 * normal
        self.assertTrue(np.all(first.max(0) >= separate.min(0)) and np.all(separate.max(0) >= first.min(0)))
        result = BOX.box_collision_trace(first[None, None], separate[None, None])
        self.assertFalse(result[0, 0])
        self.assertTrue(BOX.box_collision_trace(first[None, None], overlap[None, None])[0, 0])
        self.assertFalse(BOX.box_collision_trace(first[None, None], np.empty((0, 1, 4, 2))).any())

    def test_teacher_preserves_reference_on_ties_and_empty_support(self):
        d = evaluation()
        d["p1"][:] = 0; d["p2"][:] = 0; d["p12"][:] = 0; d["coalition_gain"][:] = 0
        prior = np.array([[-3., -2.], [-2., 0.]])
        t = PIPE.teacher_module.decision_teacher(d, (1, 1), prior)
        self.assertTrue(all(t[k] == (1, 1) for k in ("quality", "factorized", "interaction", "joint")))
        d["pair_collision"][:] = True
        t = PIPE.teacher_module.decision_teacher(d, (1, 1), prior)
        self.assertFalse(t["metrics"]["decision_has_feasible"])
        self.assertEqual(t["joint"], (1, 1))

    def test_teacher_distinguishes_joint_increment_from_unary_improvement(self):
        prior = np.array([[0., -3.], [-3., -6.]])
        t = PIPE.teacher_module.decision_teacher(evaluation(), (0, 0), prior)
        self.assertEqual(t["factorized"], (0, 1))
        self.assertEqual(t["joint"], (1, 1))
        self.assertTrue(t["metrics"]["decision_joint_support"])
        d = evaluation(); d["p1"][:] = 0; d["p12"] = np.broadcast_to(d["p2"][None], (2, 2)); d["coalition_gain"][:] = 0
        t = PIPE.teacher_module.decision_teacher(d, (0, 0), prior)
        self.assertTrue(t["metrics"]["decision_interaction_support"])
        self.assertFalse(t["metrics"]["decision_joint_support"])

    def test_nominal_conflict_relaxation_requires_split_labels(self):
        d = evaluation(); prior = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "recollect"):
            PIPE.teacher_module.decision_teacher(d, (0, 0), prior, config={"nominal_conflict_policy": "label_only"})
        d["unary_collision_1"][:] = True
        d["unary_collision_background_1"] = np.zeros(2, bool)
        d["unary_collision_background_2"] = np.zeros(2, bool)
        self.assertFalse(PIPE.teacher_module.decision_teacher(d, (0, 0), prior)["feasible_mask"].any())
        self.assertTrue(PIPE.teacher_module.decision_teacher(d, (0, 0), prior, config={"nominal_conflict_policy": "label_only"})["feasible_mask"].all())

    def test_verifier_rejects_bad_proposal_and_certifies_joint_increment(self):
        d = evaluation(); prior = np.array([[0., -3.], [-3., -6.]])
        t = PIPE.teacher_module.decision_teacher(d, (0, 0), prior)
        selected, metrics = PIPE.teacher_module.verified_decision(t, (0, 0), d, {"verification_target": "joint"})
        self.assertEqual(selected, (1, 1))
        self.assertFalse(metrics["decision_student_accepted"])
        selected, metrics = PIPE.teacher_module.verified_decision(t, (1, 1), d, {"verification_target": "joint"})
        self.assertEqual(selected, (1, 1))
        self.assertTrue(metrics["decision_student_accepted"])
        d["pair_collision"][:] = True
        t = PIPE.teacher_module.decision_teacher(d, (0, 0), prior)
        self.assertEqual(PIPE.teacher_module.verified_decision(t, (1, 1), d)[0], (0, 0))

    def test_production_legacy_collision_query_has_aabb_false_positive(self):
        try:
            from shapely.geometry import Polygon
            from shapely.strtree import STRtree
        except ImportError:
            self.skipTest("Shapely absent; pure SAT regression still runs")
        path = ROOT / "rift/cbv/planning/fine_tuner/rlft/traj_eval/traj_evaluator.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrajEvaluator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_collision_matrix")
        ns = dict(np=np, Polygon=Polygon, STRtree=STRtree)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "legacy_collision", "exec"), ns)
        box = np.array([[-2., -.2], [2., -.2], [2., .2], [-2., .2]])
        rotation = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
        first = box @ rotation.T
        second = first + np.array([-.8, .8]) / np.sqrt(2)
        self.assertTrue(ns[method.name](None, first[None, None], second[None, None])[0, 0])
        self.assertFalse(BOX.box_collision_trace(first[None, None], second[None, None])[0, 0])
        self.assertTrue(BOX.aabb_collision_trace(first[None, None], second[None, None])[0, 0])

    def test_sat_matches_polygon_intersection_across_frames_and_chunks(self):
        try:
            from shapely.geometry import Polygon
        except ImportError:
            self.skipTest("Shapely absent")
        rng = np.random.RandomState(4)
        def boxes(groups):
            result = np.empty((groups, 5, 4, 2))
            for g in range(groups):
                for t in range(5):
                    angle = rng.uniform(-np.pi, np.pi)
                    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    result[g, t] = np.array([[-2., -.8], [2., -.8], [2., .8], [-2., .8]]) @ rotation.T + rng.uniform(-5, 5, 2)
            return result
        a, b = boxes(4), boxes(3)
        expected = np.array([[any(Polygon(a[g, t]).intersects(Polygon(b[n, t])) for n in range(3)) for t in range(5)] for g in range(4)])
        np.testing.assert_array_equal(BOX.box_collision_trace(a, b, 1), expected)
        np.testing.assert_array_equal(BOX.box_collision_trace(a, b, 3), expected)

    def test_direct_head_swap_nominal_conditioning_and_initial_fallback(self):
        model = PIPE.head_module.NominalDirectScorer(candidate_dim=4)
        args, nominal = inputs()
        first = model(*args, ego_nominal=nominal)
        self.assertEqual(PIPE.head_module.select_direct(first), (0, 0))
        with torch.no_grad():
            model.action.weight.fill_(.1)
        first = model(*args, ego_nominal=nominal)
        swapped = model(*args[4:], *args[:4], ego_nominal=nominal)
        torch.testing.assert_close(first.joint_logits, swapped.joint_logits.transpose(1, 2))
        modified = model(*args, ego_nominal=nominal + 3.)
        self.assertFalse(torch.allclose(first.joint_logits, modified.joint_logits))

    def test_prediction_gate_rejects_high_logit_low_quality_action(self):
        model = PIPE.head_module.NominalDirectScorer(candidate_dim=4)
        args, nominal = inputs(); output = model(*args, ego_nominal=nominal)
        output.joint_logits = torch.tensor([[[0., 50.], [-1., -2.]]])
        output.quality_logits = torch.tensor([[[4., -4.], [-4., -4.]]])
        output.interaction = torch.tensor([[[0., 1.], [0., 0.]]])
        self.assertEqual(PIPE.head_module.select_direct(output), (0, 0))
        output.quality_logits[0, 0, 1] = 4.
        self.assertEqual(PIPE.head_module.select_direct(output), (0, 1))

    def test_shared_conflict_pair_keeps_active_member(self):
        records = [dict(id=1, ego_eta=2., cbv_eta=2., conflict_x=0., conflict_y=0.),
                   dict(id=2, ego_eta=2., cbv_eta=2., conflict_x=100., conflict_y=0.),
                   dict(id=3, ego_eta=2., cbv_eta=2., conflict_x=1., conflict_y=0.)]
        self.assertEqual(PAIR.select_conflict_pair(records), [1, 3])
        self.assertEqual(PAIR.select_conflict_pair(records, [2]), [3])
        self.assertEqual(PAIR.select_conflict_pair(records, [1, 2]), [])

    def test_production_recognition_uses_vector_speed_in_mps_for_both_etas(self):
        # Execute the production class without CARLA imports. These boundary
        # objects follow Vector3D's x/y/z contract; preflight also tests real CARLA.
        path = ROOT / "rift/cbv/recognition/rule_based/rule_cbv.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RuleCBVRecog")
        actors = {i: SimpleNamespace(id=i) for i in range(4)}
        vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        velocities = {0: vector(3., 4., 0.), 1: vector(0., 0., 5.),
                      2: None, 3: vector(0., 0., 0.)}
        seen = []
        def select(records, active):
            seen[:] = records
            return PAIR.select_conflict_pair(records, active)
        provider = SimpleNamespace(
            get_velocity=lambda actor: velocities[actor.id],
            get_location=lambda actor: SimpleNamespace(distance=lambda other: 10.),
            get_actor_by_id=lambda actor_id: actors[actor_id])
        ns = dict(BaseCBVRecog=object, CarlaDataProvider=provider,
                  select_conflict_pair=select, math=math)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "production_recognition", "exec"), ns)
        klass = ns["RuleCBVRecog"]
        owner = klass.__new__(klass)
        owner.mode = "eval"
        owner.config = {"eval": {"pair_selection": "shared_conflict"}}
        owner.max_agent_num = 2
        location = SimpleNamespace(x=10., y=0.)
        waypoint = SimpleNamespace(transform=SimpleNamespace(location=location))
        owner.route_planner = SimpleNamespace(inter_info={
            i: dict(inter_waypoint=waypoint, inter_CBV_dis=distance)
            for i, distance in ((1, 10.), (2, 8.), (3, 12.))})
        owner.get_CBV_candidates = lambda ego, active, rest: [actors[i] for i in (1, 2, 3) if i not in active]
        self.assertEqual([a.id for a in owner.get_CBVs(actors[0], [], [], [])], [1, 2])
        self.assertEqual([r["ego_eta"] for r in seen], [2., 2., 2.])
        self.assertEqual([r["cbv_eta"] for r in seen], [2., 4., 6.])
        self.assertEqual([a.id for a in owner.get_CBVs(actors[0], [1], [], [])], [2])
        self.assertEqual(owner.get_CBVs(actors[0], [1, 2], [], []), [])
        self.assertEqual(klass._velocity_speed_mps(vector(3., 4., 12.)), 13.)
        self.assertEqual(klass._velocity_speed_mps(None), 0.)
        owner.route_planner.inter_info = {}
        self.assertEqual([a.id for a in owner.get_CBVs(actors[0], [], [], [])], [1, 2])
        owner.config["eval"]["pair_selection"] = "nearest"
        self.assertEqual([a.id for a in owner.get_CBVs(actors[0], [], [], [])], [1, 2])

    def test_production_score_binding_passes_nominal_and_selects_direct(self):
        tree = ast.parse((ROOT / "rift/cbv/planning/fine_tuner/rlft/joint_rift/joint_rift_pluto.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JointRIFTPluto")
        functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"score_candidate_batch", "select_joint_pair"}]
        for f in functions:
            f.decorator_list = []
        ns = dict(torch=torch, select_direct=PIPE.head_module.select_direct)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "production_binding", "exec"), ns)
        args, nominal = inputs()
        candidates = [SimpleNamespace(candidate_features=args[i][0], common_trajectories=args[i+1][0], raw_trajectories=args[i+2][0], logits=args[i+3][0]) for i in (0, 4)]
        model = PIPE.head_module.NominalDirectScorer(candidate_dim=4)
        owner = SimpleNamespace(device="cpu", joint_scorer=model, direct_head=True)
        batch = SimpleNamespace(candidates=candidates, ego_nominal_common=nominal[0].numpy())
        output = ns["score_candidate_batch"](owner, batch)
        torch.testing.assert_close(output.joint_logits, model(*args, ego_nominal=nominal).joint_logits)
        self.assertEqual(ns["select_joint_pair"](output), (0, 0))

    def test_actual_fit_export_and_backbone_merge(self):
        directory = TEST_OUTPUT_ROOT / uuid.uuid4().hex
        directory.mkdir(parents=True)
        cache = directory / "cache"; cache.mkdir()
        base = directory / "base.ckpt"
        torch.save({"state_dict": {"model.synthetic_backbone": torch.tensor([17.])}}, base)
        checksum = hashlib.sha256(base.read_bytes()).hexdigest()
        for index in range(3):
            args, nominal = inputs()
            d = evaluation()
            d["valid_pair_mask"] = d.pop("valid_mask")
            d["base_logits"] = np.array([[0., -3.], [-3., -6.]])
            d["ego_nominal_common"] = nominal[0].numpy()
            for axis, offset in ((1, 0), (2, 4)):
                for key, value in zip(("candidate_features_" + str(axis), "cbv_" + str(axis) + "_common_trajectories", "raw_trajectories_" + str(axis), "unary_logits_" + str(axis)), args[offset:offset+4]):
                    d[key] = value[0].numpy()
            stem = cache / ("state_" + str(index))
            np.savez_compressed(str(stem) + ".npz", **d)
            Path(str(stem) + ".json").write_text(json.dumps({"schema_version": 2, "checkpoint_sha256": checksum,
                "cooperation_config": {}, "reward_config": {"interaction_semantics": "spatial_temporal"},
                "route_id": str(index), "independent_indices": [0, 0]}), encoding="utf-8")
        output = directory / "fit"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/joint_rift_v6.py"), "fit", "--cache", str(cache),
            "--output_dir", str(output), "--head_only", "--teacher", "joint", "--overfit_opportunities", "--steps", "100", "--batch_size", "3"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-2000:])
        checkpoint = output / "joint_head.ckpt"
        saved = torch.load(checkpoint, weights_only=False)
        self.assertTrue(saved["head_only"])
        prepared = directory / "prepared"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_joint_rift_v6.py"), "--base_ckpt", str(base),
            "--student_ckpt", str(checkpoint), "--run_dir", str(prepared)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        merged = torch.load(prepared / "student_merged.ckpt", weights_only=False)
        torch.testing.assert_close(merged["state_dict"]["model.synthetic_backbone"], torch.tensor([17.]))
        self.assertFalse(merged["head_only"])
        print("V6 synthetic pipeline:", directory)

    def test_summary_excludes_partial_runs_and_pairs_complete_outcomes(self):
        summary = load("v6_summary_test", ROOT / "scripts/summarize_joint_v6.py")
        directory = TEST_OUTPUT_ROOT / uuid.uuid4().hex
        for mode, progress in (("quality_oracle", [2, 2]), ("interaction_oracle", [2, 2]), ("policy", [1, 2])):
            target = directory / (mode + "_seed0") / "eval" / "agent"
            target.mkdir(parents=True)
            records = [{"route_id": "RouteScenario_" + str(i) + "_rep0", "status": "Completed",
                "scores": {"score_composed": 100., "score_route": 100., "score_penalty": 1.},
                "infractions": {"collisions_vehicle": [], "collisions_pedestrian": [], "collisions_layout": []},
                "meta": {"cbv_total_game_time": 10., "cbv_collision_count": 0, "cbv_off_road_game_time": 0., "cbv_uncomfortable_game_time": 0.}}
                for i in range(progress[0])]
            (target / "simulation_results.json").write_text(json.dumps({"_checkpoint": {"progress": progress, "records": records}}), encoding="utf-8")
        result = summary.summarize(directory)
        self.assertEqual(len(result["comparisons"]), 1)
        compared = result["comparisons"]["interaction_oracle_vs_quality_oracle"]
        self.assertEqual(compared["matched_records"], 2)
        self.assertIn("ego_vehicle_collision_count", compared["metrics"])
        self.assertEqual(sum(r["complete"] for r in result["runs"].values()), 2)

    def test_aggregation_allows_head_updates_but_rejects_changed_teacher_geometry(self):
        def state(head):
            d = evaluation(); d["valid_pair_mask"] = d["valid_mask"]
            d["base_logits"] = np.array([[0., -3.], [-3., -6.]])
            return {"data": d, "metadata": {"backbone_sha256": "frozen", "checkpoint_sha256": head,
                "independent_indices": [0, 0], "reward_config": {"collision_geometry": "exact_sat"}}}
        states = [state("head0"), state("head1")]
        self.assertEqual(PIPE.prepare(states, {}), "frozen")
        states[1]["metadata"]["reward_config"]["collision_geometry"] = "legacy_aabb"
        with self.assertRaisesRegex(ValueError, "aggregation"):
            PIPE.prepare(states, {})

    def test_pid_semantic_replay_verifies_original_labels(self):
        centers = np.array([[[0., -2.], [1., -1.], [2., 1.], [3., 2.]]], dtype=np.float32)
        ego = np.array([[0., 0., 0., 1.], [1., 0., 0., 1.], [2., 0., 0., 1.], [3., 0., 0., 1.]], dtype=np.float32)
        s = {"data": {"rollout_center_1": centers, "rollout_center_2": centers + np.array([0., 1.], dtype=np.float32),
            "raw_trajectories_1": np.zeros((1, 4, 6), np.float32), "ego_nominal_aligned_global": ego,
            "p1": np.zeros(1), "p2": np.zeros(1), "p12": np.zeros((1, 1)), "coalition_gain": np.zeros((1, 1))},
            "metadata": {"reward_config": {"interaction_semantics": "spatial_temporal"}, "evaluation_config": {"gamma": .98, "dt": .1}}}
        PIPE.relabel_semantics([s], "forward_braid")
        self.assertLess(PIPE.relabel_semantics([s], "forward_braid"), 1e-8)
        s["data"]["p1"] += 1.
        with self.assertRaisesRegex(ValueError, "replay failed"):
            PIPE.relabel_semantics([s], "forward_braid")


if __name__ == "__main__":
    unittest.main()
