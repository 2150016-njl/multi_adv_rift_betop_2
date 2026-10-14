"""Probe the actual scorer/loss source with Torch, without simulator imports.

Only the two top-level loss functions are AST-loaded from joint_losses.py;
the Lightning trainer and CARLA integration are explicitly NOT tested here.
This measures current behavior; it does not substitute a new training loss.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, Tuple
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "rift/cbv/planning/fine_tuner/rlft/joint_rift"


def load_loss_functions():
    path = MODULES / "joint_losses.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"masked_joint_log_softmax", "joint_dual_clip_loss"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in functions} != wanted:
        raise ValueError("loss functions changed; inspect source before probing")
    namespace = {"torch": torch, "Tensor": torch.Tensor, "Tuple": Tuple, "Dict": Dict}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["joint_dual_clip_loss"]


def load_scorer():
    name = "joint_rift_scorer_probe"
    spec = importlib.util.spec_from_file_location(name, MODULES / "joint_score_head.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.JointResidualScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--run_source_tests", action="store_true", help="Run original scorer/geometry/topology/pair tests via isolated source imports")
    args = parser.parse_args()
    torch.manual_seed(7)
    loss_function = load_loss_functions()
    # Full enumeration: the mathematically exact policy expectation has
    # gradient -p*(Q-EpQ), whereas uniform ratio averaging generally does not.
    old = torch.tensor([[[4., 0.], [-2., -3.]]], dtype=torch.float64)
    reward = torch.tensor([[[1., 2.], [3., -2.]]], dtype=torch.float64)
    advantage = (reward - reward.mean()) / reward.std(unbiased=False)
    mask = torch.ones_like(old, dtype=torch.bool)
    new = old.clone().requires_grad_()
    loss, _ = loss_function(new, old, advantage, mask)
    uniform_gradient = torch.autograd.grad(loss, new)[0]
    exact_loss = -(new.flatten(1).softmax(1) * advantage.flatten(1)).sum(1).mean()
    exact_gradient = torch.autograd.grad(exact_loss, new)[0]
    oldp = old.flatten(1).softmax(1).view_as(old)
    ratio = (new.flatten(1).log_softmax(1) - old.flatten(1).log_softmax(1)).exp().view_as(old)
    weighted_loss = -(oldp * ratio * advantage).sum()
    weighted_gradient = torch.autograd.grad(weighted_loss, new)[0]
    cosine = torch.nn.functional.cosine_similarity(uniform_gradient.flatten(), exact_gradient.flatten(), dim=0)

    scorer = load_scorer()(candidate_dim=4, representation_dim=8, pair_dim=4, geometry_hidden_dim=4)
    with torch.no_grad():
        scorer.residual_projection.weight.copy_(torch.tensor([[.75, -.25]]))
        scorer.residual_projection.bias.fill_(.1)
    def group(size):
        return (torch.randn(1, size, 4), torch.randn(1, size, 80, 3), torch.randn(1, size, 80, 6), torch.randn(1, size))
    first, second = group(2), group(3)
    baseline = scorer(*first, *second)
    swapped = scorer(*second, *first)
    # Future outside the declared 40-frame representation/target window.
    changed = list(first)
    changed[1] = first[1].clone()
    changed[1][:, :, 40:, :2] += 1000
    # Force the original minimum to occur after frame 40 for every pair.
    first[1][:, :, :40, :2] += 100
    second[1][:, :, :40, :2] -= 100
    first[1][:, :, 60, :2] = 0
    second[1][:, :, 60, :2] = 0
    changed[1][:, :, :40] = first[1][:, :, :40]
    horizon_before = scorer(*first, *second)
    horizon_after = scorer(*changed, *second)
    uniform_first, uniform_second = list(first), list(second)
    uniform_first[3] = torch.zeros_like(first[3])
    uniform_second[3] = torch.zeros_like(second[3])
    uniform_output = scorer(*uniform_first, *uniform_second)
    report = {
        "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256((MODULES / name).read_bytes()).hexdigest() for name in ["joint_losses.py", "joint_trainer.py", "joint_score_head.py"]},
        "uniform_ratio_gradient": uniform_gradient.flatten().tolist(),
        "exact_expectation_gradient": exact_gradient.flatten().tolist(),
        "gradient_cosine": float(cosine),
        "old_probability_weighted_ratio_matches_exact_gradient": bool(torch.allclose(weighted_gradient, exact_gradient, atol=1e-12)),
        "swap_symmetry_max_error": float((baseline.joint_logits - swapped.joint_logits.transpose(1, 2)).abs().max().detach()),
        "delta_change_from_frames_after_40": float((horizon_before.delta - horizon_after.delta).abs().max().detach()),
        "uniform_prior_base_std": float(uniform_output.base_std[0]),
        "uniform_prior_delta_max": float(uniform_output.delta.abs().max().detach()),
        "limitations": "Synthetic diagnostic of actual source math; no Lightning, simulator, checkpoint or generalization validation. Probability-weighted ratio equality is checked before clipping at new=old only.",
    }
    assert report["old_probability_weighted_ratio_matches_exact_gradient"]
    assert report["swap_symmetry_max_error"] < 1e-5
    if args.run_source_tests:
        prefix = "rift.cbv.planning.fine_tuner.rlft.joint_rift."
        for name in ("joint_score_head", "geometry", "interaction_topology", "pair_manager"):
            spec = importlib.util.spec_from_file_location(prefix + name, MODULES / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[prefix + name] = module
            spec.loader.exec_module(module)
        suite = unittest.TestSuite()
        for name in ("test_joint_score_head", "test_joint_geometry", "test_interaction_topology", "test_pair_manager"):
            spec = importlib.util.spec_from_file_location(name, ROOT / "tools/test" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        result = unittest.TextTestRunner().run(suite)
        report["source_tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}
        if not result.wasSuccessful():
            raise AssertionError("original source tests failed")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
