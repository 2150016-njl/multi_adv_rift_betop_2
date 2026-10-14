"""Verify, describe, and fit the real frozen-generator scorer from schema-v2 caches.

No CARLA/Lightning import. NPZ contains numeric arrays only. Checkpoints are
trusted local Torch artifacts. Offline fitting is a diagnostic, not on-policy RL.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

MODULES = Path(__file__).resolve().parents[1] / "rift/cbv/planning/fine_tuner/rlft/joint_rift"


def module(name):
    key = "offline_" + name
    spec = importlib.util.spec_from_file_location(key, MODULES / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[key] = value
    spec.loader.exec_module(value)
    return value


Scorer = module("joint_score_head").JointResidualScorer
objective_fn = module("cooperation_objective").cooperation_objective
losses = module("joint_losses")


def load_cache(directory, limit=0):
    states = []
    paths = sorted(directory.rglob("state_*.npz"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("schema_version", 0) < 2:
            raise ValueError(str(path) + ": schema-v1 lacks latent/raw inputs; recollect include_training_inputs=true")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        required = ("candidate_features_1", "candidate_features_2", "raw_trajectories_1", "raw_trajectories_2")
        if any(key not in data for key in required):
            raise ValueError("incomplete head input cache: " + str(path))
        data["valid_mask"] = data["valid_pair_mask"]
        independent = tuple(metadata["independent_indices"])
        states.append({"path": str(path), "metadata": metadata, "data": data,
                       "objective": objective_fn(data, independent, metadata.get("cooperation_config"))})
    if not states:
        raise ValueError("no state_*.npz caches found")
    signatures = {(s["metadata"]["checkpoint_sha256"],
                   json.dumps({k: s["metadata"].get(k, {}) for k in ("scorer_config", "cooperation_config", "reward_config")}, sort_keys=True)) for s in states}
    if len(signatures) != 1:
        raise ValueError("fixed-data fit requires one collection checkpoint and scorer config; select a snapshot-specific cache")
    return states


def forward(scorer, state):
    data = state["data"]
    device = next(scorer.parameters()).device
    args = []
    for axis in (1, 2):
        keys = ("candidate_features_" + str(axis), "cbv_" + str(axis) + "_common_trajectories",
                "raw_trajectories_" + str(axis), "unary_logits_" + str(axis))
        args.extend(torch.as_tensor(data[key], dtype=torch.float32, device=device).unsqueeze(0) for key in keys)
    masks = [torch.as_tensor(data["valid_mask"].any(axis=1 if axis == 1 else 0), device=device).unsqueeze(0) for axis in (1, 2)]
    return scorer(*args, valid_mask_1=masks[0], valid_mask_2=masks[1])


def verify(scorer, states, tolerance):
    maximum = 0.
    with torch.no_grad():
        for state in states:
            output = forward(scorer, state)
            data = state["data"]
            for attribute, key in (("joint_logits", "joint_logits"), ("base_logits", "base_logits"), ("delta", "joint_residual")):
                actual = getattr(output, attribute)[0].cpu().numpy()
                error = float(np.max(abs(actual[data["valid_mask"]] - data[key][data["valid_mask"]])))
                maximum = max(maximum, error)
    if maximum > tolerance:
        raise ValueError("cached/recomputed scores differ: " + str(maximum) + "; check checkpoint, candidate identity, frame and scorer config")
    return maximum


def metrics(scorer, states):
    rows = []
    with torch.no_grad():
        for state in states:
            output = forward(scorer, state)
            chosen = tuple(int(x) for x in np.unravel_index(output.joint_logits[0].argmax().item(), output.joint_logits.shape[1:]))
            target = state["objective"]
            valid = state["data"]["valid_mask"]
            baseline = tuple(state["metadata"]["independent_indices"])
            rows.append({"route": str(state["metadata"]["route_id"]),
                         "regret": float(target["target_score"][valid].max() - target["target_score"][chosen]),
                         "feasible": float(target["feasible_mask"][chosen]),
                         "h_gain": float(target["interaction"][chosen] - target["interaction"][baseline]),
                         "opportunity": target["metrics"]["coop_opportunity"]})
    report = {key: float(np.mean([r[key] for r in rows])) for key in ("regret", "feasible", "h_gain", "opportunity")}
    opportunities = [r for r in rows if r["opportunity"] > .5]
    report["opportunity_regret"] = float(np.mean([r["regret"] for r in opportunities])) if opportunities else None
    report["opportunity_improvement_rate"] = float(np.mean([r["feasible"] > .5 and r["h_gain"] > 1e-6 for r in opportunities])) if opportunities else None
    return report


def split_states(states, seed=0, overfit=False):
    groups = sorted({str(s["metadata"]["route_id"]) for s in states})
    if overfit:
        return states, states, {"mode": "overfit_train_equals_validation", "routes": groups}
    if len(groups) < 2:
        raise ValueError("route-held-out fit needs >=2 routes; --overfit is a labelled debugging mode")
    random.Random(seed).shuffle(groups)
    cut = min(max(int(len(groups) * .8), 1), len(groups) - 1)
    train_groups = set(groups[:cut])
    return ([s for s in states if str(s["metadata"]["route_id"]) in train_groups],
            [s for s in states if str(s["metadata"]["route_id"]) not in train_groups],
            {"mode": "route", "train_routes": groups[:cut], "validation_routes": groups[cut:]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stats", "verify", "fit"))
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--backbone_ckpt", type=Path)
    parser.add_argument("--max_states", type=int, default=0)
    parser.add_argument("--objective", choices=("legacy", "cooperation"), default="cooperation")
    parser.add_argument("--loss", choices=("rift", "old_policy", "distill", "rift_aux"), default="rift")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--require_both", action="store_true")
    parser.add_argument("--require_comfortable", action="store_true")
    parser.add_argument("--unary_budget", type=float)
    parser.add_argument("--temperature", type=float, default=.25)
    parser.add_argument("--auxiliary_weight", type=float, default=.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--residual_scale_mode", choices=("base_std", "fixed"))
    parser.add_argument("--fixed_residual_scale", type=float)
    for name, dtype in (("base_temperature", float), ("lambda_delta", float), ("residual_scale_floor", float), ("geometry_horizon", int)):
        parser.add_argument("--" + name, type=dtype)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("steps, batch_size and lr must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    states = load_cache(args.cache_dir, args.max_states)
    target_config = dict(states[0]["metadata"].get("cooperation_config", {}))
    if args.require_both:
        target_config["require_both"] = True
    if args.require_comfortable:
        target_config["require_comfortable"] = True
    if args.unary_budget is not None:
        target_config["unary_budget"] = args.unary_budget
    for state in states:
        state["objective"] = objective_fn(state["data"], tuple(state["metadata"]["independent_indices"]), target_config)
    report = {"states": len(states), "routes": sorted({s["metadata"]["route_id"] for s in states}),
              "schema_version": 2, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cooperation_config": target_config,
              "input_sha256": {s["path"]: hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() for s in states},
              "metadata_sha256": {str(Path(s["path"]).with_suffix(".json")): hashlib.sha256(Path(s["path"]).with_suffix(".json").read_bytes()).hexdigest() for s in states},
              "opportunity_states": sum(s["objective"]["metrics"]["coop_opportunity"] for s in states),
              "no_feasible_states": sum(not s["objective"]["feasible_mask"].any() for s in states),
              "by_route": {route: {"states": len(sub), "opportunity_rate": float(np.mean([s["objective"]["metrics"]["coop_opportunity"] for s in sub]))} for route in sorted({s["metadata"]["route_id"] for s in states}) for sub in [[s for s in states if s["metadata"]["route_id"] == route]]}}
    if args.output_dir.exists():
        raise ValueError("use a new output directory")
    source_checkpoint = None
    if args.mode != "stats":
        if args.backbone_ckpt is None:
            raise ValueError("--backbone_ckpt must be the exact collection checkpoint")
        checksum = hashlib.sha256(args.backbone_ckpt.read_bytes()).hexdigest()
        if checksum != states[0]["metadata"]["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA256 differs from collection cache")
        source_checkpoint = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=False)
        source_state = source_checkpoint["state_dict"]
        scorer_cfg = dict(states[0]["metadata"]["scorer_config"])
        scorer = Scorer(**scorer_cfg).to(args.device)
        head = {k[len("joint_head."):]: v for k, v in source_state.items() if k.startswith("joint_head.")}
        if head:
            scorer.load_state_dict(head, strict=True)
        report["verification_max_error"] = verify(scorer, states, args.tolerance)
        if args.mode == "fit":
            for key in ("base_temperature", "lambda_delta", "residual_scale_floor", "geometry_horizon", "residual_scale_mode", "fixed_residual_scale"):
                if getattr(args, key) is not None:
                    scorer_cfg[key] = getattr(args, key)
            new_scorer = Scorer(**scorer_cfg).to(args.device)
            new_scorer.load_state_dict(scorer.state_dict(), strict=True)
            scorer = new_scorer
            # Re-anchor the fixed-data diagnostic at the initial calibrated
            # scorer. This is explicitly NOT an on-policy CARLA training run.
            for state in states:
                with torch.no_grad():
                    state["reference_logits"] = forward(scorer, state).joint_logits.detach()
            train, val, split = split_states(states, args.seed, args.overfit)
            report["split"] = split
            report["initial_train"] = metrics(scorer, train)
            report["initial_validation"] = metrics(scorer, val)
            best = report["initial_validation"]["regret"]
            best_weights = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
            optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-5)
            history = []
            usable = [s for s in train if args.objective == "legacy" or s["objective"]["feasible_mask"].any()]
            if not usable:
                raise ValueError("no feasible training states; inspect quality constraints and data before fitting")
            for step in range(1, args.steps + 1):
                optimizer.zero_grad()
                chosen = random.choices(usable, k=min(args.batch_size, len(usable)))
                loss_value = 0.
                for state in chosen:
                    output = forward(scorer, state)
                    if args.objective == "cooperation":
                        score = state["objective"]["target_score"]
                        advantage = state["objective"]["advantage"]
                    else:
                        q = state["data"]["joint_q"]
                        score = (q - q.mean()) / (q.std() + 1e-5)
                        advantage = score
                    target = torch.as_tensor(score, dtype=torch.float32, device=args.device)[None]
                    adv = torch.as_tensor(advantage, dtype=torch.float32, device=args.device)[None]
                    if args.loss == "distill":
                        loss = losses.soft_target_loss(output.joint_logits, target, output.valid_pair_mask, args.temperature)
                    else:
                        loss, _ = losses.joint_dual_clip_loss(output.joint_logits, state["reference_logits"], adv,
                            output.valid_pair_mask, pair_weighting="old_policy" if args.loss == "old_policy" else "uniform")
                        if args.loss == "rift_aux":
                            loss = loss + args.auxiliary_weight * losses.soft_target_loss(output.joint_logits, target, output.valid_pair_mask, args.temperature)
                    (loss / len(chosen)).backward()
                    loss_value += float(loss.detach()) / len(chosen)
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.)
                optimizer.step()
                if step == 1 or step % 25 == 0 or step == args.steps:
                    validation = metrics(scorer, val)
                    history.append({"step": step, "loss": loss_value, **validation})
                    if validation["regret"] < best:
                        best = validation["regret"]
                        best_weights = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
            scorer.load_state_dict(best_weights)
            report.update(final_train=metrics(scorer, train), final_validation=metrics(scorer, val), history=history,
                          scorer_config=scorer_cfg, selection_metric="cooperation target regret on development routes",
                          note="Frozen cached-feature diagnostic. Does not test Pluto re-forward, CARLA, on-policy RL or BeTop semantic validity.")
    args.output_dir.mkdir(parents=True)
    if args.mode == "fit":
        merged = {k: v for k, v in source_checkpoint["state_dict"].items() if not k.startswith("joint_head.")}
        merged.update({"joint_head." + k: v for k, v in best_weights.items()})
        torch.save({"state_dict": merged, "scorer_config": scorer_cfg,
                    "backbone_sha256": states[0]["metadata"]["checkpoint_sha256"],
                    "cooperation_config": target_config,
                    "reward_config": states[0]["metadata"].get("reward_config", {}),
                    "offline_diagnostic": True}, args.output_dir / "joint_head.ckpt")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("input_sha256", "history", "by_route")}, indent=2))


if __name__ == "__main__":
    main()
