"""V6 teacher audit and nominal-conditioned reward distillation, no simulator imports."""
import argparse
import collections
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "rift/cbv/planning/fine_tuner/rlft/joint_rift"


def load_module(name):
    spec = importlib.util.spec_from_file_location("v6_" + name, MODULES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


teacher_module = load_module("decision_teacher")
head_module = load_module("nominal_direct_head")


def relabel_semantics(states, semantics):
    temporal = load_module("temporal_interaction")
    braid = load_module("nominal_braid")
    maximum_error = 0.
    for s in states:
        d, m = s["data"], s["metadata"]
        if any(k not in d for k in ("rollout_center_1", "rollout_center_2", "ego_nominal_aligned_global")):
            raise ValueError("semantic replay needs V6 PID rollout centers and aligned nominal; recollect")
        dtype = torch.as_tensor(d["raw_trajectories_1"]).dtype
        ego = torch.as_tensor(d["ego_nominal_aligned_global"], dtype=dtype)
        reward = m.get("reward_config", {})
        old = {k: np.array(d[k], copy=True) for k in ("p1", "p2", "p12", "coalition_gain")}
        timing = m.get("evaluation_config", {"gamma": .98, "dt": .1})
        weights = np.power(float(timing["gamma"]), np.arange(len(ego)))
        for axis in (1, 2):
            centers = torch.as_tensor(d["rollout_center_" + str(axis)], dtype=dtype)
            if semantics == "forward_braid":
                events = braid.nominal_braid_events(centers, ego, ego[0, 2], reward.get("braid_radius", 20.), reward.get("braid_distance_tau", 5.))
                d["interaction_sign_" + str(axis)] = torch.nn.functional.pad(events.signed_onset, (0, 1)).numpy()
            else:
                events = temporal.get_temporally_weighted_interaction_events(centers, ego,
                    dt=float(timing["dt"]), arrival_time_tau=float(reward.get("arrival_time_tau", 1.)))
                d["interaction_sign_" + str(axis)] = np.zeros(centers.shape[:2])
            trace = torch.nn.functional.pad(events.weighted_onset, (0, 1)).numpy()
            d["interaction_trace_" + str(axis)] = trace
            d["p" + str(axis)] = (trace * weights).sum(-1)
        union = 1. - (1. - d["interaction_trace_1"][:, None]) * (1. - d["interaction_trace_2"][None])
        d["p12"] = (union * weights).sum(-1)
        d["coalition_gain"] = d["p12"] - np.maximum(d["p1"][:, None], d["p2"][None])
        if semantics == reward.get("interaction_semantics", "spatial_temporal"):
            maximum_error = max(maximum_error, *(float(np.max(abs(d[k] - old[k]))) for k in old))
        m["reward_config"] = dict(reward, interaction_semantics=semantics)
    if maximum_error > 5e-5:
        raise ValueError("same-semantics PID replay failed: " + str(maximum_error))
    return maximum_error


def read_states(path):
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            states = []
            for name in sorted(n for n in archive.namelist() if n.endswith(".npz")):
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as payload:
                    data = {k: payload[k] for k in payload.files}
                states.append({"path": name, "metadata": json.loads(archive.read(name[:-4] + ".json")), "data": data})
    else:
        states = []
        for file in sorted(path.rglob("state_*.npz")):
            with np.load(file, allow_pickle=False) as payload:
                data = {k: payload[k] for k in payload.files}
            states.append({"path": str(file), "metadata": json.loads(file.with_suffix(".json").read_text(encoding="utf-8")), "data": data})
    if not states or any(s["metadata"].get("schema_version", 0) < 2 for s in states):
        raise ValueError("requires nonempty schema-v2 head-input cache")
    return states


def prepare(states, config, unary_budget=None):
    signatures = set()
    for s in states:
        m, d = s["metadata"], s["data"]
        d["valid_mask"] = d["valid_pair_mask"].astype(bool)
        qcfg = dict(m.get("cooperation_config", {}))
        if unary_budget is not None:
            qcfg["unary_budget"] = unary_budget
        s["quality_config"] = qcfg
        signatures.add((m.get("backbone_sha256", m.get("checkpoint_sha256")),
                        json.dumps(m.get("reward_config", {}), sort_keys=True), json.dumps(qcfg, sort_keys=True),
                        json.dumps(m.get("evaluation_config", {"gamma": .98, "horizon": 40, "dt": .1}), sort_keys=True)))
        s["baseline"] = tuple(m["independent_indices"])
        s["teacher"] = teacher_module.decision_teacher(d, s["baseline"], d["base_logits"], qcfg, config)
    if len(signatures) != 1:
        raise ValueError("cache aggregation requires one frozen backbone, reward semantics and quality configuration")
    return next(iter(signatures))[0]


def forward(model, s):
    device = next(model.parameters()).device
    def tensor(key):
        return torch.as_tensor(s["data"][key], dtype=torch.float32, device=device)[None]
    args = []
    for axis in (1, 2):
        args.extend(tensor(k) for k in ("candidate_features_" + str(axis), "cbv_" + str(axis) + "_common_trajectories", "raw_trajectories_" + str(axis), "unary_logits_" + str(axis)))
    valid = s["data"]["valid_mask"]
    return model(*args, ego_nominal=tensor("ego_nominal_common"),
                 valid_mask_1=torch.as_tensor(valid.any(1), device=device)[None],
                 valid_mask_2=torch.as_tensor(valid.any(0), device=device)[None])


def evaluate(model, states, config, teacher_kind):
    rows = []
    with torch.no_grad():
        for s in states:
            output = forward(model, s)
            selected = head_module.select_direct(output)
            t, b = s["teacher"], s["baseline"]
            feasible, score = t["feasible_mask"], t["score"]
            base_ok, ok = bool(feasible[b]), bool(feasible[selected])
            d = s["data"]
            interaction = bool(ok and t["metrics"]["decision_interaction_support"] and score[selected] - score[t["quality"]] >= config.get("min_gain", .05) - 1e-8)
            joint = bool(ok and t["metrics"]["decision_joint_support"]
                and score[selected] - score[t["factorized"]] >= config.get("min_joint_gain", .01) - 1e-8
                and score[selected] - score[t["quality"]] >= config.get("min_gain", .05) - 1e-8
                and d["p1"][selected[0]] >= config.get("min_member_pressure", .01)
                and d["p2"][selected[1]] >= config.get("min_member_pressure", .01)
                and d["coalition_gain"][selected] >= config.get("min_coalition", .01))
            rows.append({"path": s["path"], "selected": list(selected), "teacher": list(t[teacher_kind]),
                         "feasible": ok, "quality_recovery": not base_ok and ok,
                         "quality_regression": base_ok and not ok,
                         "teacher_agreement": selected == t[teacher_kind],
                         "interaction_captured": interaction, "joint_captured": joint,
                         "interaction_support": bool(t["metrics"]["decision_interaction_support"]),
                         "joint_support": bool(t["metrics"]["decision_joint_support"]),
                         "changed": selected != b, "score_gain_over_quality": float(score[selected] - score[t["quality"]])})
    counts = {k: sum(r[k] for r in rows) for k in ("feasible", "quality_recovery", "quality_regression", "teacher_agreement", "interaction_captured", "joint_captured", "interaction_support", "joint_support", "changed")}
    utility = counts["quality_recovery"] - 2 * counts["quality_regression"] + 3 * counts["interaction_captured"] + 5 * counts["joint_captured"]
    return {"states": len(rows), "counts": counts, "development_selection_utility": utility,
            "interaction_capture_rate": counts["interaction_captured"] / counts["interaction_support"] if counts["interaction_support"] else None,
            "joint_capture_rate": counts["joint_captured"] / counts["joint_support"] if counts["joint_support"] else None,
            "per_state": rows}


def balanced_groups(states):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in states:
        t, m = s["teacher"], s["metadata"]
        if t["metrics"]["decision_joint_support"]:
            label = "joint"
        elif t["metrics"]["decision_interaction_support"]:
            label = "interaction"
        elif t["metrics"]["decision_quality_repair"]:
            label = "quality"
        elif not t["feasible_mask"].any():
            label = "no_feasible"
        else:
            label = "preserve"
        segment = (str(m["route_id"]), m.get("evaluation_seed", 0), m.get("repetition_index", -1),
                   tuple(m.get("pair_ids", [])), m.get("checkpoint_sha256"))
        groups[label][segment].append(s)
    return groups


def supervised_loss(output, state, kind):
    device = output.joint_logits.device
    mask = output.valid_pair_mask
    valid = mask[0]
    feasible = torch.as_tensor(state["teacher"]["feasible_mask"], dtype=torch.bool, device=device)
    q = F.binary_cross_entropy_with_logits(output.quality_logits[0][valid], feasible[valid].float(), reduction="none")
    positive = feasible[valid]
    parts = [q[positive].mean()] if positive.any() else []
    if (~positive).any():
        parts.append(q[~positive].mean())
    quality_loss = torch.stack(parts).mean()
    target = state["teacher"][kind]
    target_flat = target[0] * valid.shape[1] + target[1]
    action_loss = F.cross_entropy(output.joint_logits.flatten(1), torch.tensor([target_flat], device=device))
    pressure = torch.as_tensor(state["teacher"]["score"], dtype=torch.float32, device=device)
    pressure_loss = F.smooth_l1_loss(output.interaction[0][valid], pressure[valid])
    for axis in (1, 2):
        actual = torch.as_tensor(state["data"]["p" + str(axis)], dtype=torch.float32, device=device)
        pressure_loss = pressure_loss + F.smooth_l1_loss(getattr(output, "member_pressure_" + str(axis))[0], actual)
    return action_loss + 2 * quality_loss + 10 * pressure_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("stats", "fit"))
    p.add_argument("--cache", type=Path, required=True, help="cache directory or evidence ZIP")
    p.add_argument("--additional_cache", action="append", default=[], help="Append student-visited cache for dataset aggregation; same frozen backbone/reward required")
    p.add_argument("--config", type=Path, help="Restore teacher thresholds from prepared eval.yaml")
    p.add_argument("--semantics_override", choices=("forward_braid", "spatial_temporal"), help="Replay labels on recorded PID rollouts; old V5 caches are insufficient")
    p.add_argument("--verify_semantics", action="store_true", help="Verify same-semantics replay on V6 PID cache before other analysis")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--backbone_ckpt", type=Path)
    p.add_argument("--resume_head", type=Path)
    p.add_argument("--head_only", action="store_true", help="Fit cached inputs without loading backbone; prepare merges this small export with the verified server backbone")
    p.add_argument("--teacher", choices=("quality", "factorized", "interaction", "joint"), default="joint")
    p.add_argument("--head_kind", choices=("joint", "unary_guarded"), default="joint")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overfit_opportunities", action="store_true")
    p.add_argument("--validation_routes", default="")
    p.add_argument("--unary_budget", type=float)
    p.add_argument("--min_gain", type=float, default=.05)
    p.add_argument("--min_joint_gain", type=float, default=.01)
    p.add_argument("--min_member_pressure", type=float, default=.01)
    p.add_argument("--min_coalition", type=float, default=.01)
    p.add_argument("--quality_threshold", type=float, default=.8)
    p.add_argument("--nominal_conflict_policy", choices=("reject", "label_only"), default=None)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("positive optimizer settings required")
    if args.output_dir.exists():
        raise ValueError("use a fresh output directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    config = {k: getattr(args, k) for k in ("min_gain", "min_joint_gain", "min_member_pressure", "min_coalition", "nominal_conflict_policy")}
    config.update(enabled=True, pressure_weight=1., coalition_weight=1.)
    config["nominal_conflict_policy"] = args.nominal_conflict_policy or "reject"
    if args.config:
        config.update(yaml.safe_load(args.config.read_text(encoding="utf-8"))["decision_teacher"])
    if args.nominal_conflict_policy is not None:
        config["nominal_conflict_policy"] = args.nominal_conflict_policy
    states = read_states(args.cache)
    inputs = [str(args.cache.resolve())] + [str(Path(v).resolve()) for v in args.additional_cache]
    if len(inputs) != len(set(inputs)):
        raise ValueError("duplicate cache inputs")
    for extra_cache in args.additional_cache:
        states.extend(read_states(Path(extra_cache)))
    replay_error = None
    if args.verify_semantics:
        original = {s["metadata"].get("reward_config", {}).get("interaction_semantics", "spatial_temporal") for s in states}
        if len(original) != 1:
            raise ValueError("verify each original label semantics separately")
        replay_error = relabel_semantics(states, next(iter(original)))
    if args.semantics_override:
        relabel_semantics(states, args.semantics_override)
    backbone_hash = prepare(states, config, args.unary_budget)
    groups = balanced_groups(states)
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "decision_teacher_config": config, "states": len(states),
              "routes": sorted({str(s["metadata"]["route_id"]) for s in states}),
              "strata": {k: {"states": sum(len(v) for v in group.values()), "segments": len(group)} for k, group in groups.items()},
              "by_route": {route: {"states": len(sub), "interaction_support": sum(s["teacher"]["metrics"]["decision_interaction_support"] for s in sub),
                                     "joint_support": sum(s["teacher"]["metrics"]["decision_joint_support"] for s in sub)}
                           for route in sorted({str(s["metadata"]["route_id"]) for s in states})
                           for sub in [[s for s in states if str(s["metadata"]["route_id"]) == route]]},
              "scope": "Frozen-generator teacher distillation; no on-policy PPO or closed-loop guarantee."}
    report["same_semantics_replay_max_error"] = replay_error
    collision_audit = {"comparable_exact_sat_states": 0, "states_with_aabb_false_positive": 0, "candidate_snapshot_false_positives": 0}
    for state in states:
        if state["metadata"].get("reward_config", {}).get("collision_geometry") != "exact_sat" or "collision_aabb_ego_trace_1" not in state["data"]:
            continue
        collision_audit["comparable_exact_sat_states"] += 1
        affected = False
        for axis in (1, 2):
            d = state["data"]
            key = "collision_aabb_ego_trace_" + str(axis)
            if key not in d:
                continue
            aabb = (d[key] | d["collision_aabb_background_trace_" + str(axis)]).any(-1)
            exact = (d["collision_ego_trace_" + str(axis)] | d["collision_background_trace_" + str(axis)]).any(-1)
            false_positive = aabb & ~exact
            collision_audit["candidate_snapshot_false_positives"] += int(false_positive.sum())
            affected |= bool(false_positive.any())
        collision_audit["states_with_aabb_false_positive"] += int(affected)
    report["collision_audit"] = collision_audit
    if not collision_audit["comparable_exact_sat_states"]:
        collision_audit["states_with_aabb_false_positive"] = None
        collision_audit["candidate_snapshot_false_positives"] = None
    if args.mode == "fit":
        if not args.head_only and (args.backbone_ckpt is None or hashlib.sha256(args.backbone_ckpt.read_bytes()).hexdigest() != backbone_hash):
            raise ValueError("backbone checkpoint must match the cache's frozen backbone hash")
        if args.overfit_opportunities:
            key = "joint" if args.teacher == "joint" else "interaction"
            train = [s for s in states if s["teacher"]["metrics"]["decision_" + key + "_support"]]
            if not train:
                raise ValueError("no requested opportunity support; audit/recollect before fitting")
            val, split = train, {"mode": "explicit_overfit_no_generalization"}
        else:
            routes = report["routes"].copy()
            rng.shuffle(routes)
            val_routes = set(args.validation_routes.split(",")) if args.validation_routes else set(routes[max(1, int(len(routes) * .8)):])
            if args.teacher in {"joint", "interaction"}:
                support_key = "decision_" + args.teacher + "_support"
                support_routes = {str(s["metadata"]["route_id"]) for s in states if s["teacher"]["metrics"][support_key]}
                if len(support_routes) < 2:
                    raise ValueError("requested opportunity occurs in fewer than two routes; use labelled overfit or collect more routes")
                if not args.validation_routes and not val_routes.intersection(support_routes):
                    val_routes.add(next(r for r in routes if r in support_routes))
                if not val_routes.intersection(support_routes) or not support_routes.difference(val_routes):
                    raise ValueError("need requested opportunity support in both route-disjoint train and validation")
            if not val_routes or not val_routes.issubset(set(routes)):
                raise ValueError("need nonempty validation routes contained in the cache")
            train = [s for s in states if str(s["metadata"]["route_id"]) not in val_routes]
            val = [s for s in states if str(s["metadata"]["route_id"]) in val_routes]
            if not train or not val:
                raise ValueError("need nonempty route-disjoint train and validation sets")
            split = {"mode": "route", "validation_routes": sorted(val_routes),
                     "train_routes": sorted({str(s["metadata"]["route_id"]) for s in train}),
                     "development_support_stratification": not bool(args.validation_routes)}
        if args.teacher == "joint" and not any(s["teacher"]["metrics"]["decision_joint_support"] for s in train):
            raise ValueError("NO JOINT SUPPORT: do not train a quality-repair model and call it joint cooperation; collect semantic opportunities or use --teacher interaction as a labelled control")
        head_cfg = dict(candidate_dim=states[0]["data"]["candidate_features_1"].shape[-1], hidden_dim=64,
                        trajectory_frames=40, trajectory_stride=5, prior_weight=.05,
                        quality_threshold=args.quality_threshold, min_gain=config["min_gain"], head_kind=args.head_kind)
        model = head_module.NominalDirectScorer(**head_cfg).to(args.device)
        if args.resume_head:
            previous = torch.load(args.resume_head, map_location="cpu", weights_only=False)
            if previous.get("backbone_sha256") != backbone_hash or previous.get("scorer_config") != model.config:
                raise ValueError("resume head/backbone configuration mismatch")
            model.load_state_dict({k[len("joint_head."):]: v for k, v in previous["state_dict"].items() if k.startswith("joint_head.")})
        report.update(split=split, scorer_config=model.config,
                      initial_train=evaluate(model, train, config, args.teacher), initial_validation=evaluate(model, val, config, args.teacher))
        best = report["initial_validation"]["development_selection_utility"]
        best_weights, best_step = copy.deepcopy(model.state_dict()), 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        strata = balanced_groups(train)
        history = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            value = 0.
            for _ in range(args.batch_size):
                group = strata[rng.choice(sorted(strata))]
                subset = group[rng.choice(list(group))]
                state = rng.choice(subset)
                loss = supervised_loss(forward(model, state), state, args.teacher) / args.batch_size
                loss.backward()
                value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                metrics = evaluate(model, val, config, args.teacher)
                item = {"step": step, "loss": value, **{k: v for k, v in metrics.items() if k != "per_state"}}
                history.append(item)
                # Selection is development-only and explicitly values repair,
                # regressions and opportunity capture, not pooled-frame regret.
                if metrics["development_selection_utility"] > best:
                    best, best_step = metrics["development_selection_utility"], step
                    best_weights = copy.deepcopy(model.state_dict())
                print(json.dumps(item), flush=True)
        model.load_state_dict(best_weights)
        report.update(history=history, best_step=best_step,
                      final_train=evaluate(model, train, config, args.teacher), final_validation=evaluate(model, val, config, args.teacher))
        source = {} if args.head_only else torch.load(args.backbone_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        merged = {k: v for k, v in source.items() if not k.startswith("joint_head.")}
        merged.update({"joint_head." + k: v.detach().cpu() for k, v in best_weights.items()})
        args.output_dir.mkdir(parents=True)
        torch.save({"state_dict": merged, "scorer_config": model.config, "backbone_sha256": backbone_hash,
                    "decision_teacher_config": config, "cooperation_config": states[0]["quality_config"],
                    "reward_config": states[0]["metadata"].get("reward_config", {}), "v6_distillation": True,
                    "teacher_kind": args.teacher, "selection_step": best_step, "head_only": args.head_only}, args.output_dir / "joint_head.ckpt")
    else:
        args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"history", "initial_train", "initial_validation", "final_train", "final_validation"}}, indent=2))


if __name__ == "__main__":
    main()
