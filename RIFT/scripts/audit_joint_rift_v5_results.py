"""Read V5 evidence ZIPs without extraction; audit gates, support and outcomes."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def load_objective():
    path = Path(__file__).resolve().parents[1] / "rift/cbv/planning/fine_tuner/rlft/joint_rift/cooperation_objective.py"
    spec = importlib.util.spec_from_file_location("v5_audit_objective", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cooperation_objective


def describe(values):
    a = np.asarray(list(values), dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())} if len(a) else {"count": 0}


def best(matrix, mask):
    return tuple(int(v) for v in np.unravel_index(np.where(mask, matrix, -np.inf).argmax(), matrix.shape)) if mask.any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_zip", type=Path, required=True)
    parser.add_argument("--results_zip", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    objective = load_objective()
    rows, sweeps = [], collections.defaultdict(list)
    with zipfile.ZipFile(args.cache_zip) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".npz")):
            meta = json.loads(archive.read(name[:-4] + ".json"))
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                d = {k: data[k] for k in data.files}
            d["valid_mask"] = d["valid_pair_mask"].astype(bool)
            b = tuple(meta["independent_indices"])
            cfg = meta["cooperation_config"]
            o = objective(d, b, cfg)
            f, hard, h = o["feasible_mask"], None, o["interaction"]
            v = d["valid_mask"]
            pair = v & ~d["pair_collision"].astype(bool)
            collision = pair & ~d["unary_collision_1"][:, None] & ~d["unary_collision_2"][None, :]
            hard = collision & ~d["unary_offroad_1"][:, None] & ~d["unary_offroad_2"][None, :]
            unary = v & (d["unary_return_1"][:, None] >= d["unary_return_1"][b[0]]) & (d["unary_return_2"][None, :] >= d["unary_return_2"][b[1]])
            one = np.zeros_like(v)
            one[b[0], :] = True
            one[:, b[1]] = True
            oracle = o["oracle"]
            base = d["base_logits"]
            scale = max(float(base[v].std()), meta["scorer_config"]["residual_scale_floor"])
            needed = (base[b] - base) / (2 * scale)
            improve = f & (h >= h[b] + cfg["min_gain"])
            residual = h - h.mean(axis=0)[None, :] - h.mean(axis=1)[:, None] + h.mean()
            p1, p2 = d["p1"], d["p2"]
            q, u = d["joint_q"], d["unary_return_1"][:, None] + d["unary_return_2"][None, :]
            qo = best(q, v)
            probability = np.exp(base - np.max(base[v])); probability[~v] = 0; probability /= probability.sum()
            row = {"file": name, "route": str(meta["route_id"]), "rep": meta["repetition_index"],
                   "step": meta["time_step"], "pair_ids": json.dumps(meta["pair_ids"]),
                   "base_i": b[0], "base_j": b[1], "oracle_i": oracle[0] if oracle else -1,
                   "oracle_j": oracle[1] if oracle else -1,
                   "valid_pairs": int(v.sum()), "pair_safe_pairs": int(pair.sum()),
                   "collision_free_pairs": int(collision.sum()), "hard_safe_pairs": int(hard.sum()),
                   "unary_budget_pairs": int(unary.sum()), "feasible_pairs": int(f.sum()),
                   "independent_feasible": int(f[b]), "opportunity": int(o["metrics"]["coop_opportunity"]),
                   "gain": o["metrics"]["coop_gain"], "base_h": float(h[b]),
                   "max_p1": float(p1.max()), "max_p2": float(p2.max()),
                   "max_cg": float(d["coalition_gain"][v].max()),
                   "max_cg_hard": float(d["coalition_gain"][hard].max()) if hard.any() else None,
                   "h_std": float(h[v].std()), "u_std": float(u[v].std()),
                   "h_nonadditive_std": float(residual.std()),
                   "feasible_h_span": float(np.ptp(h[f])) if f.any() else 0.,
                   "both_p_active": int(p1.max() > 1e-8 and p2.max() > 1e-8),
                   "max_h_gain_any": float(h[v].max() - h[b]),
                   "max_h_gain_hard": float(h[hard].max() - h[b]) if hard.any() else None,
                   "one_change_opportunity": int((improve & one).any()),
                   "needs_both_changes": int(improve.any() and not (improve & one).any()),
                   "best_one_h": float(h[f & one].max()) if (f & one).any() else None,
                   "oracle_lambda_lower_bound": float(needed[oracle]) if oracle else None,
                   "any_opportunity_lambda_lower_bound": float(needed[improve].min()) if improve.any() else None,
                   "opportunity_prior_mass": float(probability[improve].sum()),
                   "feasible_prior_mass": float(probability[f].sum()),
                   "q_gain": float(q[qo] - q[b]), "q_unary_gain": float(u[qo] - u[b]),
                   "q_h_gain": float(h[qo] - h[b]),
                   "q_collision_gain": float(-20 * (int(d["pair_collision"][qo]) - int(d["pair_collision"][b]))),
                   "oracle_u1_gain": float(d["unary_return_1"][oracle[0]] - d["unary_return_1"][b[0]]) if oracle else None,
                   "oracle_u2_gain": float(d["unary_return_2"][oracle[1]] - d["unary_return_2"][b[1]]) if oracle else None,
                   "oracle_cg": float(d["coalition_gain"][oracle]) if oracle else None}
            rows.append(row)
            configs = {"baseline": {}, "no_unary_budget": {"unary_budget": 1e9},
                       "no_unary_collision_gate": {"require_unary_collision_free": False},
                       "no_offroad_gate": {"require_offroad_free": False},
                       "strict_both": {"require_both": True}, "comfortable": {"require_comfortable": True},
                       **{"budget_" + str(x): {"unary_budget": x} for x in (1, 2, 5, 10)},
                       **{"gain_" + str(x): {"min_gain": x} for x in (.01, .05, .2)}}
            for label, changes in configs.items():
                result = objective(d, b, dict(cfg, **changes))
                sweeps[label].append(result["metrics"])
    cache_summary = {"states": len(rows), "routes": sorted({r["route"] for r in rows}),
        "by_route": {route: {"states": len(sub), "opportunities": sum(r["opportunity"] for r in sub),
                            "reps": sorted({r["rep"] for r in sub}), "max_cg": max(r["max_cg"] for r in sub)}
                     for route in sorted({r["route"] for r in rows}) for sub in [[r for r in rows if r["route"] == route]]},
        "counts": {key: sum(r[key] > 1e-8 for r in rows) for key in ("pair_safe_pairs", "collision_free_pairs", "hard_safe_pairs", "feasible_pairs", "independent_feasible", "opportunity", "both_p_active", "max_cg", "h_std", "h_nonadditive_std", "one_change_opportunity", "needs_both_changes", "feasible_h_span")},
        "descriptive": {key: describe(r[key] for r in rows if r[key] is not None) for key in ("feasible_pairs", "gain", "max_cg", "h_std", "u_std", "h_nonadditive_std", "feasible_h_span", "feasible_prior_mass", "q_gain", "q_unary_gain", "q_h_gain", "q_collision_gain")},
        "opportunities": [r for r in rows if r["opportunity"]],
        "sweeps": {label: {"has_feasible": sum(r["coop_has_feasible"] for r in vals),
                           "opportunities": sum(r["coop_opportunity"] for r in vals),
                           "oracle_both": sum(r["coop_oracle_nominal_both"] for r in vals),
                           "gain_mean": float(np.mean([r["coop_gain"] for r in vals]))} for label, vals in sweeps.items()}}
    closed, fits, integrity, simulations, provided = {}, {}, {}, {}, {}
    with zipfile.ZipFile(args.results_zip) as archive:
        for name in archive.namelist():
            if name.endswith(".py"):
                path = Path(__file__).resolve().parents[2] / name
                integrity[name] = "identical" if path.is_file() and path.read_bytes() == archive.read(name) else "different_or_absent"
            if name.endswith("summary.json") and any(part in name for part in ("/fit_", "/overfit_")):
                fits[name] = json.loads(archive.read(name))
            if name.endswith("vs_independent.json"):
                provided[name] = json.loads(archive.read(name))
            if name.endswith("simulation_results.json"):
                data = json.loads(archive.read(name))
                records = data["_checkpoint"]["records"]
                simulations[name] = {"records": len(records), "progress": data["_checkpoint"].get("progress"),
                    "statuses": dict(collections.Counter(r["status"] for r in records)),
                    "scores": {k: float(np.mean([r["scores"][k] for r in records])) if records else None for k in ("score_composed", "score_route", "score_penalty")},
                    "cbv_collisions": sum(r["meta"]["cbv_collision_count"] for r in records),
                    "per_record": [{"route": r["route_id"], "status": r["status"], "scores": r["scores"], "meta": r["meta"], "infractions": r["infractions"]} for r in records]}
            if not name.endswith("joint_rift_eval_metrics.jsonl"):
                continue
            records = [json.loads(line) for line in archive.read(name).decode().splitlines() if line.strip()]
            if not records:
                continue
            groups = collections.defaultdict(list)
            for r in records:
                groups[str(r["route_id"])].append(r)
            fields = ("coop_opportunity", "coop_has_feasible", "coop_feasible_rate", "coop_gain", "coop_executed_feasible", "coop_executed_gain", "coop_nominal_both_rate", "independent_same_as_joint_rate")
            opportunities = [r for r in records if r["coop_opportunity"]]
            changed = [r for r in records if r["executed_indices"] != r["independent_indices"]]
            intervention = [r for r in records if r.get("cooperation_oracle_intervention")]
            closed[name] = {"states": len(records), "routes": len(groups),
                "state_mean": {k: float(np.mean([r[k] for r in records])) for k in fields},
                "route_equal_mean": {k: float(np.mean([np.mean([r[k] for r in sub]) for sub in groups.values()])) for k in fields},
                "counts": {"opportunity": len(opportunities), "changed": len(changed), "interventions": len(intervention),
                           "interventions_no_positive_h_gain": sum(r["coop_executed_gain"] <= 1e-8 for r in intervention),
                           "interventions_negative_h_gain": sum(r["coop_executed_gain"] < -1e-8 for r in intervention),
                           "opportunity_selected_feasible_positive": sum(r["coop_executed_feasible"] and r["coop_executed_gain"] > 1e-8 for r in opportunities),
                           "p1_and_p2_std_nonzero": sum(r["reward_scale_coalition_std"] > 1e-8 for r in records),
                           "cg_both_nonzero": sum(r["both_necessary_rate"] > 1e-8 for r in records)},
                "same_state_policy_vs_independent": {
                    "independent_feasible": sum(r["coop_independent_feasible"] for r in records),
                    "policy_feasible": sum(r["coop_selected_feasible"] for r in records),
                    "quality_recoveries": sum(not r["coop_independent_feasible"] and bool(r["coop_selected_feasible"]) for r in records),
                    "quality_regressions": sum(bool(r["coop_independent_feasible"]) and not r["coop_selected_feasible"] for r in records),
                    "opportunities_captured": sum(bool(r["coop_selected_feasible"]) and r["coop_selected_gain"] > 1e-8 for r in opportunities),
                    "opportunities_captured_gain_at_least_0_1": sum(bool(r["coop_selected_feasible"]) and r["coop_selected_gain"] >= .1 for r in opportunities),
                    "policy_h_gain_mean": float(np.mean([r["coop_selected_gain"] for r in records]))},
                "by_route": {route: {"states": len(sub), "opportunity": sum(r["coop_opportunity"] for r in sub),
                            "changed": sum(r["executed_indices"] != r["independent_indices"] for r in sub),
                            **{k: float(np.mean([r[k] for r in sub])) for k in fields}} for route, sub in groups.items()},
                "changed_examples": [{k: r.get(k) for k in ("route_id", "repetition_index", "time_step", "independent_indices", "policy_indices", "executed_indices", "coop_executed_gain", "coop_executed_feasible", "coop_opportunity", "coop_gain", "coop_independent_feasible")} for r in changed[:30]]}
    fit_brief = {}
    for name, d in fits.items():
        fit_brief[name] = {k: d[k] for k in ("initial_train", "final_train", "initial_validation", "final_validation", "split")}
        fit_brief[name]["best_logged_step"] = min(d["history"], key=lambda r: r["regret"])["step"]
        fit_brief[name]["last_logged"] = d["history"][-1]
        fit_brief[name]["history_sha256"] = hashlib.sha256(json.dumps(d["history"], sort_keys=True).encode()).hexdigest()
    result = {"input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.cache_zip, args.results_zip)},
              "source_integrity": integrity, "cache": cache_summary, "closed_loop": closed, "fits": fit_brief,
              "simulations": simulations, "provided_comparisons": provided}
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"cache_counts": cache_summary["counts"], "sweeps": cache_summary["sweeps"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
