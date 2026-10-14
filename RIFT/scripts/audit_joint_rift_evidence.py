"""Recompute Joint-RIFT mechanisms without CARLA, Lightning, or pickle.

Reads ZIP members in memory; never extracts or executes archived code.
Run from the workspace root (see --help). Results are descriptive, not
independent trials: consecutive states and candidate pairs are correlated.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from collections import Counter
import zipfile

import numpy as np


def additive_residual(values, mask):
    """Least-squares row+column residual on exactly the supplied support.

    Complete Cartesian support uses double centering. Safe support can have
    holes and requires least squares, not zero filling or double centering.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape or not mask.any():
        raise ValueError("values and nonempty mask must share [G1, G2]")
    if not np.isfinite(values[mask]).all():
        raise ValueError("nonfinite valid values")
    rows, cols = np.where(mask)
    active_rows, active_cols = mask.any(1), mask.any(0)
    sub = values[np.ix_(active_rows, active_cols)]
    if mask[np.ix_(active_rows, active_cols)].all():
        residual = sub - sub.mean(1, keepdims=True) - sub.mean(0, keepdims=True) + sub.mean()
        residual = residual.ravel()
    else:
        design = np.zeros((len(rows), sum(values.shape)))
        design[np.arange(len(rows)), rows] = 1.0
        design[np.arange(len(rows)), values.shape[0] + cols] = 1.0
        target = values[mask]
        residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    variance = float(values[mask].var())
    ratio = float(np.mean(residual ** 2) / variance) if variance > 1e-12 else 0.0
    return residual, ratio


def argmax_pair(values, mask):
    if not mask.any():
        raise ValueError("empty support")
    return tuple(int(x) for x in np.unravel_index(np.where(mask, values, -np.inf).argmax(), values.shape))


def audit_state(arrays, metadata, scales, pressure_weight=1.0, coalition_weight=1.0, collision_weight=20.0):
    q = arrays["joint_q"]
    mask = arrays["valid_pair_mask"].astype(bool)
    collision = arrays["pair_collision"].astype(bool)
    base = arrays["base_logits"].astype(np.float64)
    delta = arrays["joint_residual"].astype(np.float64)
    unary = arrays["unary_return_1"][:, None] + arrays["unary_return_2"][None, :]
    pressure = pressure_weight * arrays["p12"] + coalition_weight * arrays["coalition_gain"]
    rebuilt = unary + pressure - collision_weight * collision
    independent = argmax_pair(base, mask)
    policy = argmax_pair(arrays["joint_logits"], mask)
    oracle = argmax_pair(q, mask)
    for key, selected in (("independent_indices", independent), ("policy_indices", policy), ("oracle_indices", oracle)):
        if key in metadata and tuple(metadata[key]) != selected:
            raise ValueError("array/metadata selection mismatch: " + key)
    q_residual, nfr_q = additive_residual(q, mask)
    h_residual, nfr_h = additive_residual(pressure, mask)
    c_residual, nfr_c = additive_residual(collision.astype(float), mask)
    _, nfr_delta = additive_residual(delta, mask)
    base_std = float(base[mask].std())
    gap = float(base[independent] - base[oracle])
    diagnostics = metadata.get("diagnostics", {})
    result = {
        "route_id": str(metadata["route_id"]), "repetition": metadata["repetition_index"],
        "step": metadata["time_step"], "valid_pairs": int(mask.sum()),
        "independent": list(independent), "policy": list(policy), "oracle": list(oracle),
        "q_reconstruction_max_error": float(abs(q[mask] - rebuilt[mask]).max()),
        "logit_reconstruction_max_error": float(abs(arrays["joint_logits"][mask] - (base + delta)[mask]).max()),
        "policy_equals_independent": policy == independent,
        "policy_equals_oracle": policy == oracle,
        "q_gain": float(q[oracle] - q[independent]),
        "unary_gain_at_q_oracle": float(unary[oracle] - unary[independent]),
        "pressure_gain_at_q_oracle": float(pressure[oracle] - pressure[independent]),
        "collision_gain_at_q_oracle": float(-collision_weight * (float(collision[oracle]) - float(collision[independent]))),
        "nfr_q": nfr_q, "nfr_pressure": nfr_h, "nfr_delta": nfr_delta, "nfr_collision": nfr_c,
        "q_nonadditive_minus_collision_max_error": float(abs(q_residual + collision_weight * c_residual).max()),
        "pressure_nonadditive_rms": float(np.sqrt(np.mean(h_residual ** 2))),
        "unary_std": float(unary[mask].std()), "pressure_std": float(pressure[mask].std()),
        "coalition_std": float(arrays["coalition_gain"][mask].std()),
        "pair_collision_rate": float(collision[mask].mean()),
        "policy_collision": bool(collision[policy]), "oracle_collision": bool(collision[oracle]),
        "oracle_both_necessary": diagnostics.get("oracle_both_necessary"),
        "base_std": base_std, "oracle_base_gap": gap,
        # Necessary only: a flexible bounded residual could beat the base argmax.
        # This does not assert simultaneous realizability by the actual network.
        "lambda_lower_bound_for_oracle": gap / (2 * base_std) if base_std > 1e-12 else (0.0 if gap == 0 else None),
        "oracle_base_probability": float(np.exp(base[oracle] - (np.log(np.exp(base[mask] - base[mask].max()).sum()) + base[mask].max()))),
        "current_hv_eligible": bool(diagnostics.get("oracle_joint_gain", -np.inf) >= 10 and nfr_q >= .01 and not collision[oracle] and collision[mask].mean() <= .05),
        "learned_q_gain": float(q[policy] - q[independent]),
        "learned_pressure_gain": float(pressure[policy] - pressure[independent]),
        "independent_collision": bool(collision[independent]),
    }
    safe = mask & ~collision
    result["safe_pairs"] = int(safe.sum())
    if safe.any():
        safe_q = argmax_pair(q, safe)
        safe_unary = argmax_pair(unary, safe)
        result.update({
            "safe_oracle": list(safe_q), "safe_unary_oracle": list(safe_unary),
            "safe_joint_over_safe_unary_q_gain": float(q[safe_q] - q[safe_unary]),
            "safe_joint_over_safe_unary_pressure_gain": float(pressure[safe_q] - pressure[safe_unary]),
            "safe_pressure_nfr": additive_residual(pressure, safe)[1],
        })
        # A Full-Q optimum can equal the unary optimum while other safe
        # candidates still offer a better interaction/quality tradeoff.
        no_unary_loss = safe & (unary >= unary[independent] - 1e-10)
        each_unary_no_loss = (
            safe
            & (arrays["unary_return_1"][:, None] >= arrays["unary_return_1"][independent[0]] - 1e-10)
            & (arrays["unary_return_2"][None, :] >= arrays["unary_return_2"][independent[1]] - 1e-10)
        )
        for label, support in (("no_unary_sum_loss", no_unary_loss), ("each_unary_no_loss", each_unary_no_loss)):
            result["safe_pressure_gain_" + label] = float(pressure[support].max() - pressure[independent]) if support.any() else None
        pressure_target = no_unary_loss & (pressure > pressure[independent] + 1e-6)
        pressure_target_0_1 = no_unary_loss & (pressure > pressure[independent] + .1)
        bound_permits = (base[independent] - base) < 2 * base_std
        result["pressure_target_bound_permits_lambda_1"] = bool((pressure_target & bound_permits).any())
        result["pressure_target_gain_gt_0_1"] = bool(pressure_target_0_1.any())
        result["pressure_target_gain_gt_0_1_bound_permits_lambda_1"] = bool((pressure_target_0_1 & bound_permits).any())
        result["learned_safe_pressure_gain_no_unary_sum_loss"] = bool(no_unary_loss[policy] and pressure_target[policy])
        # Exactly the current evaluator's nominal both-necessary definition:
        # min(P12-P1, P12-P2) = P12-max(P1,P2) = coalition_gain.
        both = safe & (arrays["coalition_gain"] > .1)
        result["safe_nominal_both_pairs_epsilon_0_1"] = int(both.sum())
        both_no_loss = both & no_unary_loss
        result["safe_nominal_both_no_unary_sum_loss_pairs"] = int(both_no_loss.sum())
        result["safe_pressure_gain_both_no_unary_sum_loss"] = float(pressure[both_no_loss].max() - pressure[independent]) if both_no_loss.any() else None
    sweep = []
    for alpha in scales:
        selected = argmax_pair(base + alpha * delta, mask)
        sweep.append({
            "route_id": result["route_id"], "step": result["step"], "alpha": alpha,
            "selected": list(selected), "changed": selected != independent,
            "oracle_hit": selected == oracle, "q_gain": float(q[selected] - q[independent]),
            "pressure_gain": float(pressure[selected] - pressure[independent]),
            "collision": bool(collision[selected]),
        })
    return result, sweep


def dense_audit(path, scales, weights):
    states, sweep = [], []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npz") or "/candidate_dumps/" not in name:
                continue
            metadata = json.loads(archive.read(name[:-4] + ".json"))
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                result, state_sweep = audit_state(data, metadata, scales, *weights)
            result["archive_member"] = name
            states.append(result)
            sweep.extend(state_sweep)
    if not states:
        raise ValueError("no candidate dumps found in " + str(path))
    mean_keys = ["nfr_q", "nfr_pressure", "nfr_delta", "q_gain", "unary_gain_at_q_oracle", "pressure_gain_at_q_oracle", "collision_gain_at_q_oracle", "pair_collision_rate", "safe_joint_over_safe_unary_q_gain", "lambda_lower_bound_for_oracle"]
    summary = {
        "input": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "states": len(states), "routes": sorted({s["route_id"] for s in states}),
        "means": {k: float(np.mean([s[k] for s in states if s.get(k) is not None])) for k in mean_keys},
        "counts": {k: sum(bool(s[k]) for s in states) for k in ["policy_equals_independent", "policy_equals_oracle", "policy_collision", "oracle_collision", "current_hv_eligible"]},
        "pressure_nonadditive_states_gt_1e_8": sum(s["pressure_nonadditive_rms"] > 1e-8 for s in states),
        "states_with_safe_pairs": sum(s["safe_pairs"] > 0 for s in states),
        "safe_q_oracle_equals_safe_unary_oracle": sum(s.get("safe_oracle") == s.get("safe_unary_oracle") for s in states if s["safe_pairs"] > 0),
        "oracle_exceeds_lambda_1_bound_states": sum(s["lambda_lower_bound_for_oracle"] > 1 for s in states if s["lambda_lower_bound_for_oracle"] is not None),
        "reconstruction_max_error": max(s["q_reconstruction_max_error"] for s in states),
        "logit_reconstruction_max_error": max(s["logit_reconstruction_max_error"] for s in states),
        "scale_sweep": [{"alpha": a, "changed_states": sum(r["changed"] for r in sweep if r["alpha"] == a), "oracle_hits": sum(r["oracle_hit"] for r in sweep if r["alpha"] == a), "mean_q_gain": float(np.mean([r["q_gain"] for r in sweep if r["alpha"] == a])), "mean_pressure_gain": float(np.mean([r["pressure_gain"] for r in sweep if r["alpha"] == a])), "collision_states": sum(r["collision"] for r in sweep if r["alpha"] == a)} for a in scales],
        "opportunity_counts": {
            "full_q_oracle_gain_gt_1e_6": sum(s["q_gain"] > 1e-6 for s in states),
            "learned_q_gain_gt_1e_6": sum(s["learned_q_gain"] > 1e-6 for s in states),
            "learned_q_gain_lt_minus_1e_6": sum(s["learned_q_gain"] < -1e-6 for s in states),
            "independent_safe_states": sum(not s["independent_collision"] for s in states),
            "independent_safe_with_pressure_gain_no_unary_sum_loss": sum(not s["independent_collision"] and s.get("safe_pressure_gain_no_unary_sum_loss") is not None and s["safe_pressure_gain_no_unary_sum_loss"] > 1e-6 for s in states),
            "safe_nominal_both_states_epsilon_0_1": sum(s.get("safe_nominal_both_pairs_epsilon_0_1", 0) > 0 for s in states),
            "safe_nominal_both_and_no_unary_sum_loss_states": sum(s.get("safe_nominal_both_no_unary_sum_loss_pairs", 0) > 0 for s in states),
            "safe_nominal_both_pressure_gain_and_no_unary_sum_loss_states": sum(s.get("safe_pressure_gain_both_no_unary_sum_loss") is not None and s["safe_pressure_gain_both_no_unary_sum_loss"] > 1e-6 for s in states),
            "independent_safe_with_pressure_target_bound_permitted_lambda_1": sum(not s["independent_collision"] and s.get("pressure_target_bound_permits_lambda_1", False) for s in states),
            "independent_safe_with_pressure_gain_gt_0_1": sum(not s["independent_collision"] and s.get("pressure_target_gain_gt_0_1", False) for s in states),
            "independent_safe_with_pressure_gain_gt_0_1_bound_permitted_lambda_1": sum(not s["independent_collision"] and s.get("pressure_target_gain_gt_0_1_bound_permits_lambda_1", False) for s in states),
            "independent_safe_learned_pressure_gain_no_unary_sum_loss": sum(not s["independent_collision"] and s.get("learned_safe_pressure_gain_no_unary_sum_loss", False) for s in states),
        },
        "limitations": ["Consecutive states are correlated; no inferential confidence interval is computed.", "Q oracle is a virtual evaluator optimum, not closed-loop cooperation.", "Saved arrays lack raw Pluto features, candidate latent and executed rollout traces; exact scorer refitting is unavailable. The nominal both-necessary proxy can be recovered from coalition_gain > epsilon, but separate p1/p2 marginals and real counterfactuals are unavailable.", "Alpha sweep rescales the existing residual; it is not retraining lambda_delta.", "Safe support residual is a least-squares descriptive projection; feasibility masks alone can couple otherwise additive policies.", "Safe means predicted CBV-CBV collision-free only; a nondecreasing aggregate unary return is not a guarantee that each safety/comfort constraint is met."],
    }
    return summary, states, sweep


def holdout_audit(path):
    results = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if "/holdout10_" not in name or not name.endswith("joint_rift_eval_metrics.jsonl"):
                continue
            records = [json.loads(line) for line in archive.read(name).splitlines() if line.strip()]
            numeric = ["independent_same_as_joint_rate", "joint_same_as_oracle_rate", "oracle_both_necessary", "oracle_pair_collision", "policy_pair_collision", "oracle_joint_gain", "reward_scale_unary_return_std", "reward_scale_p12_std", "reward_scale_coalition_std"]
            result = {"run": name.split("/outputs/")[1].split("/")[0], "joint_states": len(records)}
            result.update({key: float(np.mean([r[key] for r in records if key in r])) if any(key in r for r in records) else None for key in numeric})
            result["by_route"] = {route: {"states": len(sub), "policy_differs_from_independent_rate": float(np.mean([1-r["independent_same_as_joint_rate"] for r in sub])), "oracle_both_necessary_rate": float(np.mean([r["oracle_both_necessary"] for r in sub]))} for route in sorted({str(r["route_id"]) for r in records}) for sub in [[r for r in records if str(r["route_id"]) == route]]}
            results.append(result)
    return results


def formal_result_audit(path):
    """Match actual route/repetition keys; retain outcome and quality metrics."""
    runs = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if "/holdout10_" not in name or not name.endswith("simulation_results.json"):
                continue
            run = name.split("/outputs/")[1].split("/")[0]
            payload = json.loads(archive.read(name))["_checkpoint"]
            records = payload["records"]
            indexed = {r["route_id"]: r for r in records}
            if len(indexed) != len(records):
                raise ValueError("duplicate route/repetition key in " + run)
            runs[run] = {"global": payload["global_record"]["scores_mean"], "records": indexed}
    pairs, independent_drift = [], []
    for run in sorted(runs):
        if "_policy_seed" not in run:
            continue
        control = run.replace("_policy_seed", "_independent_seed")
        if control not in runs or runs[run]["records"].keys() != runs[control]["records"].keys():
            raise ValueError("unmatched routes: " + run)
        for key, policy in runs[run]["records"].items():
            independent = runs[control]["records"][key]
            row = {"run": run, "route_repetition": key,
                   "policy_status": policy["status"], "independent_status": independent["status"]}
            for metric in ("score_composed", "score_route", "score_penalty"):
                row[metric + "_delta"] = policy["scores"][metric] - independent["scores"][metric]
            for metric in ("cbv_collision_count", "cbv_off_road_game_time", "cbv_uncomfortable_game_time", "cbv_total_game_time", "duration_game"):
                row[metric + "_delta"] = policy["meta"][metric] - independent["meta"][metric]
            pairs.append(row)
    for seed in range(4):
        uniform = runs.get("holdout10_uniform_independent_seed" + str(seed))
        hv4 = runs.get("holdout10_hv4_independent_seed" + str(seed))
        if uniform and hv4:
            independent_drift.append({"seed": seed, "hv4_independent_minus_uniform_independent_score": hv4["global"]["score_composed"] - uniform["global"]["score_composed"]})
    summary = {
        "runs": {run: {"records": len(data["records"]), "scores": data["global"],
                       "statuses": dict(Counter(r["status"] for r in data["records"].values()))}
                 for run, data in sorted(runs.items())},
        "independent_control_drift": independent_drift,
        "interpretation": "Same-seed independent runs should be an A/A control if frozen model tensors and initial states are identical. Score differences alone cannot distinguish simulator variation, setup drift, or a frozen-model regression. Verify tensor hashes and action traces.",
    }
    return summary, pairs


def write_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in record.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense_zip", type=Path, required=True)
    parser.add_argument("--formal_zip", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", type=float, default=[0, .5, 1, 2, 5, 10])
    parser.add_argument("--pressure_weight", type=float, default=1)
    parser.add_argument("--coalition_weight", type=float, default=1)
    parser.add_argument("--collision_weight", type=float, default=20)
    args = parser.parse_args()
    weights = (args.pressure_weight, args.coalition_weight, args.collision_weight)
    summary, states, sweep = dense_audit(args.dense_zip, args.scales, weights)
    # Fail before writing if the assumed configuration is inconsistent.
    if summary["reconstruction_max_error"] > 1e-6:
        raise ValueError("Q does not match configured reward weights; inspect run configuration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary["reward_weights"] = dict(zip(["pressure", "coalition", "collision"], weights))
    if args.formal_zip:
        summary["formal_archive_sha256"] = hashlib.sha256(args.formal_zip.read_bytes()).hexdigest()
        summary["formal_holdout_runs"] = holdout_audit(args.formal_zip)
        summary["formal_results"], route_comparisons = formal_result_audit(args.formal_zip)
        write_csv(args.output_dir / "formal_route_comparisons.csv", route_comparisons)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "states.csv", states)
    write_csv(args.output_dir / "scale_sweep.csv", sweep)
    print(json.dumps({k: summary[k] for k in ["states", "routes", "means", "counts", "pressure_nonadditive_states_gt_1e_8", "reconstruction_max_error", "scale_sweep"]}, indent=2))


if __name__ == "__main__":
    main()
