"""Audit complete V6 runs and compare teachers without pooling partial runs."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import re


def summarize(directory):
    spec = importlib.util.spec_from_file_location("v6_outcomes", Path(__file__).with_name("compare_joint_v5_outcomes.py"))
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    original_values = compare.values
    def extended_values(record):
        result = original_values(record)
        infractions = record.get("infractions")
        for label in ("vehicle", "pedestrian", "layout"):
            result["ego_" + label + "_collision_count"] = (float(len(infractions.get("collisions_" + label, []))) if infractions is not None else None)
        result["ego_blocked"] = float("blocked" in record["status"].lower())
        result["ego_timed_out"] = float("timed out" in record["status"].lower())
        return result
    compare.values = extended_values
    runs, complete = {}, collections.defaultdict(list)
    for result_path in sorted(directory.rglob("simulation_results.json")):
        run = result_path.parents[2]
        match = re.fullmatch(r"(.+)_seed(\d+)", run.name)
        if not match:
            continue
        mode, seed = match.group(1), int(match.group(2))
        payload = json.loads(result_path.read_text(encoding="utf-8"))["_checkpoint"]
        progress = payload.get("progress", [])
        is_complete = bool(len(progress) == 2 and progress[0] == progress[1] and progress[1] > 0 and len(payload["records"]) == progress[1])
        metrics_path = result_path.with_name("joint_rift_eval_metrics.jsonl")
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.is_file() else []
        fields = ("decision_has_feasible", "decision_quality_repair", "decision_interaction_support", "decision_joint_support",
                  "decision_bilateral_support", "decision_executed_changed", "decision_executed_feasible", "decision_executed_gain_over_quality", "decision_executed_gain_over_factorized")
        if rows and any(any(k not in r for k in fields) for r in rows):
            raise ValueError("missing V6 decision diagnostics: " + str(metrics_path))
        groups = collections.defaultdict(list)
        for row in rows:
            groups[str(row["route_id"])].append(row)
        per_route = {key: {"joint_states": len(sub), **{k: sum(r[k] for r in sub) / len(sub) for k in fields}} for key, sub in groups.items()}
        runs[str(result_path)] = {"mode": mode, "seed": seed, "complete": is_complete, "progress": progress,
            "episode_records": len(payload["records"]), "statuses": dict(collections.Counter(r["status"] for r in payload["records"])),
            "joint_states": len(rows), "joint_routes": len(groups), "per_route": per_route,
            "route_equal_mean": {k: sum(r[k] for r in per_route.values()) / len(per_route) for k in fields} if per_route else None}
        if is_complete:
            complete[mode].append(result_path)
    comparisons = {}
    for first, second in (("policy", "independent"), ("verified_policy", "independent"), ("interaction_oracle", "quality_oracle"),
                          ("joint_oracle", "factorized_oracle"), ("joint_oracle", "quality_oracle"),
                          ("independent_aa", "independent")):
        if first in complete and second in complete:
            comparisons[first + "_vs_" + second] = compare.compare(compare.load(complete[first]), compare.load(complete[second]))
    return {"runs": runs, "comparisons": comparisons,
            "interpretation": "Only complete runs enter matched episode comparisons. State diagnostics describe each policy's own visited distribution; zero-joint routes are not imputed as safe or unsafe. Oracle improvement is not automatically learned cooperation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.run_dir)
    if not report["runs"]:
        raise ValueError("no simulation result files found")
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"runs": len(report["runs"]), "complete": sum(r["complete"] for r in report["runs"].values()),
                      "comparisons": list(report["comparisons"]), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
