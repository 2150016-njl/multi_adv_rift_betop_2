"""Matched CARLA outcomes with route-cluster bootstrap, conditional on trained models."""

import argparse
import json
from pathlib import Path
import re

import numpy as np


def load(paths):
    result = {}
    for path in paths:
        matches = re.findall(r"seed(\d+)", str(path))
        if not matches:
            raise ValueError("result path must identify evaluation seed: " + str(path))
        seed = int(matches[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))["_checkpoint"]
        for record in payload["records"]:
            key = (seed, record["route_id"])
            if key in result:
                raise ValueError("duplicate seed/route/repetition record")
            result[key] = record
    return result


def values(record):
    meta = record["meta"]
    total = float(meta["cbv_total_game_time"])
    return {**{key: float(record["scores"][key]) for key in ("score_composed", "score_route", "score_penalty")},
            "cbv_collision_count": float(meta["cbv_collision_count"]),
            "cbv_offroad_fraction": float(meta["cbv_off_road_game_time"]) / total if total > 0 else None,
            "cbv_uncomfortable_fraction": float(meta["cbv_uncomfortable_game_time"]) / total if total > 0 else None}


def compare(policy, independent, bootstrap=2000, seed=0):
    if not policy or policy.keys() != independent.keys():
        raise ValueError("require identical nonempty evaluation seed/route/repetition keys")
    rows = []
    for key in sorted(policy):
        p, b = values(policy[key]), values(independent[key])
        rows.append({"seed": key[0], "route_repetition": key[1], "route": re.sub(r"_rep\d+$", "", key[1]),
                     "policy_status": policy[key]["status"], "independent_status": independent[key]["status"],
                     **{k: p[k] - b[k] if p[k] is not None and b[k] is not None else None for k in p}})
    rng = np.random.RandomState(seed)
    report = {}
    for metric in values(next(iter(policy.values()))):
        routes = sorted({row["route"] for row in rows if row[metric] is not None})
        means = np.array([np.mean([r[metric] for r in rows if r["route"] == route and r[metric] is not None]) for route in routes])
        item = {"routes": len(routes), "missing_pairs": sum(r[metric] is None for r in rows),
                "route_equal_mean_delta": float(means.mean()) if len(means) else None}
        if len(means) >= 2:
            distribution = means[rng.randint(0, len(means), size=(bootstrap, len(means)))].mean(-1)
            item["route_cluster_bootstrap_95_percent"] = [float(x) for x in np.quantile(distribution, [.025, .975])]
        report[metric] = item
    return {"matched_records": len(rows), "metrics": report, "per_record": rows,
            "interpretation": "All deltas are policy minus independent. Higher Ego driving score means better Ego outcome, not automatically better CBV cooperation. Bootstrap resamples route clusters retaining their evaluation seeds/repetitions; conditional on these fixed trained models, not training-seed generalization. Descriptive interval for the sampled route population; simulator A/A variation still needs measurement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", nargs="+", type=Path, required=True)
    parser.add_argument("--independent", nargs="+", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    report = compare(load(args.policy), load(args.independent))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_record"}, indent=2))


if __name__ == "__main__":
    main()
